"""F 3D — composant (a1-3D) : F JETABLE 3D, transposition du motif E4a d'un
axe (pré-enregistrement `claude/prereg-cout-f-3d-2026-08-03.md`, commit
`661237a`, écrit AVANT ce fichier).

RÔLE, ÉCRIT AVANT LE CODE ET NON NÉGOCIABLE APRÈS : ce module est la
RÉFÉRENCE de motif du kernel fusionné 3D — il sert au test d'équivalence
(I-2 du prereg) et **ne porte AUCUN chiffre de verdict**. Le rapport ρ ne
se lit qu'entre deux kernels FUSIONNÉS de la même famille ; comparer un
fusionné 2D à un naïf 3D reproduirait à l'envers la faute du run 1 de la
tranche (mesurer la lenteur du code de session, pas la structure).

CE QUI CHANGE PAR RAPPORT À `substrat_jetable.py` — les trois seules
choses, et elles SONT la question (§3 du prereg) :
  1. **trois axes de flux** au lieu de deux (il n'y a pas de 3D à deux
     directions) ;
  2. **cinq champs par système** (h, hu, hv, **hw**, s) au lieu de quatre —
     on ne fait pas de 3D avec deux composantes d'impulsion ; le troisième
     moment est intrinsèque à la dimension ;
  3. **deux tangentielles par axe** au lieu d'une — conséquence mécanique
     de (1)+(2) : chaque face transporte deux composantes tangentielles.

CE QUI NE CHANGE PAS : padding réfléchissant (miroir + négation de la qdm
NORMALE, sur trois paires de faces), désingularisation, pentes minmod,
réconciliation positivité, reconstruction MUSCL, flux HLL dry-aware,
transport tangentiel upwind, corrections de pression hydrostatiques
(calculées, nulles à b ≡ 0), divergence, plancher sec, 2 étages SSP-RK2,
réduction CFL payée non consommée, dt figé, f32 de bout en bout. Les
constantes sont IMPORTÉES du jetable 2D — jamais redéfinies : un écart de
constante ferait de ρ un rapport entre deux physiques, pas entre deux
dimensions.

LA MINORATION INTERDITE D'AVANCE (prereg §3) : garder 4 champs en 3D (pas
de `hw`) allégerait le stencil et rendrait ρ complaisant. Le précédent est
gravé — `substrat_jetable.py:6-8` « l'alternative "3+5 passifs" minorerait
le coût et fabriquerait un M-a complaisant ».

CE QUE CE F N'EST PAS : un schéma eau 3D (dette ouverte, arbitrage Romain) ;
de l'incompressible (aucun solve de Poisson, aucune pression globale) ; une
validation numérique (aucune exigence de fidélité). C'est un motif de COÛT,
comme son aîné 2D — `substrat_jetable.py:5-6` « doit être représentatif en
COÛT (stencil, c=8 champs), pas en physique-jeu »."""
from __future__ import annotations

import numpy as np

from src.f1_gpu.substrat_jetable import (
    CFL_JETABLE,
    DRY_EPS,
    DT_JETABLE,
    GRAVITE,
    _desing,
    _flux_hll,
    _minmod,
)

# NOTE DE DISCIPLINE — `_desing`, `_minmod` et `_flux_hll` sont IMPORTÉS,
# jamais recopiés. Ils sont purement élémentaires (aucun axe, aucune
# shape) : rien en eux n'est 2D. Une première rédaction de ce module avait
# ré-écrit `_flux_hll` « parce que le module d'origine est 2D » — motif
# faux, et exactement la faute dont la sonde v1 est morte (deux copies qui
# divergent). Corrigé avant tout run.

CHAMPS_PAR_SYSTEME_3D: int = 5      # (h, hu, hv, hw, s) — E4a transposé
AXES_3D: tuple[int, ...] = (-3, -2, -1)     # (z, y, x)

# Indice de champ portant l'impulsion NORMALE à chaque axe, et les deux
# TANGENTIELLES. Convention unique du module, relue par le kernel fusionné
# et par les tests — un désaccord ici serait une physique différente, pas
# une implémentation différente.
#   axe -1 (x) : normale hu (1), tangentielles (hv=2, hw=3)
#   axe -2 (y) : normale hv (2), tangentielles (hu=1, hw=3)
#   axe -3 (z) : normale hw (3), tangentielles (hu=1, hv=2)
NORMALE_PAR_AXE: dict[int, int] = {-1: 1, -2: 2, -3: 3}
TANGENTIELLES_PAR_AXE: dict[int, tuple[int, int]] = {
    -1: (2, 3), -2: (1, 3), -3: (1, 2)}


def etat_initial_jetable_3d(n_fenetres: int, n_systemes: int, n: int,
                            graine: int) -> np.ndarray:
    """État initial (B, S, 5, n, n, n) float32 CPU : h = 1 + relief lissé
    basse fréquence (seedé), impulsions nulles, s petit champ positif
    lissé. Même intention que l'aîné 2D : lisse ⇒ le jetable reste borné
    sur la série ; le contenu n'est PAS l'objet de la mesure."""
    rng = np.random.default_rng(graine)
    m = max(n // 8, 1) + 1
    grossier = rng.standard_normal((n_fenetres, n_systemes, 2, m, m, m))
    facteur = -(-n // m)                    # plafond : couvre n
    lisse = np.kron(grossier, np.ones((1, 1, 1, facteur, facteur, facteur)))
    lisse = lisse[..., :n, :n, :n]
    q = np.zeros((n_fenetres, n_systemes, CHAMPS_PAR_SYSTEME_3D, n, n, n),
                 dtype=np.float32)
    q[:, :, 0] = 1.0 + 0.05 * lisse[:, :, 0]
    q[:, :, 4] = 0.05 * np.abs(lisse[:, :, 1])
    return q


def _tranche(arr, axe: int, debut=None, fin=None):
    """`arr[..., debut:fin, ...]` le long de `axe` — le découpage des
    autres axes reste plein. Un seul point de vérité pour tout le module :
    trois axes écrits à la main auraient trois façons de se tromper."""
    index = [slice(None)] * arr.ndim
    index[axe] = slice(debut, fin)
    return arr[tuple(index)]


def _pad_reflexif_3d(q, xp):
    """Cellules fantômes réfléchissantes sur (..., 5, n, n, n) ->
    (..., 5, n+2, n+2, n+2) : edge partout, composante NORMALE de la qdm
    négée à chacune des trois paires de murs (motif `_pad_reflexif`
    étendu d'un axe)."""
    pad = [(0, 0)] * (q.ndim - 3) + [(1, 1), (1, 1), (1, 1)]
    qp = xp.pad(q, pad, mode="edge")
    for axe, champ in NORMALE_PAR_AXE.items():
        # Le champ vit sur l'axe -4 une fois le padding posé ; on négocie
        # le mur par tranche le long de `axe`.
        bas = [slice(None)] * qp.ndim
        haut = [slice(None)] * qp.ndim
        source_bas = [slice(None)] * qp.ndim
        source_haut = [slice(None)] * qp.ndim
        for liste in (bas, haut, source_bas, source_haut):
            liste[-4] = champ
        bas[axe], source_bas[axe] = 0, 1
        haut[axe], source_haut[axe] = -1, -2
        qp[tuple(bas)] = -qp[tuple(source_bas)]
        qp[tuple(haut)] = -qp[tuple(source_haut)]
    return qp


def _pentes(arr, xp, axe: int):
    """Pentes limitées minmod le long de `axe` sur un tableau padé."""
    s = xp.zeros_like(arr)
    interieur = [slice(None)] * arr.ndim
    interieur[axe] = slice(1, -1)
    s[tuple(interieur)] = _minmod(
        _tranche(arr, axe, 1, -1) - _tranche(arr, axe, None, -2),
        _tranche(arr, axe, 2, None) - _tranche(arr, axe, 1, -1), xp)
    return s


def _divergence_axe(qp, xp, axe: int):
    """Contributions de l'axe `axe` à L(q), pour les cellules RÉELLES.

    Retourne `(dh, dn, dt1, dt2, ds)` de shape (B, S, n, n, n) : `dn` est
    la divergence de l'impulsion NORMALE à l'axe, `dt1`/`dt2` celles des
    deux TANGENTIELLES dans l'ordre de `TANGENTIELLES_PAR_AXE` — c'est le
    seul endroit du module où l'ordre compte, et l'appelant le relit là."""
    g = GRAVITE
    normale = NORMALE_PAR_AXE[axe]
    tang1, tang2 = TANGENTIELLES_PAR_AXE[axe]

    hp = qp[:, :, 0]
    bp = xp.zeros_like(hp)              # bathymétrie plate MATÉRIALISÉE
    etap = hp + bp
    unp = _desing(hp, qp[:, :, normale], xp)
    ut1p = _desing(hp, qp[:, :, tang1], xp)
    ut2p = _desing(hp, qp[:, :, tang2], xp)
    csp = _desing(hp, qp[:, :, 4], xp)

    # `qp` porte le champ sur l'axe -4 ; une fois les composantes
    # extraites, les axes spatiaux sont les trois derniers — `axe` reste
    # donc valide tel quel (indexation négative).
    seta = _pentes(etap, xp, axe)
    sun = _pentes(unp, xp, axe)
    sut1 = _pentes(ut1p, xp, axe)
    sut2 = _pentes(ut2p, xp, axe)
    scs = _pentes(csp, xp, axe)
    mauvais = (((etap - 0.5 * seta) - bp < 0.0)
               | ((etap + 0.5 * seta) - bp < 0.0))
    seta = xp.where(mauvais, 0.0, seta)
    sun = xp.where(mauvais, 0.0, sun)
    sut1 = xp.where(mauvais, 0.0, sut1)
    sut2 = xp.where(mauvais, 0.0, sut2)
    scs = xp.where(mauvais, 0.0, scs)

    def face_g(champ, pente):
        return (_tranche(champ, axe, None, -1)
                + 0.5 * _tranche(pente, axe, None, -1))

    def face_d(champ, pente):
        return (_tranche(champ, axe, 1, None)
                - 0.5 * _tranche(pente, axe, 1, None))

    etaL, etaR = face_g(etap, seta), face_d(etap, seta)
    unL, unR = face_g(unp, sun), face_d(unp, sun)
    ut1L, ut1R = face_g(ut1p, sut1), face_d(ut1p, sut1)
    ut2L, ut2R = face_g(ut2p, sut2), face_d(ut2p, sut2)
    csL, csR = face_g(csp, scs), face_d(csp, scs)
    bL, bR = _tranche(bp, axe, None, -1), _tranche(bp, axe, 1, None)

    hL, hR = etaL - bL, etaR - bR
    zstar = xp.maximum(bL, bR)
    hsL = xp.maximum(0.0, etaL - zstar)
    hsR = xp.maximum(0.0, etaR - zstar)
    Fh, Fn = _flux_hll(hsL, hsL * unL, hsR, hsR * unR, xp)
    Ft1 = xp.where(Fh >= 0.0, Fh * ut1L, Fh * ut1R)
    Ft2 = xp.where(Fh >= 0.0, Fh * ut2L, Fh * ut2R)
    Fs = xp.where(Fh >= 0.0, Fh * csL, Fh * csR)
    Fn_g = Fn + 0.5 * g * (hL ** 2 - hsL ** 2)   # corrigé côté L
    Fn_d = Fn + 0.5 * g * (hR ** 2 - hsR ** 2)   # corrigé côté R

    autres = tuple(a for a in AXES_3D if a != axe)

    def interieur(arr):
        """Retire le padding des DEUX autres axes spatiaux."""
        for a in autres:
            arr = _tranche(arr, a, 1, -1)
        return arr

    def div(F):
        return interieur(_tranche(F, axe, 1, None)
                         - _tranche(F, axe, None, -1))

    dn = interieur(_tranche(Fn_g, axe, 1, None)
                   - _tranche(Fn_d, axe, None, -1))
    return div(Fh), dn, div(Ft1), div(Ft2), div(Fs)


def _rhs_jetable_3d(q, xp):
    """Opérateur spatial L(q) sur (B, S, 5, n, n, n) — trois axes, 5 champs
    par système, deux tangentielles par axe. Retourne
    (Lh, Lhu, Lhv, Lhw, Ls) sur (B, S, n, n, n)."""
    qp = _pad_reflexif_3d(q, xp)
    contributions = {axe: _divergence_axe(qp, xp, axe) for axe in AXES_3D}

    somme_h = None
    somme_s = None
    somme_qdm = {1: None, 2: None, 3: None}
    for axe, (dh, dn, dt1, dt2, ds) in contributions.items():
        somme_h = dh if somme_h is None else somme_h + dh
        somme_s = ds if somme_s is None else somme_s + ds
        tang1, tang2 = TANGENTIELLES_PAR_AXE[axe]
        for champ, apport in ((NORMALE_PAR_AXE[axe], dn),
                              (tang1, dt1), (tang2, dt2)):
            somme_qdm[champ] = (apport if somme_qdm[champ] is None
                                else somme_qdm[champ] + apport)
    return (-somme_h, -somme_qdm[1], -somme_qdm[2], -somme_qdm[3], -somme_s)


def _plancher_sec_3d(h, hu, hv, hw, s, xp):
    """Plancher sec (motif `_plancher_sec`, étendu à la 3e impulsion)."""
    h = xp.maximum(h, 0.0)
    sec = h <= DRY_EPS
    return (xp.where(sec, 0.0, h), xp.where(sec, 0.0, hu),
            xp.where(sec, 0.0, hv), xp.where(sec, 0.0, hw),
            xp.where(sec, 0.0, s))


def reduction_cfl_3d(q, xp) -> float:
    """Réduction CFL globale sur les TROIS vitesses — payée chaque frame
    pour son coût (réduction device + rapatriement d'UN scalaire), jamais
    consommée (dt figé, comme en 2D)."""
    h = q[:, :, 0]
    c = xp.sqrt(GRAVITE * xp.maximum(h, 0.0))
    u = _desing(h, q[:, :, 1], xp)
    v = _desing(h, q[:, :, 2], xp)
    w = _desing(h, q[:, :, 3], xp)
    smax = float(xp.maximum(xp.maximum(xp.abs(u) + c, xp.abs(v) + c),
                            xp.abs(w) + c).max())
    return CFL_JETABLE / max(smax, 1e-12)


def pas_f_jetable_3d(q, xp, sortie=None) -> tuple:
    """Un pas SSP-RK2 complet du F jetable 3D sur (B, S, 5, n, n, n) f32 :
    réduction CFL (coût, non consommée) + 2 étages `_rhs_jetable_3d` +
    plancher sec par étage. Signature identique à l'aîné 2D.

    Retourne (q_suivant, dt_cfl_diagnostic)."""
    if q.ndim != 6 or q.shape[2] != CHAMPS_PAR_SYSTEME_3D:
        raise ValueError(
            f"pas_f_jetable_3d : shape {q.shape} != (B, S, "
            f"{CHAMPS_PAR_SYSTEME_3D}, n, n, n) — les cinq champs "
            "(h, hu, hv, hw, s) sont load-bearing (prereg §3, minoration "
            "à quatre champs INTERDITE d'avance).")
    dt = DT_JETABLE
    dt_cfl = reduction_cfl_3d(q, xp)

    h, hu, hv, hw, s = (q[:, :, 0], q[:, :, 1], q[:, :, 2],
                        q[:, :, 3], q[:, :, 4])
    L1 = _rhs_jetable_3d(q, xp)
    e1 = xp.empty_like(q)
    etage1 = _plancher_sec_3d(h + dt * L1[0], hu + dt * L1[1],
                              hv + dt * L1[2], hw + dt * L1[3],
                              s + dt * L1[4], xp)
    for indice, champ in enumerate(etage1):
        e1[:, :, indice] = champ

    L2 = _rhs_jetable_3d(e1, xp)
    etage2 = _plancher_sec_3d(
        0.5 * h + 0.5 * (e1[:, :, 0] + dt * L2[0]),
        0.5 * hu + 0.5 * (e1[:, :, 1] + dt * L2[1]),
        0.5 * hv + 0.5 * (e1[:, :, 2] + dt * L2[2]),
        0.5 * hw + 0.5 * (e1[:, :, 3] + dt * L2[3]),
        0.5 * s + 0.5 * (e1[:, :, 4] + dt * L2[4]), xp)

    if sortie is None:
        sortie = xp.empty_like(q)
    for indice, champ in enumerate(etage2):
        sortie[:, :, indice] = champ
    return sortie, dt_cfl
