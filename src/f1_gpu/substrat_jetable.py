"""F1 tranche-1 — composant (a1) : F JETABLE c=8 f32 (choix B5/E4a).

Décision gravée T1 (§A15) : « substrat v2 étendu à c=8 champs
(coût-représentatif) » — le choix de F « doit être représentatif en COÛT
(stencil, c=8 champs), pas en physique-jeu ». E4a (§A15-complément) : « 8
champs = (h, hu, hv, s)×2 sous stencil complet — majorant honnête,
l'alternative "3+5 passifs" minorerait le coût et fabriquerait un M-a
complaisant. »

Ce module est la transcription xp (B1) du MOTIF DE COÛT du stencil réel
`src/solver_wetdry.py::_rhs_o2` (INTOUCHÉ) : mêmes passes — padding
réfléchissant, désingularisation, pentes minmod x/y sur (η, u, v, c_s),
réconciliation positivité, états d'interface, flux HLL dry-aware,
transport tangentiel upwind, corrections de pression hydrostatiques,
divergence — plus 2 étages SSP-RK2, plancher sec, et la réduction CFL
globale par frame (le moteur réel la paie : c'est une réduction device +
rapatriement scalaire). AUCUNE exigence de fidélité numérique : c'est un
F jetable ((a) ≠ (b), §A15-complément) — le portage FIDÈLE est le
composant (b), tranche-2, NON acheté.

B5 (détail, cf. `__init__.py`) :
  - état q = tableau (B, 2, 4, n, n) float32 — B fenêtres batchées (une
    batche par niveau, 3 fenêtres), 2 systèmes indépendants, 4 champs
    (h, hu, hv, s). La vectorisation batch/système ne réduit PAS le calcul
    par cellule ; elle réduit l'overhead de lancement (le moteur réel
    batcherait des fenêtres identiques) ;
  - dt FIGÉ `DT_JETABLE` (la stabilité du jetable n'est pas l'objet de la
    mesure) ; la réduction CFL est CALCULÉE chaque frame pour son coût,
    jamais consommée ;
  - bathymétrie PLATE (champ b ≡ 0 matérialisé) : z*, h*, corrections de
    pression calculés quand même — coût identique, stabilité assurée ;
  - désingularisation par `where` vectorisé (l'indexation booléenne du
    chemin CPU f64 serait un COÛT DIFFÉRENT sur GPU — le motif d'accès
    est adapté au device, le nombre de passes est conservé)."""
from __future__ import annotations

import numpy as np

DT_JETABLE: float = 1e-3   # dt figé du jetable (B5) — pas un paramètre physique
DRY_EPS: float = 1e-4      # seuil sec, valeur du solveur réel reconduite
GRAVITE: float = 9.81      # jetable (le g réel vit dans config.py, intouché)
CFL_JETABLE: float = 0.4   # borne du solveur réel, reconduite (réduction seule)


def etat_initial_jetable(n_fenetres: int, n: int, graine: int) -> np.ndarray:
    """État initial (B, 2, 4, n, n) float32 CPU : h = 1 + relief lissé
    basse fréquence (seedé), hu = hv = 0, s = petit champ positif lissé.
    Lisse => le jetable reste borné sur la série (330 frames à DT_JETABLE) ;
    le contenu n'est PAS l'objet de la mesure, seul le coût l'est."""
    rng = np.random.default_rng(graine)
    grossier = rng.standard_normal((n_fenetres, 2, 2, max(n // 8, 1) + 1,
                                    max(n // 8, 1) + 1))
    facteur = -(-n // grossier.shape[-1])  # plafond : couvre n
    lisse = np.kron(grossier, np.ones((1, 1, 1, facteur, facteur)))
    lisse = lisse[..., :n, :n]
    q = np.zeros((n_fenetres, 2, 4, n, n), dtype=np.float32)
    q[:, :, 0] = 1.0 + 0.05 * lisse[:, :, 0]
    q[:, :, 3] = 0.05 * np.abs(lisse[:, :, 1])
    return q


def _desing(h, hq, xp):
    """u = hq/h vectorisé, 0 sous le seuil sec (motif de coût de
    `desingularize_velocity`, adapté device — B5)."""
    return xp.where(h > DRY_EPS, hq / xp.maximum(h, DRY_EPS), 0.0)


def _minmod(a, b, xp):
    """Limiteur minmod (motif exact de `_minmod` du solveur réel)."""
    return xp.where(a * b <= 0.0, 0.0,
                    xp.sign(a) * xp.minimum(xp.abs(a), xp.abs(b)))


def _pentes_x(arr, xp):
    """Pentes limitées selon x sur un tableau padé (..., n+2, n+2)."""
    s = xp.zeros_like(arr)
    s[..., 1:-1] = _minmod(arr[..., 1:-1] - arr[..., :-2],
                           arr[..., 2:] - arr[..., 1:-1], xp)
    return s


def _pentes_y(arr, xp):
    """Pentes limitées selon y (axe -2) sur un tableau padé."""
    s = xp.zeros_like(arr)
    s[..., 1:-1, :] = _minmod(arr[..., 1:-1, :] - arr[..., :-2, :],
                              arr[..., 2:, :] - arr[..., 1:-1, :], xp)
    return s


def _flux_hll(hL, hqL, hR, hqR, xp):
    """Flux HLL 1D dry-aware (motif exact de `_hll_flux` du solveur réel) :
    vitesses de Davis, côté sec -> raréfaction du côté mouillé."""
    g = GRAVITE
    uL = _desing(hL, hqL, xp)
    uR = _desing(hR, hqR, xp)
    cL = xp.sqrt(g * xp.maximum(hL, 0.0))
    cR = xp.sqrt(g * xp.maximum(hR, 0.0))
    sL = xp.where(hL > DRY_EPS, xp.minimum(uL - cL, uR - cR), uR - 2.0 * cR)
    sR = xp.where(hR > DRY_EPS, xp.maximum(uL + cL, uR + cR), uL + 2.0 * cL)
    FhuL = hqL * uL + 0.5 * g * hL ** 2
    FhuR = hqR * uR + 0.5 * g * hR ** 2

    def hll(FL, FR, UL, UR):
        return xp.where(
            sL >= 0, FL,
            xp.where(sR <= 0, FR,
                     (sR * FL - sL * FR + sL * sR * (UR - UL))
                     / (sR - sL + 1e-30)))

    return hll(hqL, hqR, hL, hR), hll(FhuL, FhuR, hqL, hqR)


def _pad_reflexif(q, xp):
    """Cellules fantômes réfléchissantes sur (..., 4, n, n) -> (..., 4,
    n+2, n+2) : edge partout, composante normale de la qdm négée aux murs
    (motif de `_pad_reflective`)."""
    pad = [(0, 0)] * (q.ndim - 2) + [(1, 1), (1, 1)]
    qp = xp.pad(q, pad, mode="edge")
    qp[..., 1, :, 0] = -qp[..., 1, :, 1]
    qp[..., 1, :, -1] = -qp[..., 1, :, -2]
    qp[..., 2, 0, :] = -qp[..., 2, 1, :]
    qp[..., 2, -1, :] = -qp[..., 2, -2, :]
    return qp


def _rhs_jetable(q, xp):
    """Opérateur spatial L(q) — le motif de `_rhs_o2` (MUSCL
    surface-gradient + hydrostatique + HLL) sur (B, 2, 4, n, n), 4e champ s
    advecté sous le MÊME jeu de pentes/upwind que la qdm tangentielle
    (stencil complet, E4a). Retourne (Lh, Lhu, Lhv, Ls) sur (B, 2, n, n)."""
    g = GRAVITE
    qp = _pad_reflexif(q, xp)
    hp, hup, hvp, hsp = qp[:, :, 0], qp[:, :, 1], qp[:, :, 2], qp[:, :, 3]
    bp = xp.zeros_like(hp)          # bathymétrie plate MATÉRIALISÉE (B5)
    etap = hp + bp
    up = _desing(hp, hup, xp)
    vp = _desing(hp, hvp, xp)
    csp = _desing(hp, hsp, xp)      # concentration s/h (4e champ)

    # ----- Axe x : pentes + réconciliation positivité -----
    seta = _pentes_x(etap, xp)
    su = _pentes_x(up, xp)
    sv = _pentes_x(vp, xp)
    scs = _pentes_x(csp, xp)
    mauvais = (((etap - 0.5 * seta) - bp < 0.0)
               | ((etap + 0.5 * seta) - bp < 0.0))
    seta = xp.where(mauvais, 0.0, seta)
    su = xp.where(mauvais, 0.0, su)
    sv = xp.where(mauvais, 0.0, sv)
    scs = xp.where(mauvais, 0.0, scs)

    etaL = etap[..., :-1] + 0.5 * seta[..., :-1]
    etaR = etap[..., 1:] - 0.5 * seta[..., 1:]
    uL = up[..., :-1] + 0.5 * su[..., :-1]
    uR = up[..., 1:] - 0.5 * su[..., 1:]
    vL = vp[..., :-1] + 0.5 * sv[..., :-1]
    vR = vp[..., 1:] - 0.5 * sv[..., 1:]
    csL = csp[..., :-1] + 0.5 * scs[..., :-1]
    csR = csp[..., 1:] - 0.5 * scs[..., 1:]
    bL, bR = bp[..., :-1], bp[..., 1:]
    hL, hR = etaL - bL, etaR - bR
    zstar = xp.maximum(bL, bR)
    hsL = xp.maximum(0.0, etaL - zstar)
    hsR = xp.maximum(0.0, etaR - zstar)
    Fh_x, Fhun_x = _flux_hll(hsL, hsL * uL, hsR, hsR * uR, xp)
    Fhvt_x = xp.where(Fh_x >= 0.0, Fh_x * vL, Fh_x * vR)
    Fs_x = xp.where(Fh_x >= 0.0, Fh_x * csL, Fh_x * csR)
    Fhun_x_g = Fhun_x + 0.5 * g * (hL ** 2 - hsL ** 2)
    Fhun_x_d = Fhun_x + 0.5 * g * (hR ** 2 - hsR ** 2)

    # ----- Axe y : mêmes passes -----
    seta = _pentes_y(etap, xp)
    su = _pentes_y(up, xp)
    sv = _pentes_y(vp, xp)
    scs = _pentes_y(csp, xp)
    mauvais = (((etap - 0.5 * seta) - bp < 0.0)
               | ((etap + 0.5 * seta) - bp < 0.0))
    seta = xp.where(mauvais, 0.0, seta)
    su = xp.where(mauvais, 0.0, su)
    sv = xp.where(mauvais, 0.0, sv)
    scs = xp.where(mauvais, 0.0, scs)

    etaB = etap[..., :-1, :] + 0.5 * seta[..., :-1, :]
    etaT = etap[..., 1:, :] - 0.5 * seta[..., 1:, :]
    vB = vp[..., :-1, :] + 0.5 * sv[..., :-1, :]
    vT = vp[..., 1:, :] - 0.5 * sv[..., 1:, :]
    uB = up[..., :-1, :] + 0.5 * su[..., :-1, :]
    uT = up[..., 1:, :] - 0.5 * su[..., 1:, :]
    csB = csp[..., :-1, :] + 0.5 * scs[..., :-1, :]
    csT = csp[..., 1:, :] - 0.5 * scs[..., 1:, :]
    bB, bT = bp[..., :-1, :], bp[..., 1:, :]
    hB, hT = etaB - bB, etaT - bT
    zstar_y = xp.maximum(bB, bT)
    hsB = xp.maximum(0.0, etaB - zstar_y)
    hsT = xp.maximum(0.0, etaT - zstar_y)
    Gh_y, Ghvn_y = _flux_hll(hsB, hsB * vB, hsT, hsT * vT, xp)
    Ghut_y = xp.where(Gh_y >= 0.0, Gh_y * uB, Gh_y * uT)
    Gs_y = xp.where(Gh_y >= 0.0, Gh_y * csB, Gh_y * csT)
    Ghvn_y_b = Ghvn_y + 0.5 * g * (hB ** 2 - hsB ** 2)
    Ghvn_y_t = Ghvn_y + 0.5 * g * (hT ** 2 - hsT ** 2)

    # ----- Divergences (dx = dy = 1 côté jetable) -----
    def divx(F):
        return F[..., 1:-1, 1:] - F[..., 1:-1, :-1]

    def divy(G):
        return G[..., 1:, 1:-1] - G[..., :-1, 1:-1]

    Lh = -(divx(Fh_x) + divy(Gh_y))
    Lhu = -((Fhun_x_g[..., 1:-1, 1:] - Fhun_x_d[..., 1:-1, :-1])
            + divy(Ghut_y))
    Lhv = -(divx(Fhvt_x)
            + (Ghvn_y_b[..., 1:, 1:-1] - Ghvn_y_t[..., :-1, 1:-1]))
    Ls = -(divx(Fs_x) + divy(Gs_y))
    return Lh, Lhu, Lhv, Ls


def _plancher_sec(h, hu, hv, s, xp):
    """Plancher sec (motif de `_floor_dry`, étendu au champ s)."""
    h = xp.maximum(h, 0.0)
    sec = h <= DRY_EPS
    return (xp.where(sec, 0.0, h), xp.where(sec, 0.0, hu),
            xp.where(sec, 0.0, hv), xp.where(sec, 0.0, s))


def reduction_cfl(q, xp) -> float:
    """Réduction CFL globale (motif de `_cfl_dt`) : le moteur réel la paie
    chaque pas — réduction device + rapatriement d'UN scalaire. Calculée
    pour son COÛT ; jamais consommée par le jetable (dt figé, B5)."""
    h, hu, hv = q[:, :, 0], q[:, :, 1], q[:, :, 2]
    c = xp.sqrt(GRAVITE * xp.maximum(h, 0.0))
    u = _desing(h, hu, xp)
    v = _desing(h, hv, xp)
    smax = float(xp.maximum(xp.abs(u) + c, xp.abs(v) + c).max())
    return CFL_JETABLE / max(smax, 1e-12)


def pas_f_jetable(q, xp, sortie=None) -> tuple:
    """Un pas SSP-RK2 complet du F jetable sur (B, 2, 4, n, n) f32 :
    réduction CFL (coût, non consommée) + 2 étages `_rhs_jetable` +
    plancher sec par étage. Écrit dans `sortie` si fourni (tampon d'état
    préalloué, E4b/B2 — `sortie is q` autorisé), sinon alloue.

    Retourne (q_suivant, dt_cfl_diagnostic)."""
    dt = DT_JETABLE
    dt_cfl = reduction_cfl(q, xp)

    h, hu, hv, s = q[:, :, 0], q[:, :, 1], q[:, :, 2], q[:, :, 3]
    L1h, L1hu, L1hv, L1s = _rhs_jetable(q, xp)
    e1 = xp.empty_like(q)
    (e1[:, :, 0], e1[:, :, 1], e1[:, :, 2], e1[:, :, 3]) = _plancher_sec(
        h + dt * L1h, hu + dt * L1hu, hv + dt * L1hv, s + dt * L1s, xp)

    L2h, L2hu, L2hv, L2s = _rhs_jetable(e1, xp)
    hn, hun, hvn, sn = _plancher_sec(
        0.5 * h + 0.5 * (e1[:, :, 0] + dt * L2h),
        0.5 * hu + 0.5 * (e1[:, :, 1] + dt * L2hu),
        0.5 * hv + 0.5 * (e1[:, :, 2] + dt * L2hv),
        0.5 * s + 0.5 * (e1[:, :, 3] + dt * L2s), xp)

    if sortie is None:
        sortie = xp.empty_like(q)
    sortie[:, :, 0] = hn
    sortie[:, :, 1] = hun
    sortie[:, :, 2] = hvn
    sortie[:, :, 3] = sn
    return sortie, dt_cfl
