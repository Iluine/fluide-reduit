"""Verrous du F jouet 3D — écrits pour que le chiffre ρ ne puisse pas être
fabriqué (pré-enregistrement `claude/prereg-cout-f-3d-2026-08-03.md`,
commit `661237a`).

Ce que chaque famille de tests empêche, nommé :
  - **équivalence de motif** (I-2 du prereg) : un kernel rapide PARCE QU'IL
    OMET du travail échoue ici. C'est la garde anti-complaisance ; sans
    elle, ρ mesurerait une amputation ;
  - **isotropie** : un axe mal câblé (tangentielle inversée, normale
    décalée) passerait l'équivalence si le jetable ET le fusionné portaient
    la même erreur de convention. L'isotropie ne peut pas être fausse par
    accord — elle compare le motif à lui-même ;
  - **conservation** : les murs sont réfléchissants, donc la masse est
    exactement conservée. Une divergence mal découpée le casse ;
  - **la minoration interdite** : quatre champs en 3D (pas de `hw`)
    allégeraient le stencil et rendraient ρ complaisant. Le refus est
    testé, pas seulement écrit ;
  - **kernel 2D intouché** : ρ est un rapport ; toucher le dénominateur
    déplacerait le résultat sans qu'aucun test ne s'en plaigne.

Les tests GPU se sautent proprement sans cupy — mais un saut n'est JAMAIS
un succès : le driver de mesure re-vérifie l'équivalence avant de
chronométrer, et refuse de mesurer si elle échoue."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.f1_gpu import substrat_fusionne as fusionne_2d
from src.f1_gpu.substrat_jetable import DRY_EPS, DT_JETABLE, GRAVITE
from src.f1_gpu.substrat_jetable_3d import (
    CHAMPS_PAR_SYSTEME_3D,
    NORMALE_PAR_AXE,
    TANGENTIELLES_PAR_AXE,
    etat_initial_jetable_3d,
    pas_f_jetable_3d,
)

try:
    import cupy as cp
except Exception:                                    # pragma: no cover
    cp = None

sans_gpu = pytest.mark.skipif(cp is None, reason="cupy absent (tests VM)")

# Empreinte du kernel 2D telle que mesurée à l'ancre M-a′
# (`outputs/f1/ma_prime.json`, médiane 5.055920 ms / 3 fenêtres). Elle est
# le DÉNOMINATEUR de ρ : la figer, c'est empêcher qu'on déplace le rapport
# en retouchant la référence.
EMPREINTE_SOURCE_2D: str = (
    "e18015f57e14263414239b18c51d25cd335250f58dde2ccb892a6e2c1d6f30b4")

GRAINE: int = 909

# ÉTAT SÉVÈRE — TROUVÉ PAR MUTATION, CORRIGÉ AVANT TOUT RUN.
#
# Première rédaction de ces tests : ils comparaient UN pas depuis
# `etat_initial_jetable_3d`. Un mutant qui envoyait la tangentielle de
# l'axe z dans la MAUVAISE impulsion passait les 17 verrous. Deuxième
# rédaction : trois pas d'échauffement — le mutant passait encore.
#
# La cause, mesurée et non devinée : dans l'état jetable, `h ≈ 1 ± 0.05`
# et les impulsions restent vers `4e-3`. Or le flux tangentiel vaut
# `F_h · u_t`, donc **d'ordre u²** ≈ 1,6e-5, et sa divergence ~1e-9 après
# multiplication par `dt` — **trois ordres sous `atol = 1e-5`**. Aucun
# échauffement n'y change quoi que ce soit : le terme est invisible par
# construction. De même la réconciliation positivité ne se déclenche
# jamais (`h ± 0.5·s_η` reste franchement positif), et aucune cellule
# n'est sèche : trois branches du stencil n'étaient jamais parcourues.
#
# L'état ci-dessous les allume toutes : vitesses d'ordre 1 (les flux
# tangentiels deviennent visibles), zones SÈCHES (branches dry-aware et
# plancher sec), gradients raides (réconciliation positivité). Il ne
# change RIEN à la mesure — le driver chronomètre l'état jetable standard,
# le même que l'ancre 2D. Il change ce que les verrous peuvent voir.
GRAINE_SEVERE: int = 4242


def _etat_severe(n_fenetres: int, n_systemes: int, n: int):
    """État qui parcourt TOUTES les branches du stencil : sec/mouillé,
    positivité, transport tangentiel d'ordre 1. Le seul état sur lequel
    une comparaison de motif ait un pouvoir de détection."""
    rng = np.random.default_rng(GRAINE_SEVERE)
    forme = (n_fenetres, n_systemes, n, n, n)
    q = np.zeros((n_fenetres, n_systemes, CHAMPS_PAR_SYSTEME_3D, n, n, n),
                 dtype=np.float32)
    h = 0.6 + 1.2 * rng.random(forme)
    # Poches SÈCHES franches (~12 % des cellules) : branches dry-aware,
    # plancher sec, raréfaction du côté mouillé.
    sec = rng.random(forme) < 0.12
    h = np.where(sec, 0.0, h)
    q[:, :, 0] = h
    for champ in (1, 2, 3):
        # Impulsions d'ordre 1 et INDÉPENDANTES entre elles — deux
        # composantes corrélées masqueraient un échange de tangentielles.
        q[:, :, champ] = np.where(sec, 0.0, h * (rng.random(forme) - 0.5))
    q[:, :, 4] = np.where(sec, 0.0, h * 0.3 * rng.random(forme))
    return q


def _echanger(q, axe_a: int, axe_b: int, champ_a: int, champ_b: int):
    """Échange deux axes d'espace ET les deux impulsions correspondantes —
    l'opération avec laquelle F doit commuter s'il est isotrope."""
    r = np.swapaxes(q, axe_a, axe_b).copy()
    r[:, :, [champ_a, champ_b]] = r[:, :, [champ_b, champ_a]]
    return r


PERMUTATIONS = [(-1, -2, 1, 2, "x<->y"), (-1, -3, 1, 3, "x<->z"),
                (-2, -3, 2, 3, "y<->z")]


# --------------------------------------------------------------------
# 1. Le motif : cinq champs, trois axes, deux tangentielles
# --------------------------------------------------------------------

def test_la_convention_des_axes_est_une_partition():
    """Pour chaque axe, {normale} ∪ {tangentielles} doit être exactement
    les trois impulsions {1, 2, 3} — sans doublon ni oubli. Un doublon
    ferait compter une impulsion deux fois (majoration silencieuse), un
    oubli l'omettrait (minoration silencieuse)."""
    for axe in (-1, -2, -3):
        composantes = {NORMALE_PAR_AXE[axe], *TANGENTIELLES_PAR_AXE[axe]}
        assert composantes == {1, 2, 3}, (
            f"axe {axe} : composantes {sorted(composantes)} != (1, 2, 3)")
    # Chaque impulsion est normale à EXACTEMENT un axe.
    normales = sorted(NORMALE_PAR_AXE.values())
    assert normales == [1, 2, 3], f"normales {normales} != [1, 2, 3]"


def test_quatre_champs_sont_refuses():
    """La minoration interdite d'avance (prereg §3) : un état à quatre
    champs ferait de la 3D à deux impulsions, allégerait le stencil et
    rendrait ρ complaisant."""
    q = np.zeros((1, 1, 4, 8, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="load-bearing"):
        pas_f_jetable_3d(q, np)


def test_les_constantes_sont_celles_du_2d():
    """ρ doit être un rapport entre deux DIMENSIONS, pas entre deux
    physiques. Les constantes sont importées du jetable 2D ; ce test
    échoue si une redéfinition apparaît."""
    from src.f1_gpu import substrat_jetable_3d as j3d
    from src.f1_gpu import substrat_fusionne_3d as f3d
    assert j3d.GRAVITE is GRAVITE
    assert j3d.DRY_EPS is DRY_EPS
    assert j3d.DT_JETABLE is DT_JETABLE
    assert f3d.GRAVITE is GRAVITE
    assert f3d.DRY_EPS is DRY_EPS
    assert (f3d.TOL_EQUIVALENCE_RTOL is fusionne_2d.TOL_EQUIVALENCE_RTOL)
    assert (f3d.TOL_EQUIVALENCE_ATOL is fusionne_2d.TOL_EQUIVALENCE_ATOL)


def test_inventaire_du_motif_3d():
    """LE VERROU DES TERMES INERTES — celui qu'aucun test numérique ne peut
    remplacer, et qui n'existe que parce que la mutation l'a exigé.

    Deux termes du motif sont **payés en arithmétique et nuls en valeur** à
    bathymétrie plate : les corrections de pression well-balanced
    (`0.5f·G·(h² − h*²)`, identiquement nulles quand `η ≥ 0`) et la
    réconciliation positivité (que le limiteur minmod rend presque toujours
    inactive). Le kernel 2D le déclare lui-même — « corrections de pression
    well-balanced — CALCULÉES (nulles à b = 0 […]) : le motif paie
    l'arithmétique ».

    Conséquence mesurée par mutation : les SUPPRIMER du kernel 3D ne change
    **aucun** chiffre et passe **tous** les verrous numériques — tout en
    rendant le 3D moins cher. C'est exactement la minoration silencieuse
    qui fabriquerait un ρ complaisant. Seul un inventaire STRUCTUREL peut
    l'attraper.

    Ce test est fragile par nature (il lit du texte) : un renommage le fait
    tomber. C'est voulu — il tombe bruyamment, on revérifie, on met à jour.
    Un verrou qui ne casse jamais ne garde rien."""
    source_2d = fusionne_2d._SOURCE
    from src.f1_gpu.substrat_fusionne_3d import _SOURCE_3D as source_3d

    # Motifs dont le compte DOIT être identique : ils ne dépendent pas de
    # la dimension. Un écart = un terme ajouté ou retiré au passage 3D.
    for motif in ("flux_1d(", "pentes_axe(", "pentes_nulles()", "sqrtf(",
                  "fmaxf(", "0.5f * G * (h", "s.eta) < 0.0f", "> DRY_EPS"):
        assert source_3d.count(motif) == source_2d.count(motif), (
            f"motif « {motif} » : {source_3d.count(motif)} en 3D contre "
            f"{source_2d.count(motif)} en 2D — le passage à trois axes ne "
            "doit RIEN ajouter ni retirer à ce terme.")

    # Motifs dont le compte augmente d'exactement un, et pour une raison
    # nommée : un axe de plus, un champ de plus.
    for motif, raison in (("divergence_axe(", "le troisième axe de flux"),
                          ("minmod(", "la pente du cinquième champ (w)")):
        attendu = source_2d.count(motif) + 1
        assert source_3d.count(motif) == attendu, (
            f"motif « {motif} » : {source_3d.count(motif)} en 3D, attendu "
            f"{attendu} ({raison}).")


def test_le_kernel_2d_est_intouche():
    """Le dénominateur de ρ. Si ce test tombe, l'ancre M-a′ ne décrit plus
    le code qui tourne, et aucun rapport n'est lisible."""
    empreinte = hashlib.sha256(fusionne_2d._SOURCE.encode("utf-8")).hexdigest()
    assert empreinte == EMPREINTE_SOURCE_2D, (
        "le kernel 2D a changé : l'ancre 1,685 ms ne le décrit plus. "
        "Un ρ mesuré contre un dénominateur modifié n'est pas comparable "
        "au budget gravé.")


# --------------------------------------------------------------------
# 2. Isotropie et conservation — sur le jetable (référence CPU)
# --------------------------------------------------------------------

@pytest.mark.parametrize("axe_a,axe_b,champ_a,champ_b,nom", PERMUTATIONS)
def test_isotropie_du_jetable(axe_a, axe_b, champ_a, champ_b, nom):
    """F doit commuter avec l'échange de deux axes (et de leurs
    impulsions). Un axe mal câblé casse cette symétrie — et l'équivalence
    fusionné/jetable ne le verrait pas si les deux portaient la même
    erreur."""
    q0 = _etat_severe(1, 1, 12)
    direct, _ = pas_f_jetable_3d(_echanger(q0, axe_a, axe_b, champ_a, champ_b),
                                 np)
    croise, _ = pas_f_jetable_3d(q0.copy(), np)
    croise = _echanger(croise, axe_a, axe_b, champ_a, champ_b)
    ecart = float(np.abs(direct - croise).max())
    assert ecart < 1e-6, f"anisotropie {nom} : écart max {ecart:.3e}"


def test_masse_conservee_murs_reflechissants():
    """Murs réfléchissants ⇒ aucun flux net au bord ⇒ Σh est invariante.
    Une divergence mal découpée (padding oublié sur un axe, interface
    décalée) le casse immédiatement."""
    q = etat_initial_jetable_3d(1, 1, 12, GRAINE)
    masse0 = float(q[:, :, 0].sum())
    for _ in range(6):
        q, _ = pas_f_jetable_3d(q, np)
    masse = float(q[:, :, 0].sum())
    assert abs(masse - masse0) / masse0 < 1e-5, (
        f"masse {masse0} -> {masse}")


def test_les_trois_impulsions_vivent():
    """Les trois composantes doivent se développer, et à des ordres de
    grandeur comparables. Si `hw` restait nul, le stencil serait de fait
    en 2D et ρ mesurerait une 3D fantôme."""
    q = etat_initial_jetable_3d(1, 1, 12, GRAINE)
    for _ in range(4):
        q, _ = pas_f_jetable_3d(q, np)
    amplitudes = [float(np.abs(q[:, :, k]).max()) for k in (1, 2, 3)]
    assert min(amplitudes) > 0.0, f"impulsion morte : {amplitudes}"
    assert max(amplitudes) / min(amplitudes) < 5.0, (
        f"impulsions déséquilibrées {amplitudes} — un axe traité "
        "différemment des autres")


# --------------------------------------------------------------------
# 3. Le fusionné 3D — celui qui porte le chiffre
# --------------------------------------------------------------------

@sans_gpu
@pytest.mark.parametrize("n_systemes", [1, 2])
def test_equivalence_de_motif_3d(n_systemes):
    """I-2 du prereg : le fusionné calcule ce qu'il déclare. Tolérances de
    M-a′ reconduites — l'écart admis est le réordonnancement f32, pas une
    physique différente."""
    from src.f1_gpu.substrat_fusionne_3d import (
        TOL_EQUIVALENCE_ATOL, TOL_EQUIVALENCE_RTOL, pas_f_fusionne_3d)
    q0 = _etat_severe(2, n_systemes, 16)
    attendu, _ = pas_f_jetable_3d(q0.copy(), np)
    obtenu, _ = pas_f_fusionne_3d(cp.asarray(q0), cp)
    np.testing.assert_allclose(
        cp.asnumpy(obtenu), attendu,
        rtol=TOL_EQUIVALENCE_RTOL, atol=TOL_EQUIVALENCE_ATOL)


@sans_gpu
@pytest.mark.parametrize("axe_a,axe_b,champ_a,champ_b,nom", PERMUTATIONS)
def test_isotropie_du_fusionne(axe_a, axe_b, champ_a, champ_b, nom):
    """L'isotropie testée sur le kernel lui-même : la sélection
    normale/tangentielles du CUDA est écrite trois fois à la main (une
    branche par axe) — c'est le seul endroit du kernel où une faute de
    copie ne casserait rien d'autre."""
    from src.f1_gpu.substrat_fusionne_3d import pas_f_fusionne_3d
    q0 = _etat_severe(1, 1, 16)
    direct, _ = pas_f_fusionne_3d(
        cp.asarray(_echanger(q0, axe_a, axe_b, champ_a, champ_b)), cp)
    croise, _ = pas_f_fusionne_3d(cp.asarray(q0), cp)
    croise = _echanger(cp.asnumpy(croise), axe_a, axe_b, champ_a, champ_b)
    ecart = float(np.abs(cp.asnumpy(direct) - croise).max())
    assert ecart < 1e-6, f"anisotropie {nom} du kernel : écart {ecart:.3e}"


@sans_gpu
def test_fenetre_non_cubique_refusee():
    """Le slot 3D est un cube (§A51 : 512² = 64³). Une fenêtre plate
    mesurerait autre chose que le slot du budget."""
    from src.f1_gpu.substrat_fusionne_3d import pas_f_fusionne_3d
    q = cp.zeros((1, 1, CHAMPS_PAR_SYSTEME_3D, 8, 8, 4), dtype=cp.float32)
    with pytest.raises(ValueError, match="cubique"):
        pas_f_fusionne_3d(q, cp)


@sans_gpu
def test_aucun_debordement_de_registres():
    """Témoin M-6, verrouillé : un kernel qui déborde en mémoire locale
    mesure son propre débordement, pas la dimension. Si ce test tombe, ρ
    reste lisible mais DOIT être prononcé avec la réserve
    d'implémentation nommée (I-3 du prereg)."""
    from src.f1_gpu.substrat_fusionne_3d import attributs_kernel_3d
    attributs = attributs_kernel_3d(cp)
    assert attributs["memoire_locale_octets"] == 0, (
        f"débordement en mémoire locale : {attributs}")


@sans_gpu
def test_sortie_en_place_autorisee():
    """`sortie is q` est le mode du driver (B2, aucun état alloué en
    série). Le kernel ne lit `q_base` qu'à sa propre cellule — ce test le
    vérifie plutôt que de le croire."""
    from src.f1_gpu.substrat_fusionne_3d import (
        TOL_EQUIVALENCE_ATOL, TOL_EQUIVALENCE_RTOL, pas_f_fusionne_3d)
    q0 = _etat_severe(1, 1, 16)
    hors_place, _ = pas_f_fusionne_3d(cp.asarray(q0), cp)
    en_place = cp.asarray(q0)
    pas_f_fusionne_3d(en_place, cp, sortie=en_place)
    np.testing.assert_allclose(
        cp.asnumpy(en_place), cp.asnumpy(hors_place),
        rtol=TOL_EQUIVALENCE_RTOL, atol=TOL_EQUIVALENCE_ATOL)
