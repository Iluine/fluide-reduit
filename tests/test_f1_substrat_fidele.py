"""Tests F1 — M-b tranche-1 / C1 : le F FIDÈLE b-aware (§A28-A29).

Ce que ces tests protègent :
  1. le VERROU d'empreinte de `_SOURCE_FIDELE` dès le premier commit
     (§A29-C) — on ne refait pas le coup de L3 resté sans verrou ;
  2. la RÉUTILISATION structurelle du préfixe b-agnostique du figé ;
  3. la **C-PROPERTY sur TERRAIN RÉEL** (pas plat) : un lac au repos ne
     génère aucune vitesse parasite — c'est la preuve que le bien-équilibré
     d'Audusse est porté correctement, et c'est ce que le kernel figé (plat)
     ne pouvait pas tester ;
  4. le BORD PLUGGABLE : mode 0 (mur) et mode 1 (halo) sont le MÊME
     opérateur intérieur — prouvé bit-exact sur un étage ;
  5. les kernels FIGÉS restent intouchés (empreintes citées)."""
import hashlib

import numpy as np
import pytest

from config import GridConfig
from src.f1_gpu.substrat_fidele import (
    _NOM_KERNEL_FIDELE,
    _PREFIXE_BASE,
    _SOURCE_FIDELE,
    _TAILLE_BLOC,
    LONGUEUR_SOURCE_FIDELE,
    MODE_HALO,
    MODE_REFLECHISSANT,
    SHA256_SOURCE_FIDELE,
    pas_f_fidele,
    reduction_cfl_fidele,
)
from src.f1_gpu.backend import cupy_disponible
from src.sediment import default_terrain

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")

# Empreinte GRAVÉE dès le premier commit (§A29-C). Re-calculable :
# sha256(_SOURCE_FIDELE.encode()), longueur figée à côté.
SHA256_ATTENDU: str = (
    "6dd207cae0a9c23a6a042db825a3259051e8491c1f118d614e52a4b4ebc6265d")
LONGUEUR_ATTENDUE: int = 9248

N: int = 64


def _terrain():
    return default_terrain(GridConfig()).astype(np.float32)


def _lac(eta_const: float):
    """Un lac au repos sur le terrain réel : η = h + b = const là où c'est
    mouillé, u = v = 0. C'est l'état sur lequel la C-property se lit."""
    b0 = _terrain()
    h = np.maximum(eta_const - b0, 0.0).astype(np.float32)
    q = np.zeros((1, 1, 3, N, N), np.float32)
    q[0, 0, 0] = h                                  # hu = hv = 0
    b = b0.reshape(1, 1, N, N)
    return q, b, h


# ----- 1. verrou d'empreinte -----

def test_empreinte_source_fidele_verrouillee():
    """§A29-C : le CUDA ne bouge pas sans que ce verrou parle. sha256 de la
    chaîne EXTRAITE, longueur figée à côté (recalculable)."""
    assert len(_SOURCE_FIDELE) == LONGUEUR_ATTENDUE
    assert LONGUEUR_SOURCE_FIDELE == LONGUEUR_ATTENDUE
    empreinte = hashlib.sha256(_SOURCE_FIDELE.encode()).hexdigest()
    assert empreinte == SHA256_ATTENDU
    assert SHA256_SOURCE_FIDELE == SHA256_ATTENDU


# ----- 2. réutilisation du préfixe b-agnostique -----

def test_prefixe_b_agnostique_vient_du_fusionne():
    """Les helpers device b-agnostiques (desing, minmod, defines) sont
    REPRIS du figé, non recopiés — comme L3 a repris son préfixe."""
    from src.f1_gpu.substrat_fusionne import _SOURCE as _SOURCE_FUSIONNE

    assert _PREFIXE_BASE == _SOURCE_FUSIONNE[:len(_PREFIXE_BASE)]
    assert _SOURCE_FIDELE.startswith(_PREFIXE_BASE)
    for helper in ("desing", "minmod"):
        assert helper in _PREFIXE_BASE
    # les fonctions b-aware sont NEUVES (pas dans le préfixe)
    for neuf in ("lire_b", "pentes_axe_b", "flux_1d_b", "divergence_axe_b"):
        assert neuf not in _PREFIXE_BASE
        assert neuf in _SOURCE_FIDELE


# ----- 3. C-PROPERTY sur terrain réel -----

@gpu_requis
def test_c_property_terrain_reel_plein():
    """Lac PLEINEMENT mouillé (η au-dessus de tout le lit) sur terrain
    RÉEL : au repos, aucune vitesse parasite. C'est le bien-équilibré
    d'Audusse — le figé plat ne pouvait pas le tester."""
    import cupy as cp
    b0 = _terrain()
    q, b, h = _lac(float(b0.max() + 0.3))
    sortie, _ = pas_f_fidele(cp.asarray(q), cp.asarray(b), cp, dt=0.01)
    s = cp.asnumpy(sortie)[0, 0]
    assert float(np.max(np.abs(s[1:3]))) < 1e-6     # vitesse parasite ~ 0
    assert float(np.max(np.abs(s[0] - h))) < 1e-6   # h inchangé


@gpu_requis
def test_c_property_terrain_reel_front_wet_dry():
    """Lac PARTIEL : fronts wet/dry présents (le régime que M-b exerce).
    La vitesse parasite reste nulle ; h peut bruiter au front sec (propriété
    du solveur, cf. solver_wetdry), mais PAS de moment parasite."""
    import cupy as cp
    b0 = _terrain()
    eta = float(b0.min() + 0.5 * (b0.max() - b0.min()))
    q, b, h = _lac(eta)
    frac_wet = float((h > 1e-4).mean())
    assert 0.2 < frac_wet < 0.8                     # vrai front, pas trivial
    sortie, _ = pas_f_fidele(cp.asarray(q), cp.asarray(b), cp, dt=0.01)
    s = cp.asnumpy(sortie)[0, 0]
    assert float(np.max(np.abs(s[1:3]))) < 1e-6     # C-property au front


@gpu_requis
def test_reste_au_repos_sur_plusieurs_pas():
    """La C-property n'est pas un accident du premier pas : le lac reste au
    repos sur une petite série."""
    import cupy as cp
    b0 = _terrain()
    q, b, _ = _lac(float(b0.min() + 0.5 * (b0.max() - b0.min())))
    etat = cp.asarray(q)
    bd = cp.asarray(b)
    for _ in range(8):
        etat, _ = pas_f_fidele(etat, bd, cp, dt=0.01)
    s = cp.asnumpy(etat)[0, 0]
    assert float(np.max(np.abs(s[1:3]))) < 1e-6


# ----- 4. bord pluggable : même opérateur intérieur -----

def _halo_reflexion(q_np, b_np):
    """Halo padé de 2 rempli par la RÈGLE EXACTE de mode 0 (clamp edge +
    négation de la qdm normale). Sert à prouver que mode 1 avec un halo
    réfléchi reproduit mode 0 — donc que l'opérateur intérieur est le même,
    et que le bord n'est qu'une SOURCE de données."""
    h, hu, hv = q_np[0, 0]
    b = b_np[0, 0]
    nh = N + 4
    hq = np.zeros((1, 1, 3, nh, nh), np.float32)
    hb = np.zeros((1, 1, nh, nh), np.float32)
    for row in range(-2, N + 2):
        for col in range(-2, N + 2):
            r, c = min(max(row, 0), N - 1), min(max(col, 0), N - 1)
            sx = -1.0 if (col < 0 or col >= N) else 1.0
            sy = -1.0 if (row < 0 or row >= N) else 1.0
            hq[0, 0, 0, row + 2, col + 2] = h[r, c]
            hq[0, 0, 1, row + 2, col + 2] = sx * hu[r, c]
            hq[0, 0, 2, row + 2, col + 2] = sy * hv[r, c]
            hb[0, 0, row + 2, col + 2] = b[r, c]
    return hq, hb


@gpu_requis
def test_bord_pluggable_meme_operateur_interieur():
    """Sur UN étage, mode 0 (mur) et mode 1 (halo = réflexion exacte)
    produisent le MÊME résultat, BIT à BIT. Preuve de l'invariant liant :
    le bord est une donnée injectée, l'opérateur intérieur ne change pas.

    (La cohérence du halo ENTRE étages RK2 est de tranche-2 — mode 0 reflète
    l'état courant à chaque étage, un halo figé se périmerait ; c'est
    exactement ce que le prereg dit imposer à tranche-2.)"""
    import cupy as cp
    from src.f1_gpu.substrat_fidele import _obtenir_kernel_fidele

    b0 = _terrain()
    rng = np.random.default_rng(3)
    eta = float(b0.min() + 0.4 * (b0.max() - b0.min()))
    h = (np.maximum(eta - b0, 0.0) + 0.02 * rng.random((N, N))).astype(
        np.float32)
    hu = (0.05 * rng.standard_normal((N, N)) * (h > 1e-3)).astype(np.float32)
    hv = (0.05 * rng.standard_normal((N, N)) * (h > 1e-3)).astype(np.float32)
    q = np.stack([h, hu, hv]).reshape(1, 1, 3, N, N)
    b = b0.reshape(1, 1, N, N)
    hq, hb = _halo_reflexion(q, b)

    qd, bd = cp.asarray(q), cp.asarray(b)
    kernel = _obtenir_kernel_fidele(cp)
    total = N * N
    grille = ((total + _TAILLE_BLOC - 1) // _TAILLE_BLOC,)
    o0, o1 = cp.empty_like(qd), cp.empty_like(qd)

    def lancer(out, mode, HQ, HB):
        kernel(grille, (_TAILLE_BLOC,),
               (qd, qd, out, bd, HQ, HB, np.float32(0.02), np.float32(0.0),
                np.int32(N), np.int64(total), np.int32(mode)))

    lancer(o0, MODE_REFLECHISSANT, qd, bd)
    lancer(o1, MODE_HALO, cp.asarray(hq), cp.asarray(hb))
    assert float(cp.max(cp.abs(o0 - o1))) == 0.0    # bit à bit


# ----- 5. sémantique et kernels figés -----

def test_semantique_trois_champs_validee():
    """L'état évolué est (h, hu, hv) — 3 champs ; s n'est pas advecté. Les
    shapes fautives sont refusées fail-loud."""
    if not cupy_disponible():
        pytest.skip("cupy indisponible")
    import cupy as cp
    q4 = cp.zeros((1, 1, 4, N, N), cp.float32)      # 4 champs : refusé
    b = cp.zeros((1, 1, N, N), cp.float32)
    with pytest.raises(ValueError, match="h, hu, hv"):
        pas_f_fidele(q4, b, cp, dt=0.01)
    q3 = cp.zeros((1, 1, 3, N, N), cp.float32)
    with pytest.raises(ValueError, match="b_eff"):    # b mal formé
        pas_f_fidele(q3, cp.zeros((1, 1, N, N + 1), cp.float32), cp, dt=0.01)


def test_kernels_figes_intouches():
    """Le F fidèle est un `_SOURCE_FIDELE` NEUF : les kernels figés gardent
    leurs empreintes (re-calculables sur la chaîne extraite)."""
    from src.f1_gpu.substrat_fusionne import _SOURCE
    from src.f1_gpu.substrat_l3 import _SOURCE_L3

    assert hashlib.sha256(_SOURCE.encode()).hexdigest() == (
        "e18015f57e14263414239b18c51d25cd335250f58dde2ccb892a6e2c1d6f30b4")
    assert hashlib.sha256(_SOURCE_L3.encode()).hexdigest() == (
        "9533a130b6624ce1b5b76d7d8e206aaae2e55fcc34e2a8d4eafa315e8014781a")
    assert _NOM_KERNEL_FIDELE == "etage_ssp_wetdry_o2_fidele"


@gpu_requis
def test_cfl_calculee_non_triviale():
    """La CFL est CALCULÉE (pour la vérification de régime C4) : finie et
    positive sur un lac réel."""
    import cupy as cp
    b0 = _terrain()
    q, b, _ = _lac(float(b0.min() + 0.5 * (b0.max() - b0.min())))
    dt_cfl = reduction_cfl_fidele(cp.asarray(q), cp)
    assert 0.0 < dt_cfl < 10.0
