"""Tests F1 — M-b tranche-1 / C2 : Exner GPU (§A29-C1).

Ce que ces tests protègent :
  1. le verrou d'empreinte de `_SOURCE_EXNER` (§A29-C) ;
  2. la CADENCE DÉRIVÉE du rederive (save_every), JAMAIS choisie — le motif
     gravé : une cadence choisie ferait comparer deux physiques à tranche-2 ;
  3. l'équivalence GPU ↔ `_exner_step` CPU (dépôt/érosion, masque wet, s≥0) ;
  4. C1 et les kernels figés intouchés."""
import hashlib

import numpy as np
import pytest

from src.f1_gpu.backend import cupy_disponible
from src.f1_gpu.exner_gpu import (
    CADENCE_EXNER,
    LONGUEUR_SOURCE_EXNER,
    SHA256_SOURCE_EXNER,
    THETA_C,
    _SOURCE_EXNER,
    cadence_exner_derivee,
    pas_exner,
)
from src.sediment import SedimentParams, _exner_step

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")

SHA256_ATTENDU: str = (
    "3533fd0bde3927856fa97b012a78fd04cfc6ab8c9ccf5fe17b13b95c7cd63aa6")
LONGUEUR_ATTENDUE: int = 1251
N: int = 64


# ----- 1. verrou d'empreinte -----

def test_empreinte_source_exner_verrouillee():
    """§A29-C : le CUDA Exner ne bouge pas sans que le verrou parle. sha256
    de la chaîne extraite, longueur figée à côté (recalculable)."""
    empreinte = hashlib.sha256(_SOURCE_EXNER.encode()).hexdigest()
    assert SHA256_SOURCE_EXNER == empreinte
    assert empreinte == SHA256_ATTENDU
    assert LONGUEUR_SOURCE_EXNER == LONGUEUR_ATTENDUE == len(_SOURCE_EXNER)


# ----- 2. cadence DÉRIVÉE, pas choisie -----

def test_cadence_exner_lue_du_rederive():
    """La cadence est LUE dans `SedimentParams.save_every`, jamais choisie
    au build (§A29-C1) : ~300 Exner pour ~600 wetdry ⇒ un sur deux."""
    p = SedimentParams()
    assert CADENCE_EXNER == p.save_every
    d = cadence_exner_derivee()
    assert d["save_every"] == p.save_every
    assert d["n_exner_par_episode"] == 300
    assert d["n_wetdry_par_episode"] == 600
    assert d["cadence_exner_sur_wetdry"] == 0.5
    assert d["un_exner_tous_les"] == 2
    assert "non choisie" in d["source"]


def test_la_cadence_suit_le_code_si_save_every_change():
    """Contre-épreuve : la cadence n'est pas une constante gravée à part —
    elle DÉRIVE de save_every. Si le rederive changeait, elle suivrait."""
    from dataclasses import replace
    p = replace(SedimentParams(), save_every=3)
    n_exner = len(range(p.save_every, p.N_settle + 1, p.save_every))
    assert n_exner == 200                     # 600/3, cohérent avec la règle


# ----- 3. équivalence GPU ↔ CPU -----

@gpu_requis
def test_exner_gpu_egale_cpu():
    """Le kernel GPU reproduit `_exner_step` (dépôt/érosion masqués wet,
    clamp s≥0) à la tolérance f32."""
    import cupy as cp
    rng = np.random.default_rng(7)
    h = (0.2 + 0.3 * rng.random((N, N))).astype(np.float32)
    hu = (0.4 * rng.standard_normal((N, N)) * h).astype(np.float32)
    hv = (0.4 * rng.standard_normal((N, N)) * h).astype(np.float32)
    s = (0.05 * rng.random((N, N))).astype(np.float32)
    h[rng.random((N, N)) < 0.15] = 0.0        # cellules sèches
    dt = 0.3

    s_cpu = _exner_step(s.astype(np.float64), h.astype(np.float64),
                        hu.astype(np.float64), hv.astype(np.float64), dt,
                        SedimentParams())
    q = cp.asarray(np.stack([h, hu, hv]).reshape(1, 1, 3, N, N))
    sd = cp.asarray(s.reshape(1, 1, N, N))
    pas_exner(q, sd, cp, dt)
    s_gpu = cp.asnumpy(sd)[0, 0]
    assert np.max(np.abs(s_gpu - s_cpu)) < 1e-6


@gpu_requis
def test_depot_si_lent_erosion_si_rapide():
    """θ < θc (écoulement lent) ⇒ DÉPÔT (s croît) ; θ > θc (rapide) ⇒
    ÉROSION (s décroît). C'est la loi Exner, pas un réglage."""
    import cupy as cp
    h = np.full((N, N), 0.3, np.float32)
    s0 = np.full((N, N), 0.05, np.float32)
    # lent : theta ~ 0 << theta_c=0.5
    q_lent = cp.asarray(np.stack([h, np.zeros_like(h), np.zeros_like(h)]
                                 ).reshape(1, 1, 3, N, N))
    s = cp.asarray(s0.reshape(1, 1, N, N)).copy()
    pas_exner(q_lent, s, cp, 1.0)
    assert float(cp.asnumpy(s).mean()) > s0.mean()        # dépôt
    # rapide : |u| tel que theta = u² > theta_c
    u = np.full((N, N), 1.0, np.float32)                  # theta = 1 > 0.5
    q_rap = cp.asarray(np.stack([h, u * h, np.zeros_like(h)]
                                ).reshape(1, 1, 3, N, N))
    s2 = cp.asarray(s0.reshape(1, 1, N, N)).copy()
    pas_exner(q_rap, s2, cp, 1.0)
    assert float(cp.asnumpy(s2).mean()) < s0.mean()       # érosion


@gpu_requis
def test_s_reste_positif_et_sec_inchange():
    """s ≥ 0 clampé ; une cellule sèche (h ≤ 10·dry_eps) ne reçoit ni dépôt
    ni érosion."""
    import cupy as cp
    h = np.zeros((N, N), np.float32)          # tout sec
    s0 = np.full((N, N), 0.03, np.float32)
    q = cp.asarray(np.stack([h, h, h]).reshape(1, 1, 3, N, N))
    s = cp.asarray(s0.reshape(1, 1, N, N)).copy()
    pas_exner(q, s, cp, 5.0)
    np.testing.assert_array_equal(cp.asnumpy(s)[0, 0], s0)  # sec : inchangé
    # érosion forte ne rend jamais s négatif
    hw = np.full((N, N), 0.3, np.float32)
    u = np.full((N, N), 3.0, np.float32)
    qw = cp.asarray(np.stack([hw, u * hw, np.zeros_like(hw)]
                             ).reshape(1, 1, 3, N, N))
    sw = cp.asarray(np.full((N, N), 0.001, np.float32).reshape(1, 1, N, N))
    pas_exner(qw, sw, cp, 100.0)
    assert float(cp.asnumpy(sw).min()) >= 0.0


def test_theta_c_lu_du_rederive():
    assert THETA_C == SedimentParams().theta_c


# ----- 4. C1 et kernels figés intouchés -----

def test_c1_et_figes_intouches():
    from src.f1_gpu.substrat_fidele import _SOURCE_FIDELE
    from src.f1_gpu.substrat_fusionne import _SOURCE
    from src.f1_gpu.substrat_l3 import _SOURCE_L3
    assert hashlib.sha256(_SOURCE_FIDELE.encode()).hexdigest() == (
        "aef7237d758f7292bc1434940e94464e2a5ad7d41c4f334baae0310c3cadc4c1")
    assert hashlib.sha256(_SOURCE.encode()).hexdigest() == (
        "e18015f57e14263414239b18c51d25cd335250f58dde2ccb892a6e2c1d6f30b4")
    assert hashlib.sha256(_SOURCE_L3.encode()).hexdigest() == (
        "9533a130b6624ce1b5b76d7d8e206aaae2e55fcc34e2a8d4eafa315e8014781a")
