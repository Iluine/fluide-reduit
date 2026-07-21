"""Tests F1 — M-b tranche-2 : le bras témoin (§A31).

Ce que ces tests protègent :
  1. le témoin reproduit `run_episode` en f32 — config IDENTIQUE au rederive,
     SEULE l'arithmétique diffère (c'est ce qui isole la source (a)) ;
  2. l'invariant liant : le témoin appelle `pas_f_fidele` (le MÊME objet
     qu'en tranche-1), réfléchissant (bord = vrai bord, physiquement correct) ;
  3. la cadence Exner est LUE de save_every, jamais choisie ;
  4. C1 intouché."""
import hashlib
from dataclasses import replace

import numpy as np
import pytest

from config import GridConfig
from src.f1_gpu.backend import cupy_disponible
from src.f1_gpu.exner_gpu import CADENCE_EXNER
from src.f1_gpu.temoin_fidele import (
    evoluer_episode_temoin,
    evoluer_histoire_temoin,
)
from src.sediment import SedimentParams, default_terrain, run_episode

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")


def _terrain():
    return default_terrain(GridConfig()).astype(np.float32)


# ----- 1. le témoin reproduit run_episode (seule l'arithmétique diffère) -----

@gpu_requis
def test_temoin_f32_reproduit_rederive_f64():
    """La CONFIG est identique (pulse, wetdry, Exner, réfléchissant) ; seule
    f32 vs f64 diffère. L'écart sur s doit être de niveau f32 — c'est ce qui
    permet d'ISOLER (a). Épisode réduit pour la vitesse ; la cadence Exner
    reste LUE."""
    import cupy as cp
    b0 = _terrain()
    s0 = np.zeros_like(b0)
    centre = (0.5, 0.5)
    s_tem = evoluer_episode_temoin(s0, b0, centre, cp, n_settle=60)
    s_cpu = run_episode(s0.astype(np.float64), b0.astype(np.float64), centre,
                        replace(SedimentParams(), N_settle=60))
    assert np.isfinite(s_tem).all() and float(s_tem.min()) >= 0.0
    assert float(s_tem.max()) > 0.0                    # dépôt non trivial
    rel = np.max(np.abs(s_tem - s_cpu)) / max(np.max(np.abs(s_cpu)), 1e-9)
    assert rel < 1e-3                                  # niveau f32, pas plus


@gpu_requis
def test_temoin_deterministe():
    """Même seed/centre ⇒ même s (le témoin est reproductible)."""
    import cupy as cp
    b0 = _terrain()
    s0 = np.zeros_like(b0)
    a = evoluer_episode_temoin(s0, b0, (0.4, 0.6), cp, n_settle=40)
    b = evoluer_episode_temoin(s0, b0, (0.4, 0.6), cp, n_settle=40)
    np.testing.assert_array_equal(a, b)


@gpu_requis
def test_histoire_temoin_centres_fournis():
    """Le témoin REÇOIT les centres (mêmes que le rederive), il ne les tire
    pas — pour rester config-identique."""
    import cupy as cp
    b0 = _terrain()
    centres = [(0.3, 0.3), (0.5, 0.5)]
    out = evoluer_histoire_temoin(101, 2, b0, cp, centres,
                                  replace(SedimentParams(), N_settle=30),
                                  checkpoints={1, 2})
    assert set(out) == {1, 2}
    assert np.isfinite(out[2]).all()


# ----- 2/3. invariant liant + cadence + mode -----

def test_invariant_liant_et_cadence():
    """Le témoin importe `pas_f_fidele` (le MÊME objet qu'en tranche-1) et
    le mode réfléchissant ; la cadence Exner est LUE de save_every."""
    import src.f1_gpu.temoin_fidele as tem
    from src.f1_gpu.substrat_fidele import MODE_REFLECHISSANT, pas_f_fidele
    assert tem.pas_f_fidele is pas_f_fidele
    assert tem.MODE_REFLECHISSANT == MODE_REFLECHISSANT
    assert CADENCE_EXNER == SedimentParams().save_every


def test_c1_intouche():
    from src.f1_gpu.substrat_fidele import _SOURCE_FIDELE
    assert hashlib.sha256(_SOURCE_FIDELE.encode()).hexdigest() == (
        "6dd207cae0a9c23a6a042db825a3259051e8491c1f118d614e52a4b4ebc6265d")
