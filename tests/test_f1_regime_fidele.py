"""Tests F1 — M-b tranche-1 / C4 : vérification de régime (§A29-C1).

Ce que ces tests protègent :
  1. l'état synthétisé EXERCE le régime (fronts wet/dry, CFL sollicitée à
     4–12× de marge) — un chrono sur du sec ne vaudrait rien (§A28) ;
  2. la vérification porte sur la SÉRIE, pas seulement l'init (amendement B) ;
  3. FAIL-LOUD si la marge CFL s'effondre — pas de lecture T1 sans PASS (B9)."""
import pytest

from src.f1_gpu.backend import cupy_disponible
from src.f1_gpu.regime_fidele import (
    DT_FRAME,
    MARGE_CFL_BASSE,
    MARGE_CFL_HAUTE,
    etat_synthetise,
    exiger_regime_pass,
    mesurer_regime,
    verifier_regime_serie,
)

gpu_requis = pytest.mark.skipif(
    not cupy_disponible(), reason="cupy/device CUDA indisponible")


# ----- 1. l'état synthétisé exerce le régime -----

@gpu_requis
@pytest.mark.parametrize("n", [64, 512])
def test_etat_synthetise_exerce_le_regime(n):
    """Fronts wet/dry présents ET marge CFL dans [4, 12] — à l'échelle
    réduite comme à l'échelle V4."""
    import cupy as cp
    q, b = etat_synthetise(n, cp, graine=101)
    m = mesurer_regime(q, b, cp)
    assert m["front_present"] is True
    assert 0.2 < m["fraction_wet"] < 0.8
    assert MARGE_CFL_BASSE <= m["marge_cfl"] <= MARGE_CFL_HAUTE


@gpu_requis
def test_dt_frame_sous_cfl_marge_positive():
    """dt_frame siège SOUS la CFL (mipmap §A28) : la marge dt_cfl/dt_frame
    est > 1 — le pas de frame ne viole pas la CFL."""
    import cupy as cp
    q, b = etat_synthetise(256, cp)
    m = mesurer_regime(q, b, cp, DT_FRAME)
    assert m["dt_cfl"] > DT_FRAME                # sous-CFL
    assert m["marge_cfl"] > 1.0


# ----- 2. vérification sur la série -----

def _serie(marge, frac=0.45, front=True, n=5):
    return [{"marge_cfl": marge, "fraction_wet": frac,
             "front_present": front, "dt_cfl": marge * DT_FRAME}
            for _ in range(n)]


def test_serie_regime_tenu_passe():
    v = verifier_regime_serie(_serie(6.0))
    assert v["pass"] is True
    assert v["fronts_partout"] is True
    exiger_regime_pass(v)                        # ne lève pas


def test_serie_marge_effondree_echoue_fail_loud():
    """Amendement B : une seule frame à marge effondrée (C-property cassée,
    smax monté) fait FAIL — le chrono mesurerait une divergence."""
    serie = _serie(6.0, n=3) + [{"marge_cfl": 2.0, "fraction_wet": 0.45,
                                 "front_present": True, "dt_cfl": 0.03}]
    v = verifier_regime_serie(serie)
    assert v["pass"] is False
    assert v["marge_cfl_effondree"] is True
    assert "marge CFL hors" in v["motif"]
    with pytest.raises(RuntimeError, match="RÉGIME NON EXERCÉ"):
        exiger_regime_pass(v)


def test_serie_trop_lente_echoue():
    """Marge > 12 (état quasi au repos) : la CFL n'est pas sollicitée, le
    chrono serait peu représentatif — FAIL."""
    v = verifier_regime_serie(_serie(20.0))
    assert v["pass"] is False
    assert "hors [4,12]" in v["motif"]


def test_serie_sans_front_echoue():
    """Domaine sans front (tout mouillé uniforme ou tout sec) : FAIL."""
    v = verifier_regime_serie(_serie(6.0, front=False))
    assert v["pass"] is False
    assert "pas de front" in v["motif"]


def test_serie_trop_seche_echoue():
    v = verifier_regime_serie(_serie(6.0, frac=0.05))
    assert v["pass"] is False
    assert "fraction wet hors" in v["motif"]


def test_serie_vide_echoue():
    v = verifier_regime_serie([])
    assert v["pass"] is False
    with pytest.raises(RuntimeError, match="RÉGIME NON EXERCÉ"):
        exiger_regime_pass(v)


def test_exiger_refuse_none():
    with pytest.raises(RuntimeError, match="RÉGIME NON EXERCÉ"):
        exiger_regime_pass(None)


# ----- 3. C1/kernels figés intouchés -----

def test_c1_intouche():
    import hashlib
    from src.f1_gpu.substrat_fidele import _SOURCE_FIDELE
    assert hashlib.sha256(_SOURCE_FIDELE.encode()).hexdigest() == (
        "aef7237d758f7292bc1434940e94464e2a5ad7d41c4f334baae0310c3cadc4c1")
