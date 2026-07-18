"""Tests F1 tranche-1 — chrono (B6) et compteurs de transfert (composant
(c)). Backend numpy (B1) : la logique (warmup exclu, stats, comptage
d'octets par direction/frame) est identique sous cupy — c'est elle qu'on
teste ici ; le chemin cuda-Events est couvert par le smoke GPU."""
import numpy as np
import pytest

from src.f1_gpu.chrono import chronometrer_frames, stats_ms
from src.f1_gpu.transferts import TransfertComptable


def test_stats_ms_mediane_et_p99():
    stats = stats_ms([float(v) for v in range(1, 101)])
    assert stats["n"] == 100
    assert stats["mediane_ms"] == pytest.approx(50.5)
    assert stats["p99_ms"] == pytest.approx(np.percentile(range(1, 101), 99))
    assert stats["min_ms"] == 1.0
    assert stats["max_ms"] == 100.0


def test_stats_ms_vide_fail_loud():
    with pytest.raises(ValueError):
        stats_ms([])


def test_chronometrer_warmup_exclu_et_serie_complete():
    appels = []

    def frame():
        appels.append(1)

    resultat = chronometrer_frames(np, frame, warmup=3, serie=7)
    assert len(appels) == 10                      # warmup + série appelés
    assert len(resultat["serie_ms"]) == 7         # ... mais série seule agrégée
    assert len(resultat["warmup_ms"]) == 3
    assert resultat["stats"]["n"] == 7
    assert all(ms >= 0.0 for ms in resultat["serie_ms"])


def test_chronometrer_parametres_invalides():
    with pytest.raises(ValueError):
        chronometrer_frames(np, lambda: None, warmup=-1, serie=10)
    with pytest.raises(ValueError):
        chronometrer_frames(np, lambda: None, warmup=0, serie=0)


def test_comptage_octets_par_direction_et_par_frame():
    comptable = TransfertComptable(np)
    descendu = comptable.descendre(np.zeros(10, dtype=np.float32))   # 40 o
    assert descendu.shape == (10,)
    comptable.remonter(np.zeros(8, dtype=np.uint8))                  # 8 o
    bilan1 = comptable.frame_suivante()
    assert bilan1["h2d_octets"] == 40 and bilan1["h2d_n"] == 1
    assert bilan1["d2h_octets"] == 8 and bilan1["d2h_n"] == 1

    comptable.descendre(np.zeros((2, 3), dtype=np.float64))          # 48 o
    bilan2 = comptable.frame_suivante()
    assert bilan2["h2d_octets"] == 48 and bilan2["d2h_octets"] == 0
    assert comptable.bilans == [bilan1, bilan2]


def test_remonter_renvoie_une_copie_cpu():
    comptable = TransfertComptable(np)
    source = np.arange(4, dtype=np.float32)
    copie = comptable.remonter(source)
    assert np.array_equal(copie, source)
    copie[0] = 99.0
    assert source[0] == 0.0


def test_stats_regime_exclut_les_premiers_bilans():
    comptable = TransfertComptable(np)
    for octets in (1000, 10, 10, 10):
        comptable.descendre(np.zeros(octets, dtype=np.uint8))
        comptable.frame_suivante()
    stats = comptable.stats_regime(exclure_premiers=1)
    assert stats["n_frames"] == 3
    assert stats["h2d_octets"]["mediane"] == 10.0
    assert stats["transfert_total_octets"]["mediane"] == 10.0


def test_stats_regime_vide_fail_loud():
    with pytest.raises(ValueError):
        TransfertComptable(np).stats_regime()
