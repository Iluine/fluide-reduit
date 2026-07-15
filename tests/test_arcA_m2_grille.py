"""Tests Manche 2 (§A13) Tasks 2/3 — grille (`scripts/run_arcA_m2_grille.py`).

RAPIDES sauf deux intégrations réelles courtes (~15 s au total : vérité 3
épisodes + cellule 2 épisodes) — le protocole complet est couvert par les
tests Task 0/Task 1 ; ici on teste la MÉCANIQUE de la grille (comptabilité,
partage de vérité bit-exact, contrôles post-hoc, géométrie quadtree)."""
from __future__ import annotations

import numpy as np
import pytest

import scripts.run_arcA_m2_grille as g
from src.summary_quadtree import regenerate_qt, size_floats_qt, summarize_qt


# =============================================================================
# Comptabilité des épisodes (choix (vi)) — arithmétique gravée, pas un seuil
# =============================================================================


def test_comptabilite_episodes():
    assert g._EP_VERITE == 96 * 5 == 480
    assert g._EP_MOTEUR == (6 + 24 + 96) * 3 * 5 == 1890
    assert g._EP_FERM == 6 + 24 + 96 == 126
    assert g._EP_REDERIVE_SPOT == 24
    assert g._EP_SHUF == 72
    assert g._EP_CORRUPTION == 48
    assert g.EPISODES_GRILLE == 2640


def test_parametres_figes_a13_1():
    assert g.SEEDS == (101, 102, 103, 104, 105)
    assert g.DTS == (1, 4, 16)
    assert g.BUDGETS == (128, 256, 400)
    assert g.N_EMISSIONS == 6
    assert g.N_EP_VERITE == 96
    assert g.GATE_SECONDS == 36000.0


def test_charge_pins_coherence_sonde():
    """Le pin sévère lu doit être le littéral de la sonde (artefact non
    déplacé) ; le laxiste doit exister avec un IC ordonné."""
    pins = g.charge_pins()
    assert abs(pins["severe"]["jnd"] - 0.07334065957385666) < 1e-12
    bas, haut = pins["laxiste"]["ic"]
    assert bas < pins["laxiste"]["jnd"] < haut


# =============================================================================
# shuf-commit (choix (iii)) : histogramme préservé, topologie intacte,
# déterminisme
# =============================================================================


def _commit_synthetique(seed: int = 7, budget: int = 64):
    rng = np.random.default_rng(seed)
    field = rng.uniform(0.0, 1.0, size=(64, 64))
    return summarize_qt(field, budget)


def test_permute_registre_histogramme_et_topologie():
    registre = [_commit_synthetique(1), _commit_synthetique(2)]
    permute = g._permute_registre(registre, seed=101, dt=4)
    assert len(permute) == 2
    for orig, perm in zip(registre, permute):
        assert np.array_equal(orig.topology, perm.topology)
        assert np.array_equal(np.sort(orig.means), np.sort(perm.means))
        assert size_floats_qt(perm) == size_floats_qt(orig)
        # La permutation doit faire quelque chose (champ non constant).
        assert not np.array_equal(orig.means, perm.means)


def test_permute_registre_deterministe():
    registre = [_commit_synthetique(3)]
    a = g._permute_registre(registre, seed=101, dt=4)
    b = g._permute_registre(registre, seed=101, dt=4)
    assert np.array_equal(a[0].means, b[0].means)
    c = g._permute_registre(registre, seed=102, dt=4)
    assert not np.array_equal(a[0].means, c[0].means)


# =============================================================================
# Géométrie quadtree (choix (iv)) : aires + rectangle de feuille
# =============================================================================


def test_aires_feuilles_partition():
    commit = _commit_synthetique(11, budget=96)
    aires = g._aires_feuilles(commit)
    assert aires.size == commit.means.size
    assert int(aires.sum()) == 64 * 64  # partition exacte du domaine


def test_rect_feuille_coherent_avec_aire():
    commit = _commit_synthetique(13, budget=96)
    aires = g._aires_feuilles(commit)
    cible = int(np.argmax(aires))
    row, col, cote = g._rect_feuille(commit, cible)
    assert cote * cote == int(aires[cible])
    assert 0 <= row <= 64 - cote and 0 <= col <= 64 - cote


def test_corruption_localisee_au_reancrage():
    """Altérer UNE moyenne de feuille ne change le champ régénéré QUE dans le
    rectangle de cette feuille (localisation exacte au ré-ancrage — c'est la
    propriété que la mesure de localisation (iv) exploite)."""
    commit = _commit_synthetique(17, budget=64)
    aires = g._aires_feuilles(commit)
    cible = int(np.argmax(aires))
    row, col, cote = g._rect_feuille(commit, cible)
    base = regenerate_qt(commit)
    means_corr = np.array(commit.means, copy=True)
    means_corr[cible] += 1.0
    corr = regenerate_qt(type(commit)(topology=np.array(commit.topology, copy=True),
                                      means=means_corr, shape=commit.shape))
    diff = np.abs(corr - base)
    masque = np.zeros_like(diff, dtype=bool)
    masque[row:row + cote, col:col + cote] = True
    assert np.all(diff[~masque] == 0.0)
    assert np.all(diff[masque] > 0.0)


# =============================================================================
# Intégrations réelles COURTES : vérité partagée bit-exacte + cellule
# =============================================================================


@pytest.fixture()
def _grille_sandbox(tmp_path, monkeypatch):
    """Sandbox : sorties vers tmp, grille réduite à ~5 épisodes réels."""
    monkeypatch.setattr(g, "PARTS_DIR", tmp_path / "m2_parts")
    monkeypatch.setattr(g, "OUT_DIR", tmp_path)
    monkeypatch.setattr(g, "N_EP_VERITE", 3)
    monkeypatch.setattr(g, "N_EMISSIONS", 2)
    monkeypatch.setattr(g, "BUDGETS", (128,))
    return tmp_path


def test_verite_round_trip_bit_exact(_grille_sandbox):
    """Choix (i) : la vérité rechargée du npz est BIT-IDENTIQUE à la vérité
    vivante (np.savez float64 sans perte) — condition de licéité du partage."""
    from scripts.run_arcA_m2_registre import chaine_verite

    g.phase_verite(101)
    rechargee = g.charge_verite(101)
    vivante = chaine_verite(101, 3)
    assert set(rechargee.keys()) == {1, 2, 3}
    for t in (1, 2, 3):
        assert np.array_equal(rechargee[t], vivante[t])


def test_phase_cell_produit_series_et_registre(_grille_sandbox):
    """La cellule écrit une série Δχ par budget (longueur n_émissions,
    Δχ_1 = 0 structurel) + le registre re-chargeable (topologies/moyennes)."""
    g.phase_verite(101)
    path = g.phase_cell(101, 1)
    with np.load(path) as data:
        serie = data["delta_chi_k128"]
        assert serie.shape == (2,)
        assert serie[0] == 0.0  # Δχ_1 structurel (mesure avant tout commit)
        registre = g._arrays_vers_registre(data, "reg_k128", 2)
    assert len(registre) == 2
    assert all(size_floats_qt(c) <= 128 for c in registre)


def test_charge_verite_fail_loud(_grille_sandbox):
    with pytest.raises(RuntimeError, match="absente"):
        g.charge_verite(999)
