"""Arc A / extension 16L₀ -- tests du Task 2 (mesures M-A1/M-A2 à L=160 +
contrôles ferm/shuf à L∈{10,160}), cf. brief `.superpowers/sdd/
16L0-task2-brief.md`. RAPIDES -- AUCUNE mesure lourde complète ici (le run des
25 cellules réelles, ~30 min, est le job du contrôleur en arrière-plan) :
l'essentiel de la correction est prouvé sur des ma1/ma2 SYNTHÉTIQUES (dict
budget -> float, même forme que le retour de `_measure_cell_qt`).

`test_non_regression_controle_L10` est LE test de non-régression déterminisme
réel (re-mesure ferm@L=10 == tranche L=10 de `measures_qt.npz`, bit-à-bit) --
marqué `@pytest.mark.skipif` tant que les 5 `h160_s{seed}.npz` (Task 1) ne
sont pas produits par le contrôleur en arrière-plan ; il sera levé (et
confirmé) par le contrôleur une fois le run réel disponible."""
from __future__ import annotations

import numpy as np
import pytest

import scripts.run_arcA_measure_16L0 as m16
from scripts.run_arcA_measure import JND_LIST, L_LIST, SEEDS
from scripts.run_arcA_measure_qt import BUDGETS, MEASURES_PATH as MEASURES_QT_PATH
from scripts.run_arcC_surface_kstar_pins import sous_jnd_a_jnd

# =============================================================================
# Fabrique de ma1/ma2 SYNTHÉTIQUES -- valeur unique par (L, seed, budget), pour
# vérifier que l'assemblage place chaque cellule au bon indice d'axe.
# =============================================================================


def _valeur_unique(L: int, seed: int, budget: int) -> float:
    iB = BUDGETS.index(budget)
    return float(L * 1_000_000 + seed * 1_000 + iB)


def _mesures_synthetiques(L_list: tuple[int, ...]) -> dict:
    mesures = {}
    for L in L_list:
        for seed in SEEDS:
            ma1 = {b: _valeur_unique(L, seed, b) for b in BUDGETS}
            ma2 = {b: _valeur_unique(L, seed, b) + 0.5 for b in BUDGETS}
            mesures[(L, seed)] = (ma1, ma2)
    return mesures


# =============================================================================
# 1. Schéma (shapes/dtypes) de l'assemblage -- sur ma1/ma2 synthétiques
# =============================================================================


def test_schema_npz_v2():
    mesures = _mesures_synthetiques(m16.L_V2)
    ma1, ma2 = m16._assemble_grille(mesures, m16.L_V2)

    assert ma1.shape == (1, 5, 7)
    assert ma2.shape == (1, 5, 7)
    assert ma1.dtype == np.float64
    assert ma2.dtype == np.float64
    for iS, seed in enumerate(SEEDS):
        for iB, budget in enumerate(BUDGETS):
            assert ma1[0, iS, iB] == _valeur_unique(160, seed, budget)
            assert ma2[0, iS, iB] == _valeur_unique(160, seed, budget) + 0.5

    sous_jnd = m16._construit_sous_jnd(ma1, ma2)
    assert sous_jnd.shape == (1, 5, 7, 4)
    assert sous_jnd.dtype == np.bool_


def test_schema_npz_ferm_shuf():
    mesures = _mesures_synthetiques(m16.L_CONTROL)
    ma1, ma2 = m16._assemble_grille(mesures, m16.L_CONTROL)

    assert ma1.shape == (2, 5, 7)
    assert ma2.shape == (2, 5, 7)
    assert ma1.dtype == np.float64
    assert ma2.dtype == np.float64
    for iL, L in enumerate(m16.L_CONTROL):
        for iS, seed in enumerate(SEEDS):
            for iB, budget in enumerate(BUDGETS):
                assert ma1[iL, iS, iB] == _valeur_unique(L, seed, budget)
                assert ma2[iL, iS, iB] == _valeur_unique(L, seed, budget) + 0.5

    sous_jnd = m16._construit_sous_jnd(ma1, ma2)
    assert sous_jnd.shape == (2, 5, 7, 4)
    assert sous_jnd.dtype == np.bool_


# =============================================================================
# 2. `_construit_sous_jnd` RÉUTILISE `sous_jnd_a_jnd`, ne le réimplémente pas
# =============================================================================


def test_reutilise_sous_jnd_a_jnd():
    rng = np.random.default_rng(42)
    ma1 = rng.uniform(0.0, 0.1, size=(2, 5, 7))
    ma2 = rng.uniform(0.0, 0.1, size=(2, 5, 7))
    ma1[0, 0, 0] = np.nan  # cellule indéfinie -- convention NaN -> False (module gelé)

    resultat = m16._construit_sous_jnd(ma1, ma2)
    assert resultat.dtype == np.bool_
    for k, jnd in enumerate(JND_LIST):
        attendu = sous_jnd_a_jnd(ma1, ma2, jnd)
        assert np.array_equal(resultat[..., k], attendu), f"jnd={jnd} : désaccord"
        assert resultat[0, 0, 0, k] == np.False_  # NaN -> False, jamais True


# =============================================================================
# 3. Ordre / grilles conformes aux constantes importées (identité d'objet)
# =============================================================================


def test_budgets_et_ordre():
    assert m16.L_V2 == (160,)
    assert m16.L_CONTROL == (10, 160)

    # Identité d'objet (import, pas recopie ni recalcul) avec les modules gelés.
    assert m16.BUDGETS is BUDGETS
    assert m16.SEEDS is SEEDS
    assert m16.JND_LIST is JND_LIST

    assert list(m16.SEEDS) == [101, 102, 103, 104, 105]
    assert list(m16.BUDGETS) == [32, 64, 128, 256, 400, 1024, 2048]
    assert list(m16.JND_LIST) == [0.02, 0.03, 0.04, 0.05]
    assert m16.BUDGET_FERM == 32


# =============================================================================
# 4. Garde d'entrée : histoire L=160 manquante -> RuntimeError clair (BLOCKED)
# =============================================================================


def test_charge_historique_manquante_leve_erreur_claire(monkeypatch, tmp_path):
    monkeypatch.setattr(m16, "_npz_path_h160",
                        lambda seed: tmp_path / f"h160_s{seed}_inexistant.npz")
    with pytest.raises(RuntimeError, match="Histoire L=160 manquante"):
        m16._load_history_160(101)


# =============================================================================
# 5. Non-régression déterminisme RÉELLE : ferm@L=10 re-mesuré == measures_qt.npz
#    (bit-à-bit) -- SKIP tant que les 5 h160_s{seed}.npz (Task 1) n'existent pas.
# =============================================================================

_H160_SEEDS_PRESENTS = all(m16._npz_path_h160(s).exists() for s in SEEDS)
_DONNEES_PRETES = _H160_SEEDS_PRESENTS and MEASURES_QT_PATH.exists()


@pytest.mark.skipif(
    not _DONNEES_PRETES,
    reason="Histoires L=160 (h160_s{seed}.npz, Task 1) pas encore générées par "
           "le contrôleur en arrière-plan (~80 min) -- ce test de non-régression "
           "sera levé et confirmé par le contrôleur une fois les 5 fichiers "
           "présents (measures_qt.npz, lui, existe déjà).",
)
def test_non_regression_controle_L10():
    with np.load(MEASURES_QT_PATH, allow_pickle=False) as d:
        ma1_ferm_stocke = d["ma1_ferm"]
        ma2_ferm_stocke = d["ma2_ferm"]
    iL10 = L_LIST.index(10)

    mesures = {}
    for seed in SEEDS:
        hist = m16._load_history_160(seed)
        ma1, ma2, _center = m16._mesurer_cellule("ferm", 10, seed, hist)
        mesures[(10, seed)] = (ma1, ma2)
    ma1_recalcule, ma2_recalcule = m16._assemble_grille(mesures, (10,))

    assert np.array_equal(ma1_recalcule[0], ma1_ferm_stocke[iL10])
    assert np.array_equal(ma2_recalcule[0], ma2_ferm_stocke[iL10])
