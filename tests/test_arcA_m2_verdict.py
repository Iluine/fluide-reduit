"""Tests Manche 2 (§A13) — combinateur de verdict
(`scripts/run_arcA_m2_verdict.py`). SYNTHÉTIQUES et rapides : le combinateur
est pur, on le teste sur des tenseurs fabriqués (leçon §A12 : vérifier
empiriquement chaque branche, y compris la sortie sûre)."""
from __future__ import annotations

import numpy as np

import scripts.run_arcA_m2_verdict as v
from scripts.run_arcA_m2_grille import BUDGETS, DTS, N_EMISSIONS, SEEDS

_SHAPE = (len(SEEDS), len(DTS), len(BUDGETS), N_EMISSIONS)
_JND = 0.07334065957385666


def _tenseur(valeur: float) -> np.ndarray:
    """Tenseur constant, Δχ_1 = 0 structurel respecté."""
    t = np.full(_SHAPE, valeur, dtype=np.float64)
    t[..., 0] = 0.0
    return t


# =============================================================================
# kstar_chaine — définitions (a) + (b)
# =============================================================================


def test_kstar_ferme_au_plus_petit_budget():
    t = _tenseur(0.01)
    ks, detail = v.kstar_chaine(t[:, 0], _JND)
    assert ks == 128
    assert detail["128"]["ferme"] is True


def test_kstar_inf_si_rien_ne_ferme():
    t = _tenseur(0.20)
    ks, _ = v.kstar_chaine(t[:, 0], _JND)
    assert ks == v.K_INF


def test_kstar_exclut_emission_1():
    """(a) : un Δχ_1 énorme ne doit JAMAIS compter (structurellement 0 en
    vrai run ; le combinateur ne doit pas pouvoir être piégé par lui)."""
    t = _tenseur(0.01)
    t[..., 0] = 99.0
    ks, _ = v.kstar_chaine(t[:, 0], _JND)
    assert ks == 128


def test_kstar_clause_deux_jnd_par_seed():
    """(b) : médiane sous JND mais UNE seed > 2·JND → le budget ne ferme pas."""
    t = _tenseur(0.01)
    t[0, :, :, 3] = 3.0 * _JND  # seed 101 explose à l'émission 4, tous k
    ks, detail = v.kstar_chaine(t[:, 0], _JND)
    assert ks == v.K_INF
    assert detail["128"]["ferme"] is False
    assert detail["128"]["mediane_max"] < _JND  # c'est bien la clause seed


def test_kstar_seed_a_exactement_2jnd_ne_bloque_pas():
    """Frontière gravée §A13-1 : « aucune seed > 2×JND » — exactement 2·JND
    n'est PAS une violation (inégalité stricte)."""
    t = _tenseur(0.01)
    t[0, :, :, 3] = 2.0 * _JND
    ks, _ = v.kstar_chaine(t[:, 0], _JND)
    assert ks == 128


# =============================================================================
# composition_partout — définition (c)
# =============================================================================


def test_composition_partout_vrai_si_mediane_traverse_tous_budgets():
    t = _tenseur(0.20)
    assert v.composition_partout(t[:, 0], _JND) is True


def test_composition_partout_faux_si_un_budget_borne():
    t = _tenseur(0.20)
    t[:, :, 2, :] = 0.01  # k=400 borné sous JND
    t[..., 0] = 0.0
    assert v.composition_partout(t[:, 0], _JND) is False


# =============================================================================
# combinateur — les 4 sorties §A13-2, chacune atteinte empiriquement
# =============================================================================


def _colonnes(t: np.ndarray) -> list[dict]:
    return [v.verdict_colonne(t, jnd) for jnd in (0.0603, _JND, 0.0867)]


def test_verdict_registre_ferme():
    t = _tenseur(0.01)
    assert v.combinateur(_colonnes(t), instrument_muet=False) == "REGISTRE_FERME"


def test_verdict_mur_registre():
    t = _tenseur(0.30)
    assert v.combinateur(_colonnes(t), instrument_muet=False) == "MUR_REGISTRE"


def test_verdict_indetermine_si_colonnes_divergent():
    """ic_haut ferme (Δχ entre pin et ic_haut à k=400 seulement... ) — cas
    §A12-like : Δχ médian à 0.08, entre pin (0.0733) et ic_haut (0.0867) :
    la colonne ic_haut ferme, ic_bas et pin traversent → divergence."""
    t = _tenseur(0.08)
    assert v.combinateur(_colonnes(t), instrument_muet=False) == "INDETERMINE_JND"


def test_verdict_indetermine_cas_ni_ferme_ni_mur():
    """(d) sortie sûre : médiane bornée sous JND partout mais UNE seed > 2·JND
    → aucune colonne ne ferme, aucune composition de médiane → INDÉTERMINÉ,
    jamais MUR ni FERME par défaut."""
    t = _tenseur(0.01)
    t[0, :, :, 3] = 3.0  # une seed folle, tous Δt, tous k
    assert v.combinateur(_colonnes(t), instrument_muet=False) == "INDETERMINE_JND"


def test_verdict_instrument_muet_court_circuite():
    t = _tenseur(0.01)  # aurait été REGISTRE_FERME
    assert v.combinateur(_colonnes(t), instrument_muet=True) == "INSTRUMENT_MUET"


# =============================================================================
# Figure (fumée) — ne doit pas dépendre d'un verdict particulier
# =============================================================================


def test_figure_kstar_fumee(tmp_path):
    t = _tenseur(0.02)
    pins = {"severe": {"jnd": _JND, "ic": (0.0603, 0.0867)},
            "laxiste": {"jnd": 0.1154, "ic": (0.0937, 0.1387)}}
    colonnes = _colonnes(t)
    document = {"verdict": v.combinateur(colonnes, False), "pins": pins,
                "colonnes_severe": colonnes}
    out = tmp_path / "fig.png"
    v.figure_kstar(document, t, out)
    assert out.exists() and out.stat().st_size > 0
