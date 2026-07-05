"""Arc C / C-1a Task 1 — tests du ratio χ_rms / C_michelson mesuré sur les
stimuli RÉELS (bande "4-7"), cf. brief `.superpowers/sdd/
arcC-c1a-task1-ratio-brief.md`. AUCUNE simulation, AUCUN RNG : mesure pure sur
des champs déjà commités (`outputs/arcA/histories/h_s{seed}.npz`).

Familles (brief) :
  1. Sanity gravé par la mission : grating sinusoïdal pur dans la bande 4-7 ->
     ratio analytique = 1/√2 (< 1 % d'erreur numérique).
  2. `chi_rms` == `chi_bands(A)[0]["4-7"]` -- réutilisation R1, pas une
     réimplémentation.
  3. `band_ac` (composante spatiale de la bande) de moyenne nulle -- DC exclu.
  4. Cohérence Parseval : `std(band_ac) == chi_rms * mean_A`.
  5. Sources incluses : 19 paires (seed, L), (103, 10) exclue.
  6. Déterminisme de l'agrégation (fonctions pures, aucune RNG)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.run_arcA_revalidate import S_HALF_OP
from scripts.run_arcC_ratio_rms_michelson import (_composante_bande, agrege_famille,
                                                  charge_champ_s_true, charge_sources_incluses,
                                                  construit_mesures, mesure_ratio_champ)
from src.albedo import albedo, chi_bands


# =============================================================================
# Famille 1 : sanity sinus-pur -- ratio analytique 1/√2 (GRAVÉ par la mission)
# =============================================================================


def test_sanity_sinus_pur_ratio_analytique():
    N = 64
    A0, a, freq = 0.5, 0.1, 5  # freq = 5 c/domaine, DANS la bande 4-7
    x = np.arange(N)
    ligne = A0 + a * np.cos(2.0 * np.pi * freq * x / N)
    A = np.tile(ligne, (N, 1))  # array[y, x] -- constant en y, varie en x

    resultat = mesure_ratio_champ(A)

    assert abs(resultat["ratio"] - 1.0 / np.sqrt(2.0)) < 0.01
    assert abs(resultat["c_michelson"] - a / A0) < 1e-3
    assert abs(resultat["chi_rms"] - (a / np.sqrt(2.0)) / A0) < 1e-3


def test_sanity_sinus_pur_mean_albedo_et_diagnostics():
    """Non-régression légère : mean_albedo == A0, porteuse détectée sur "4-7"
    (seule bande avec de l'énergie AC), pas de débordement < 0 (A_min=0.4)."""
    N = 64
    A0, a, freq = 0.5, 0.1, 5
    x = np.arange(N)
    ligne = A0 + a * np.cos(2.0 * np.pi * freq * x / N)
    A = np.tile(ligne, (N, 1))

    resultat = mesure_ratio_champ(A)
    assert resultat["mean_albedo"] == pytest.approx(A0, abs=1e-9)
    assert resultat["est_porteuse"] is True
    assert resultat["bande_porteuse_max"] == "4-7"
    assert resultat["composante_deborde_zero"] is False


# =============================================================================
# Famille 2 : réutilisation EXACTE de chi_bands (pas de réimplémentation)
# =============================================================================


def test_chi_rms_est_bien_chi_bands_4_7():
    sources = charge_sources_incluses()
    seed, L = sources[0]
    s_true = charge_champ_s_true(seed, L)
    A = albedo(s_true, S_HALF_OP)

    resultat = mesure_ratio_champ(A)

    assert resultat["chi_rms"] == chi_bands(A)[0]["4-7"]


# =============================================================================
# Famille 3 & 4 : composante spatiale de bande -- moyenne nulle + Parseval
# =============================================================================


def test_band_ac_moyenne_nulle():
    sources = charge_sources_incluses()
    seed, L = sources[0]
    s_true = charge_champ_s_true(seed, L)
    A = albedo(s_true, S_HALF_OP)

    band_ac = _composante_bande(A)

    assert abs(np.mean(band_ac)) < 1e-9


def test_coherence_parseval():
    sources = charge_sources_incluses()
    seed, L = sources[1]
    s_true = charge_champ_s_true(seed, L)
    A = albedo(s_true, S_HALF_OP)

    band_ac = _composante_bande(A)
    resultat = mesure_ratio_champ(A)

    assert abs(np.std(band_ac) - resultat["chi_rms"] * resultat["mean_albedo"]) < 1e-9


# =============================================================================
# Famille 5 : sources incluses -- 19 paires, (103, 10) exclue
# =============================================================================


def test_sources_incluses_exclut_103_10():
    sources = charge_sources_incluses()

    assert len(sources) == 19
    assert (103, 10) not in sources


def test_sources_incluses_triees_par_seed_puis_l():
    sources = charge_sources_incluses()
    assert sources == sorted(sources)


# =============================================================================
# Famille 6 : déterminisme du cœur d'agrégation
# =============================================================================


def test_determinisme():
    sources = charge_sources_incluses()

    resultat_1 = construit_mesures(sources)
    resultat_2 = construit_mesures(sources)

    assert resultat_1 == resultat_2


def test_agrege_famille_dispersion_rapportee_pas_seulement_la_mediane():
    """Garde d'honnêteté (brief) : `agrege_famille` rapporte IQR + min/max +
    la liste complète, pas seulement `ratio_median`."""
    entrees = [dict(seed=1, L=10, ratio=0.10), dict(seed=1, L=20, ratio=0.20),
              dict(seed=1, L=40, ratio=0.30), dict(seed=1, L=80, ratio=0.40)]

    agg = agrege_famille(entrees)

    assert agg["n"] == 4
    assert agg["ratio_median"] == pytest.approx(0.25)
    assert agg["ratio_q1"] == pytest.approx(np.percentile([0.10, 0.20, 0.30, 0.40], 25.0))
    assert agg["ratio_q3"] == pytest.approx(np.percentile([0.10, 0.20, 0.30, 0.40], 75.0))
    assert agg["ratio_iqr"] == pytest.approx(agg["ratio_q3"] - agg["ratio_q1"])
    assert agg["ratio_min"] == pytest.approx(0.10)
    assert agg["ratio_max"] == pytest.approx(0.40)
    assert agg["par_source"] == entrees
