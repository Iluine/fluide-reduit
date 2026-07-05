"""Arc C / C-1a Task 3 — tests du ratio r0 = 2*RMS/(peak-to-peak), SCALE-
INVARIANT, mesure COMPAGNE du ratio natif χ_rms/C_michelson (décision
PREREGISTRATION.md pocCascade2phys §C12, commit `122e82a`), cf. brief
`.superpowers/sdd/arcC-c1a-task3-r0-brief.md`. AUCUNE simulation, AUCUN RNG :
mesure pure sur des champs déjà commités (`outputs/arcA/histories/h_s{seed}.npz`).

Familles (brief) :
  1. Sanity gravé par la mission : grating sinusoïdal pur dans la bande 4-7
     -> r0 analytique = 1/√2 (< 1 % d'erreur numérique).
  2. r0 est invariant à l'échelle ET au pédestal (propriété qui MOTIVE r0).
  3. Cohérence avec le natif : r0 == 2*chi_rms_natif*mean_A/pic_a_pic --
     preuve que r0 parle de la MÊME bande que le natif.
  4. Sources incluses : 19 paires (seed, L), (103, 10) exclue.
  5. Déterminisme de l'agrégation (fonctions pures, aucune RNG).
  6. Garde-fou anti-régression : le module natif n'a pas été modifié."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.run_arcA_revalidate import S_HALF_OP
from scripts.run_arcC_ratio_r0 import (_construit_entree_source, agrege_famille_r0,
                                       construit_mesures_r0, mesure_r0_champ)
from scripts.run_arcC_ratio_rms_michelson import (_composante_bande, charge_champ_s_true,
                                                  charge_sources_incluses, mesure_ratio_champ)
from src.albedo import albedo


# =============================================================================
# Famille 1 : sanity sinus-pur -- r0 analytique 1/√2 (GRAVÉ par la mission)
# =============================================================================


def test_sanity_sinus_pur_r0():
    N = 64
    A0, a, freq = 0.5, 0.1, 5  # freq = 5 c/domaine, DANS la bande 4-7
    x = np.arange(N)
    ligne = A0 + a * np.cos(2.0 * np.pi * freq * x / N)
    A = np.tile(ligne, (N, 1))  # array[y, x] -- constant en y, varie en x

    resultat = mesure_r0_champ(A)

    assert abs(resultat["r0"] - 1.0 / np.sqrt(2.0)) < 0.01
    assert abs(resultat["rms_bande"] - a / np.sqrt(2.0)) < 1e-3
    assert abs(resultat["pic_a_pic"] - 2.0 * a) < 1e-3
    assert resultat["est_porteuse"] is True


# =============================================================================
# Famille 2 : r0 invariant à l'échelle ET au pédestal (motive r0)
# =============================================================================


def test_r0_scale_invariant():
    sources = charge_sources_incluses()
    seed, L = sources[0]
    s_true = charge_champ_s_true(seed, L)
    A = albedo(s_true, S_HALF_OP)

    r0_A = mesure_r0_champ(A)["r0"]
    r0_Ac = mesure_r0_champ(A + 0.3)["r0"]  # décalage DC constant, pédestal

    assert abs(r0_A - r0_Ac) < 1e-9


# =============================================================================
# Famille 3 : cohérence avec le natif -- MÊME bande, pas une reconstruction
# =============================================================================


def test_r0_coherent_avec_chi_rms_natif():
    sources = charge_sources_incluses()
    seed, L = sources[1]
    s_true = charge_champ_s_true(seed, L)
    A = albedo(s_true, S_HALF_OP)

    m_r0 = mesure_r0_champ(A)
    m_natif = mesure_ratio_champ(A)

    r0_attendu = (2.0 * m_natif["chi_rms"] * m_natif["mean_albedo"]) / m_r0["pic_a_pic"]

    assert abs(m_r0["r0"] - r0_attendu) < 1e-9


def test_band_ac_reutilise_est_bien_celle_du_natif():
    """r0 doit être calculé sur la MÊME composante de bande que le natif --
    pas une reconstruction parallèle : `_composante_bande` (natif) doit
    produire exactement le `band_ac` dont dérive `mesure_r0_champ`."""
    sources = charge_sources_incluses()
    seed, L = sources[2]
    s_true = charge_champ_s_true(seed, L)
    A = albedo(s_true, S_HALF_OP)

    band_ac = _composante_bande(A)
    m_r0 = mesure_r0_champ(A)

    assert abs(float(np.std(band_ac)) - m_r0["rms_bande"]) < 1e-9
    assert abs(float(band_ac.max() - band_ac.min()) - m_r0["pic_a_pic"]) < 1e-9


# =============================================================================
# Famille 4 : sources incluses -- 19 paires, (103, 10) exclue
# =============================================================================


def test_sources_incluses_exclut_103_10():
    sources = charge_sources_incluses()

    assert len(sources) == 19
    assert (103, 10) not in sources


# =============================================================================
# Famille 5 : déterminisme du cœur d'agrégation
# =============================================================================


def test_determinisme():
    sources = charge_sources_incluses()

    resultat_1 = construit_mesures_r0(sources)
    resultat_2 = construit_mesures_r0(sources)

    assert resultat_1 == resultat_2


def test_agrege_famille_r0_dispersion_rapportee_pas_seulement_la_mediane():
    """Garde d'honnêteté (§C12) : `agrege_famille_r0` rapporte IQR + min/max +
    la liste complète, pas seulement `r0_median`."""
    entrees = [dict(seed=1, L=10, r0=0.10, est_porteuse=True),
              dict(seed=1, L=20, r0=0.20, est_porteuse=True),
              dict(seed=1, L=40, r0=0.30, est_porteuse=True),
              dict(seed=1, L=80, r0=0.40, est_porteuse=True)]

    agg = agrege_famille_r0(entrees)

    assert agg["n"] == 4
    assert agg["r0_median"] == pytest.approx(0.25)
    assert agg["r0_q1"] == pytest.approx(np.percentile([0.10, 0.20, 0.30, 0.40], 25.0))
    assert agg["r0_q3"] == pytest.approx(np.percentile([0.10, 0.20, 0.30, 0.40], 75.0))
    assert agg["r0_iqr"] == pytest.approx(agg["r0_q3"] - agg["r0_q1"])
    assert agg["r0_min"] == pytest.approx(0.10)
    assert agg["r0_max"] == pytest.approx(0.40)
    assert agg["par_source"] == entrees


def test_construit_entree_source_forme_gravee():
    """Forme JSON gravée par le brief : seed/L/r0/est_porteuse UNIQUEMENT
    (pas rms_bande/pic_a_pic, diagnostics internes non exposés)."""
    sources = charge_sources_incluses()
    seed, L = sources[0]
    s_true = charge_champ_s_true(seed, L)
    A = albedo(s_true, S_HALF_OP)

    entree = _construit_entree_source(seed, L, A)

    assert set(entree.keys()) == {"seed", "L", "r0", "est_porteuse"}
    assert entree["seed"] == seed
    assert entree["L"] == L


# =============================================================================
# Famille 6 : garde-fou anti-régression -- le natif n'a pas été modifié
# =============================================================================


def test_natif_non_modifie():
    N = 64
    A0, a, freq = 0.5, 0.1, 5
    x = np.arange(N)
    ligne = A0 + a * np.cos(2.0 * np.pi * freq * x / N)
    A = np.tile(ligne, (N, 1))

    resultat = mesure_ratio_champ(A)

    assert set(resultat.keys()) == {"chi_rms", "c_michelson", "ratio", "mean_albedo",
                                    "est_porteuse", "composante_deborde_zero",
                                    "bande_porteuse_max"}
    assert abs(resultat["ratio"] - 1.0 / np.sqrt(2.0)) < 0.01
