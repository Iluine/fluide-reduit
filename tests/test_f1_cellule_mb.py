"""Tests F1 — M-b tranche-2 : la CELLULE §A15 et l'observable Δχ (§A31-build).

Ce que ces tests protègent :
  1. la cellule est celle GRAVÉE, reconduite sans changement (3 seeds, Δt=4,
     6 émissions) ;
  2. le VERROU des centres : les deux bras doivent parcourir la MÊME
     histoire que le rederive INTOUCHÉ — les centres sont ceux que
     `run_history` tire, vérifiés contre lui, jamais retirés à côté ;
  3. l'observable Δχ est celui gravé (albedo au point S_HALF_OP +
     `delta_chi[...]["max_carrier"]`), primitives EXISTANTES ;
  4. le budget de commit : k_fen aire-proportionnel, cap 10 %."""
from dataclasses import replace

import numpy as np

from config import GridConfig
from src.f1_gpu.cellule_mb import (
    CAP_COMMIT,
    DELTA_T,
    EMISSIONS,
    GRAINES,
    N_EMISSIONS,
    N_EPISODES,
    budget_k_fen,
    centres_cellule,
    dchi,
)
from src.sediment import SedimentParams, default_terrain, run_episode, run_history


def _terrain():
    return default_terrain(GridConfig())


# ----- 1. la cellule gravée §A15, reconduite -----

def test_cellule_a15_reconduite_sans_changement():
    assert GRAINES == (101, 102, 103)
    assert DELTA_T == 4
    assert N_EMISSIONS == 6
    assert EMISSIONS == (4, 8, 12, 16, 20, 24)
    assert N_EPISODES == 24                     # Δt × 6 émissions


# ----- 2. le VERROU des centres contre le rederive intouché -----

def test_centres_sont_ceux_du_rederive_intouche():
    """Verrou : rejouer UN épisode avec `centres_cellule` doit donner
    EXACTEMENT ce que `run_history` donne. Si le tirage divergeait, les deux
    bras ne compareraient pas la même histoire et Δχ serait un artefact."""
    b0 = _terrain()
    params = replace(SedimentParams(), N_settle=20)
    attendu = run_history(101, 1, b0, params, checkpoints={1})[1]
    obtenu = run_episode(np.zeros_like(b0), b0,
                         centres_cellule(101, 1)[0], params)
    np.testing.assert_array_equal(obtenu, attendu)


def test_centres_longueur_et_boite():
    c = centres_cellule(102, N_EPISODES)
    assert len(c) == N_EPISODES
    assert all(0.15 <= y <= 0.85 and 0.15 <= x <= 0.85 for y, x in c)


def test_centres_dependent_du_seed():
    assert centres_cellule(101, 4) != centres_cellule(102, 4)


# ----- 3. l'observable Δχ gravé -----

def test_dchi_identite_nulle():
    """Δχ(x, x) = 0 — l'observable ne fabrique pas d'écart."""
    rng = np.random.default_rng(0)
    s = rng.random((64, 64)) * 0.5
    assert dchi(s, s) == 0.0


def test_dchi_positif_quand_le_champ_diverge():
    rng = np.random.default_rng(1)
    s = rng.random((64, 64)) * 0.5
    autre = s + 0.3 * rng.random((64, 64))
    assert dchi(autre, s) > 0.0


def test_dchi_est_le_max_carrier_au_point_d_operation():
    """L'observable est EXACTEMENT `delta_chi(albedo(·, S_HALF_OP),
    albedo(ref, S_HALF_OP))["max_carrier"]` — primitives existantes, non
    réécrites (règle 2 : espace instrument, jamais l'état)."""
    from scripts.run_arcA_revalidate import S_HALF_OP
    from src.albedo import albedo, delta_chi
    rng = np.random.default_rng(2)
    ref = rng.random((64, 64)) * 0.5
    cand = ref + 0.2 * rng.random((64, 64))
    attendu = delta_chi(albedo(cand, S_HALF_OP),
                        albedo(ref, S_HALF_OP))["max_carrier"]
    assert dchi(cand, ref) == attendu


# ----- 4. budget de commit : k_fen aire-proportionnel, cap 10 % -----

def test_budget_k_fen_aire_proportionnel():
    """Doubler l'aire double le budget (aires choisies pour que le cap tombe
    juste — la propriété est la proportionnalité, pas l'arrondi entier)."""
    assert budget_k_fen(4000) == 2 * budget_k_fen(2000)
    assert budget_k_fen(2000) == 200


def test_cap_dix_pourcent():
    assert CAP_COMMIT == 0.10
    assert budget_k_fen(64 * 64) == int(0.10 * 64 * 64)


def test_budget_au_moins_un_float():
    """Une fenêtre minuscule garde un budget non nul (sinon le canal serait
    muet par arrondi, pas par seuil)."""
    assert budget_k_fen(4) >= 1
