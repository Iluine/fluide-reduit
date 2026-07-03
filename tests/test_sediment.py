import numpy as np
import pytest

from config import GridConfig
from src.sediment import (default_terrain, run_episode, run_history, SedimentParams,
                          KD_CALIBRE, _relax_episode, _integrate_exner, _DRY_EPS)

GRID = GridConfig()
B0 = default_terrain(GRID)
RELIEF = float(B0.max() - B0.min())

# Centre de pulse fixe utilisé par les tests d'invariants (déterministe, pas tiré
# par seed) : documenté empiriquement (task-1-report.md) comme laissant ~560/4096
# cellules jamais mouillées sur l'épisode complet -- nécessaire pour tester
# "s ne bouge pas sur cellules jamais mouillées".
_CENTER = (0.5, 0.5)

# SedimentParams réduits (n_episodes/N_settle petits) pour les tests de PLOMBERIE
# (pureté, déterminisme, path-dependence) : ces propriétés ne dépendent pas de la
# magnitude de N_settle, seulement de la logique -- les garder à N_settle=600
# n'apporterait rien de plus et coûterait cher (le solveur tourne de toute façon
# jusqu'à _T_END_RELAX, indépendamment de N_settle, cf. sediment.py). Les tests
# d'invariants PHYSIQUES (masse, jamais-mouillé) utilisent eux les valeurs FIGÉES
# par défaut (N_settle=600) car c'est le contrat de l'épisode.
_SMALL_PARAMS = SedimentParams(N_settle=20, save_every=2)


def test_default_terrain_shape_and_relief():
    assert B0.shape == (64, 64)
    # relief > 0 et cohérent avec pente (0.5) + plus grande bosse (0.30)
    assert 0.5 < RELIEF < 0.9


def test_kd_calibre_is_frozen_and_positive():
    assert KD_CALIBRE > 0.0
    assert SedimentParams().k_d == KD_CALIBRE
    assert SedimentParams().k_e == pytest.approx(KD_CALIBRE / 5.0)


# Un seul épisode complet (N_settle=600, params par défaut) partagé entre les tests
# d'invariants physiques via une fixture MODULE (pas de re-calcul), pour ne payer le
# coût de la relaxation qu'une fois (~quelques secondes, cf. task-1-report.md pour
# le budget de calcul).
@pytest.fixture(scope="module")
def full_episode_trajectory():
    s0 = np.zeros_like(B0)
    b_eff = B0 + s0
    params = SedimentParams()
    times, hs, hus, hvs = _relax_episode(b_eff, _CENTER, params)
    return times, hs, hus, hvs, params, s0


def test_mass_conserved_within_measured_tolerance(full_episode_trajectory):
    times, hs, _hus, _hvs, _params, _s0 = full_episode_trajectory
    mass0 = float(hs[0].sum())
    massN = float(hs[-1].sum())
    rel_drift = abs(massN - mass0) / mass0
    # NON 1e-8 : cf. docstring module sediment.py -- simulate_wetdry_o2 (figé, non
    # modifiable) n'est pas exactement conservatif en masse une fois qu'un front
    # mouillé/sec atteint un mur réfléchissant (pente MUSCL asymétrique
    # fantôme/réel). Mesuré empiriquement pour ce centre : ~13.1 %. Tolérance fixée
    # à 0.20 (marge sur la mesure), PAS 1e-8 (résolution d'ambiguïté invalidée
    # empiriquement -- cf. task-1-report.md, DONE_WITH_CONCERNS).
    assert rel_drift < 0.20, f"dérive de masse inattendue : {rel_drift:.4f}"


def test_s_nonnegative_after_episode(full_episode_trajectory):
    times, hs, hus, hvs, params, s0 = full_episode_trajectory
    s_new = _integrate_exner(s0, times, hs, hus, hvs, params)
    assert np.all(s_new >= 0.0)


def test_s_unchanged_on_never_wetted_cells(full_episode_trajectory):
    times, hs, hus, hvs, params, s0 = full_episode_trajectory
    ever_wet = (hs > _DRY_EPS).any(axis=0)
    never_wet = ~ever_wet
    assert never_wet.sum() > 0, "aucune cellule jamais mouillee -- centre a revoir"
    s_new = _integrate_exner(s0, times, hs, hus, hvs, params)
    assert np.array_equal(s_new[never_wet], s0[never_wet])
    assert np.all(s_new[never_wet] == 0.0)


def test_run_episode_is_pure_no_mutation():
    s0 = np.zeros_like(B0)
    b0_copy = B0.copy()
    s0_copy = s0.copy()
    _ = run_episode(s0, B0, _CENTER, _SMALL_PARAMS)
    assert np.array_equal(B0, b0_copy)
    assert np.array_equal(s0, s0_copy)


def test_run_episode_deterministic_bitwise():
    s0 = np.zeros_like(B0)
    s1 = run_episode(s0, B0, _CENTER, _SMALL_PARAMS)
    s2 = run_episode(s0, B0, _CENTER, _SMALL_PARAMS)
    assert np.array_equal(s1, s2)


def test_run_history_checkpoints_and_determinism():
    hist1 = run_history(seed=7, n_episodes=3, b0=B0, params=_SMALL_PARAMS,
                        checkpoints={1, 2, 3})
    hist2 = run_history(seed=7, n_episodes=3, b0=B0, params=_SMALL_PARAMS,
                        checkpoints={1, 2, 3})
    assert set(hist1.keys()) == {1, 2, 3}
    for k in (1, 2, 3):
        assert np.array_equal(hist1[k], hist2[k])
    # dépôt cumulatif net : le checkpoint final n'est pas identiquement nul
    assert float(hist1[3].max()) > 0.0


def test_run_history_default_checkpoint_is_final_only():
    hist = run_history(seed=7, n_episodes=3, b0=B0, params=_SMALL_PARAMS)
    assert set(hist.keys()) == {3}


def test_path_dependence_two_seeds_differ():
    hist_a = run_history(seed=101, n_episodes=3, b0=B0, params=_SMALL_PARAMS)
    hist_b = run_history(seed=102, n_episodes=3, b0=B0, params=_SMALL_PARAMS)
    assert not np.array_equal(hist_a[3], hist_b[3])


def test_path_dependence_reversed_order_differs():
    rng = np.random.default_rng(2026)
    centers = [(float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.15, 0.85)))
              for _ in range(3)]

    def _run(order):
        s = np.zeros_like(B0)
        for c in order:
            s = run_episode(s, B0, c, _SMALL_PARAMS)
        return s

    s_fwd = _run(centers)
    s_rev = _run(list(reversed(centers)))
    assert not np.array_equal(s_fwd, s_rev)
