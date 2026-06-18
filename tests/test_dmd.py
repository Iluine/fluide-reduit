import numpy as np
import pytest

from src.dmd import fit_dmd, rollout, spectral_radius


def _make_trajectory(A0, z0, T):
    z = np.zeros((A0.shape[0], T))
    z[:, 0] = z0
    for t in range(T - 1):
        z[:, t + 1] = A0 @ z[:, t]
    return z


def test_fit_dmd_recovers_linear_operator():
    rng = np.random.default_rng(0)
    k = 4
    A0 = 0.95 * np.array([[np.cos(0.2), -np.sin(0.2), 0, 0],
                          [np.sin(0.2), np.cos(0.2), 0, 0],
                          [0, 0, 0.9, 0],
                          [0, 0, 0, 0.8]])
    trajs = [_make_trajectory(A0, rng.random(k), 30) for _ in range(3)]
    A = fit_dmd(trajs)
    assert A.shape == (k, k)
    assert np.linalg.norm(A - A0) / np.linalg.norm(A0) < 1e-8


def test_rollout_matches_truth_for_known_system():
    k = 3
    A0 = np.diag([0.99, 0.95, 0.9])
    z0 = np.array([1.0, 1.0, 1.0])
    true = _make_trajectory(A0, z0, 20)
    pred = rollout(A0, z0, 19)
    assert pred.shape == (k, 20)
    assert np.allclose(pred, true)


def test_spectral_radius():
    A0 = np.diag([0.5, 1.2, 0.3])
    assert abs(spectral_radius(A0) - 1.2) < 1e-9


def test_fit_dmd_raises_on_empty_input():
    with pytest.raises(ValueError):
        fit_dmd([])
