import numpy as np

from src.pod import (stack_snapshots, unstack, fit_pod, encode, decode,
                     cumulative_energy)


def _toy_field(T=20, H=8, W=8, seed=0):
    rng = np.random.default_rng(seed)
    # signal de faible rang : quelques modes spatiaux x temporels
    H_, W_ = H, W
    modes = rng.random((3, H_, W_))
    coeffs = rng.random((T, 3))
    h = 1.0 + coeffs[:, 0, None, None] * modes[0]
    u = 0.1 * coeffs[:, 1, None, None] * modes[1]
    v = 0.1 * coeffs[:, 2, None, None] * modes[2]
    return h, u, v


def test_stack_unstack_roundtrip():
    h, u, v = _toy_field()
    X = stack_snapshots(h, u, v)
    assert X.shape == (3 * 8 * 8, 20)
    h2, u2, v2 = unstack(X, 8, 8)
    assert np.allclose(h2, h) and np.allclose(u2, u) and np.allclose(v2, v)


def test_cumulative_energy_monotone_and_reaches_one():
    s = np.array([3.0, 2.0, 1.0, 0.0])
    e = cumulative_energy(s)
    assert e.shape == (4,)
    assert np.all(np.diff(e) >= -1e-12)
    assert abs(e[-1] - 1.0) < 1e-12


def test_full_rank_reconstruction_is_near_exact():
    h, u, v = _toy_field()
    X = stack_snapshots(h, u, v)
    basis = fit_pod(X, energy_threshold=1.0, max_modes=64)
    z = encode(basis, X)
    X_rec = decode(basis, z)
    assert np.linalg.norm(X_rec - X) / np.linalg.norm(X) < 1e-8


def test_reconstruction_error_decreases_with_k():
    h, u, v = _toy_field(T=40, seed=3)
    X = stack_snapshots(h, u, v)
    errs = []
    for thr in (0.5, 0.9, 0.999):
        basis = fit_pod(X, energy_threshold=thr, max_modes=64)
        z = encode(basis, X)
        errs.append(np.linalg.norm(decode(basis, z) - X) / np.linalg.norm(X))
    assert errs[0] >= errs[1] >= errs[2] - 1e-12


import numpy as np

from src.pod import (fit_pod, encode, decode, _channel_scale,
                     stack_height, unstack_height)


def test_channel_scale_single_channel_is_global_std():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 3.0, size=(100, 20))
    scale = _channel_scale(X, n_channels=1)
    assert np.allclose(scale, scale[0])              # un seul bloc -> scale uniforme
    assert np.isclose(scale[0], X.std())             # = écart-type global


def test_channel_scale_default_is_three_blocks_unchanged():
    rng = np.random.default_rng(1)
    X = np.concatenate([rng.normal(0, 1, (30, 10)),
                        rng.normal(0, 5, (30, 10)),
                        rng.normal(0, 9, (30, 10))], axis=0)
    scale = _channel_scale(X)  # défaut n_channels=3, comportement POC
    assert not np.isclose(scale[0], scale[40])       # blocs distincts
    assert not np.isclose(scale[40], scale[70])


def test_height_pod_roundtrip():
    rng = np.random.default_rng(2)
    H = W = 8
    T = 40
    h_seq = rng.normal(1.0, 0.2, size=(T, H, W))
    X = stack_height(h_seq)
    assert X.shape == (H * W, T)
    basis = fit_pod(X, energy_threshold=0.999, max_modes=64, n_channels=1)
    z = encode(basis, X)
    Xr = decode(basis, z)
    seq_r = unstack_height(Xr, H, W)
    assert seq_r.shape == (T, H, W)
    # reconstruction fidèle au seuil d'énergie choisi
    assert np.linalg.norm(seq_r - h_seq) / np.linalg.norm(h_seq) < 0.05
