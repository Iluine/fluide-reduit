import numpy as np
import pytest

from src.mass_projection import project_mass, project_mass_series
from src.metrics import total_mass


def test_project_mass_hits_target_exactly():
    rng = np.random.default_rng(0)
    h = 1.0 + 0.3 * rng.standard_normal((8, 8))
    target = 50.0
    hc = project_mass(h, target, dx=1.0, dy=1.0)
    assert abs(total_mass(hc, 1.0, 1.0) - target) < 1e-9
    assert hc.shape == h.shape


def test_project_mass_correction_is_uniform():
    # h_corr - h must be a single constant (a DC offset), preserving shape
    h = 1.0 + 0.2 * np.cos(np.linspace(0, 6, 36)).reshape(6, 6)
    hc = project_mass(h, target_mass=40.0, dx=0.5, dy=0.5)
    diff = hc - h
    assert np.allclose(diff, diff.flat[0])  # all equal => uniform offset


def test_project_mass_series_each_frame_hits_target():
    rng = np.random.default_rng(1)
    seq = 1.0 + 0.1 * rng.standard_normal((5, 6, 6))
    target = 36.0
    out = project_mass_series(seq, target, dx=1.0, dy=1.0)
    assert out.shape == seq.shape
    for t in range(seq.shape[0]):
        assert abs(total_mass(out[t], 1.0, 1.0) - target) < 1e-9


def test_project_mass_raises_on_negative_depth():
    # large mass EXCESS to remove + a shallow cell -> uniform subtraction goes negative
    h = np.full((4, 4), 1.0)
    h[0, 0] = 0.01
    # current mass ~15.01 ; target 10 -> delta ~ -0.31 -> h[0,0] negative
    with pytest.raises(AssertionError):
        project_mass(h, target_mass=10.0, dx=1.0, dy=1.0)
