import numpy as np

from src.metrics import (total_mass, mass_series, relative_l2_error,
                         error_growth, seam_jump)


def test_total_mass_and_series():
    h = np.ones((4, 4))
    assert total_mass(h, 1.0, 1.0) == 16.0
    assert total_mass(h, 0.5, 2.0) == 16.0
    seq = np.stack([h, 2 * h, 3 * h])  # (3,4,4)
    assert np.allclose(mass_series(seq, 1.0, 1.0), [16.0, 32.0, 48.0])


def test_relative_l2_error():
    a = np.array([3.0, 4.0])
    assert relative_l2_error(a, a) == 0.0
    assert isinstance(relative_l2_error(a, a), float)
    # pred=0 vs true=a -> ||a||/||a|| = 1
    assert abs(relative_l2_error(np.zeros_like(a), a) - 1.0) < 1e-9


def test_error_growth_shape_and_zero():
    seq = np.random.default_rng(0).random((6, 5, 5))
    g = error_growth(seq, seq)
    assert g.shape == (6,)
    assert np.allclose(g, 0.0)


def test_seam_jump_zero_on_constant_field():
    field = np.full((16, 16), 3.0)
    assert seam_jump(field, 4, 4, 6) == 0.0


def test_seam_jump_positive_on_discontinuity():
    field = np.zeros((16, 16))
    field[4:10, 4:10] = 1.0  # bloc à 1 entouré de 0 -> saut de 1 au bord
    j = seam_jump(field, 4, 4, 6)
    assert abs(j - 1.0) < 1e-9
