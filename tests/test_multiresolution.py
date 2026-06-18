import numpy as np

from config import GridConfig
from src.multiresolution import (Window, downsample, upsample,
                                 compose_multiresolution, window_trajectory)


def test_downsample_upsample_shapes():
    field = np.arange(64.0).reshape(8, 8)
    c = downsample(field, 2)
    assert c.shape == (4, 4)
    up = upsample(c, 2)
    assert up.shape == (8, 8)


def test_compose_window_is_exact_fine_inside():
    rng = np.random.default_rng(0)
    field = rng.random((16, 16))
    w = Window(i0=4, j0=4, size=6)
    comp = compose_multiresolution(field, w, coarse_factor=4, blend_width=0)
    assert comp.shape == field.shape
    # à l'intérieur de la fenêtre, le composé == champ fin exact
    assert np.allclose(comp[4:10, 4:10], field[4:10, 4:10])


def test_blend_reduces_seam_jump_on_smooth_field():
    from src.metrics import seam_jump
    yy, xx = np.mgrid[0:32, 0:32].astype(float)
    field = np.sin(xx / 5.0) + np.cos(yy / 7.0)  # champ lisse
    w = Window(i0=10, j0=10, size=10)
    hard = compose_multiresolution(field, w, coarse_factor=4, blend_width=0)
    soft = compose_multiresolution(field, w, coarse_factor=4, blend_width=3)
    j_hard = seam_jump(hard, w.i0, w.j0, w.size)
    j_soft = seam_jump(soft, w.i0, w.j0, w.size)
    assert j_soft <= j_hard + 1e-9


def test_window_trajectory_in_bounds():
    grid = GridConfig(H=64, W=64)
    wins = window_trajectory(grid, size=16, n_frames=10, axis="x", margin=4)
    assert len(wins) == 10
    for w in wins:
        assert 0 <= w.j0 and w.j0 + w.size <= grid.W
        assert 0 <= w.i0 and w.i0 + w.size <= grid.H
