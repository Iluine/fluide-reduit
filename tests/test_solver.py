import numpy as np

from config import GridConfig, GRAVITY
from src.solver import (make_terrain, initial_condition_dam_break,
                        initial_condition_gaussian_drop, cfl_dt,
                        lax_friedrichs_step)

GRID = GridConfig(H=32, W=32, dx=1.0, dy=1.0)


def _run(h, u, v, b, n, cfl=0.45, min_depth=1e-3):
    dt = cfl_dt(h, u, v, GRID, cfl)
    for _ in range(n):
        h, u, v = lax_friedrichs_step(h, u, v, b, dt, GRID, min_depth)
    return h, u, v


def test_cfl_dt_positive_and_scales_with_cfl():
    h, u, v = initial_condition_gaussian_drop(GRID)
    b = make_terrain(GRID, "flat")
    dt1 = cfl_dt(h, u, v, GRID, 0.45)
    dt2 = cfl_dt(h, u, v, GRID, 0.9)
    assert dt1 > 0
    assert abs(dt2 / dt1 - 2.0) < 1e-9


def test_mass_conserved_flat_terrain():
    h, u, v = initial_condition_gaussian_drop(GRID)
    b = make_terrain(GRID, "flat")
    m0 = h.sum()
    h2, _, _ = _run(h, u, v, b, n=50)
    assert abs(h2.sum() - m0) / m0 < 1e-9  # paroi réfléchissante -> masse conservée


def test_lake_at_rest_flat_terrain_stays_still():
    h = np.full((GRID.H, GRID.W), 1.5)
    u = np.zeros((GRID.H, GRID.W))
    v = np.zeros((GRID.H, GRID.W))
    b = make_terrain(GRID, "flat")
    h2, u2, v2 = _run(h, u, v, b, n=20)
    assert np.allclose(h2, 1.5, atol=1e-9)
    assert np.allclose(u2, 0.0, atol=1e-9)
    assert np.allclose(v2, 0.0, atol=1e-9)


def test_centered_drop_stays_symmetric():
    h, u, v = initial_condition_gaussian_drop(GRID, cx_frac=0.5, cy_frac=0.5)
    b = make_terrain(GRID, "flat")
    h2, _, _ = _run(h, u, v, b, n=15)
    # symétrie gauche/droite (axe x) sur un domaine et une CI symétriques
    assert np.allclose(h2, h2[:, ::-1], atol=1e-8)
    assert np.allclose(h2, h2[::-1, :], atol=1e-8)


def test_step_preserves_shape_and_dtype():
    h, u, v = initial_condition_dam_break(GRID)
    b = make_terrain(GRID, "bump")
    dt = cfl_dt(h, u, v, GRID, 0.45)
    h2, u2, v2 = lax_friedrichs_step(h, u, v, b, dt, GRID, 1e-3)
    assert h2.shape == (GRID.H, GRID.W)
    assert h2.dtype == np.float64
