import numpy as np
import pytest

from config import GridConfig
from src.solver import make_terrain
from src.terrains import (REST_SURFACE, MIN_REST_DEPTH, TerrainParams,
                          gaussian_terrain, channel_terrain,
                          make_terrain_from_params, rest_state_ic)

GRID = GridConfig(H=64, W=64, dx=1.0, dy=1.0)


def test_gaussian_terrain_matches_poc_bump():
    # Le bump du POC (make_terrain "bump") est un point de la famille : amp 0.4,
    # centre (0.5,0.5), sigma=min(H,W)/6, pas de pente. Pont de non-régression.
    b = gaussian_terrain(GRID, amp=0.4, x0_frac=0.5, y0_frac=0.5,
                         sigma=min(GRID.H, GRID.W) / 6.0, slope=0.0)
    assert b.shape == (64, 64)
    assert np.allclose(b, make_terrain(GRID, "bump"))


def test_gaussian_terrain_slope_adds_tilt():
    b = gaussian_terrain(GRID, amp=0.0, x0_frac=0.5, y0_frac=0.5, sigma=10.0, slope=0.02)
    # pente pure : croît de 0 (x=0) à 0.02 (x=W-1), constante en y
    assert b[0, 0] == pytest.approx(0.0)
    assert b[0, -1] == pytest.approx(0.02)
    assert np.allclose(b[0, :], b[-1, :])


def test_channel_terrain_is_smooth_and_bounded():
    b = channel_terrain(GRID, wall_height=1.0, y0_frac=0.5, half_width=8.0, wall_softness=2.0)
    assert b.shape == (64, 64)
    assert b.min() >= 0.0 and b.max() <= 1.0 + 1e-9
    # corridor central (autour de y0) peu élevé ; bords (parois) hauts
    assert b[32, 0] < 0.05          # centre du corridor
    assert b[0, 0] > 0.9            # paroi
    # lissage : gradient discret borné (pas de marche dure)
    grad = np.abs(np.diff(b, axis=0))
    assert grad.max() < 0.5


def test_make_terrain_from_params_dispatch():
    pb = TerrainParams("bump", amp=0.3, x0_frac=0.5, y0_frac=0.5, sigma=10.0, slope=0.0)
    pc = TerrainParams("channel", amp=1.0, x0_frac=0.5, y0_frac=0.5, sigma=8.0, slope=2.0)
    assert make_terrain_from_params(GRID, pb).shape == (64, 64)
    assert make_terrain_from_params(GRID, pc).shape == (64, 64)
    with pytest.raises(ValueError):
        make_terrain_from_params(GRID, TerrainParams("zzz", 0, 0.5, 0.5, 1, 0))


def test_terrain_params_vector_roundtrip():
    p = TerrainParams("obstacle", amp=0.8, x0_frac=0.55, y0_frac=0.45, sigma=5.0, slope=0.0)
    v = p.to_vector()
    assert v.shape == (6,)
    assert v[0] == 1.0  # obstacle
    assert v[1] == pytest.approx(0.8)
    assert p.to_dict()["kind"] == "obstacle"


def test_rest_state_ic_is_submerged_and_positive():
    b = gaussian_terrain(GRID, amp=1.0, x0_frac=0.5, y0_frac=0.5, sigma=5.0, slope=0.0)
    h, u, v = rest_state_ic(GRID, b, drop_amp=0.4, drop_x0_frac=0.6,
                            drop_y0_frac=0.4, drop_width_frac=0.1)
    assert h.shape == u.shape == v.shape == (64, 64)
    assert np.all(u == 0) and np.all(v == 0)
    # surface au repos = REST_SURFACE loin de la goutte ; h = REST_SURFACE - b là-bas
    # partout strictement positif avec marge
    assert h.min() > MIN_REST_DEPTH - 1e-9
    # la goutte ajoute de l'eau : pic > niveau de repos local
    assert h.max() > REST_SURFACE - b.min()


def test_rest_state_ic_rejects_unsubmerged_terrain():
    # un b qui perce la surface (b >= REST_SURFACE - marge) doit lever
    b = np.full((64, 64), REST_SURFACE, dtype=np.float64)
    with pytest.raises(AssertionError):
        rest_state_ic(GRID, b, drop_amp=0.4, drop_x0_frac=0.5,
                      drop_y0_frac=0.5, drop_width_frac=0.1)
