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


from src.terrains import DROP_ICS, SplitEntry, sample_split, SAMPLE_SEED


def test_sample_split_counts_and_topologies():
    entries = sample_split(GRID)
    roles = [e.role for e in entries]
    assert roles.count("train") == 9
    assert roles.count("holdout_interp") == 1
    assert roles.count("holdout_extrap") == 2
    kinds = {e.params.kind for e in entries if e.role == "train"}
    assert {"bump", "obstacle"} <= kinds  # les deux topologies présentes en train


def test_sample_split_train_in_range_and_submerged():
    for e in sample_split(GRID):
        b = make_terrain_from_params(GRID, e.params)
        assert REST_SURFACE - float(b.max()) >= MIN_REST_DEPTH  # submergé partout
        if e.role == "train" and e.params.kind == "bump":
            assert 0.2 <= e.params.amp <= 0.5
            assert 8.0 <= e.params.sigma <= 13.0
        if e.role == "train" and e.params.kind == "obstacle":
            assert 0.6 <= e.params.amp <= 1.0
            assert 4.0 <= e.params.sigma <= 7.0


def test_sample_split_extrapolation_is_geometric():
    entries = {e.regime: e for e in sample_split(GRID)}
    assert "extrap_obstacle" in entries and "extrap_channel" in entries
    obst = entries["extrap_obstacle"].params
    # extrapolation par la GÉOMÉTRIE : sigma sous la plage train [4,7], amp submergée
    assert obst.sigma < 4.0
    assert obst.amp <= 1.0
    # topologie nouvelle
    assert entries["extrap_channel"].params.kind == "channel"


def test_sample_split_is_deterministic():
    a = sample_split(GRID, seed=SAMPLE_SEED)
    b = sample_split(GRID, seed=SAMPLE_SEED)
    assert [e.params.to_vector().tolist() for e in a] == \
           [e.params.to_vector().tolist() for e in b]


def test_holdout_uses_new_ic():
    for e in sample_split(GRID):
        if e.role.startswith("holdout"):
            assert e.ic_ids == ("drop_new",)
    assert "drop_new" in DROP_ICS and "drop_center" in DROP_ICS


from config import SolverConfig
from src.terrains import rest_residual

_SOLVER = SolverConfig(cfl=0.45, n_steps=800, save_every=4, min_depth=1e-3)


def test_rest_residual_flat_is_near_zero():
    # bathymétrie plate + surface constante = lac au repos exact : aucun courant parasite
    b = np.zeros((64, 64), dtype=np.float64)
    surf_dev, speed = rest_residual(GRID, b, _SOLVER, n_steps=30)
    assert surf_dev < 1e-9
    assert speed < 1e-9


def test_rest_residual_detects_bathymetry_but_stays_bounded():
    b = gaussian_terrain(GRID, amp=0.4, x0_frac=0.5, y0_frac=0.5, sigma=10.0, slope=0.0)
    surf_dev, speed = rest_residual(GRID, b, _SOLVER, n_steps=30)
    # le schéma n'est pas well-balanced -> résidu non nul, mais doit rester petit
    # (terrain doux) : ni explosion, ni NaN
    assert np.isfinite(surf_dev) and np.isfinite(speed)
    assert surf_dev > 0.0          # détecte bien le gradient de bathymétrie
    assert surf_dev < 0.2          # reste borné (oracle sain pour un terrain doux)
