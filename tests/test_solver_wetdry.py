import numpy as np
import pytest
from config import GridConfig, GRAVITY
from src.solver_wetdry import simulate_wetdry, desingularize_velocity

GRID = GridConfig(H=32, W=32, dx=4.0 / 32, dy=4.0 / 32)


def test_desingularize_zero_below_dry():
    h = np.array([0.0, 1e-6, 1.0]); hu = np.array([0.1, 0.1, 2.0])
    u = desingularize_velocity(h, hu, dry_eps=1e-4)
    assert u[0] == 0.0 and u[1] == 0.0            # sec -> vitesse nulle (pas d'explosion)
    assert u[2] == pytest.approx(2.0)


def test_c_property_lake_at_rest_emergent_bathy():
    """C-PROPERTY : lac au repos (η=const) sur bathymétrie ÉMERGENTE (île qui perce la
    surface) reste au repos -> vitesse parasite ~ 0. C'est le test du well-balancing ET
    le garde-fou qui attrape une erreur dans le terme source d'Audusse."""
    yy, xx = np.mgrid[0:32, 0:32].astype(float)
    b = 0.6 * np.exp(-(((xx - 16) ** 2 + (yy - 16) ** 2) / 18.0))  # île, sommet 0.6
    eta0 = 0.4                                     # surface < sommet -> île émergée
    h0 = np.maximum(eta0 - b, 0.0)                 # sec sur l'île, mouillé autour
    z = np.zeros_like(h0)
    times, hs, hus, hvs = simulate_wetdry(h0, z.copy(), z.copy(), b, GRID,
                                          cfl=0.4, t_end=2.0, dry_eps=1e-4)
    speed = np.sqrt((hus / np.maximum(hs, 1e-6)) ** 2
                    + (hvs / np.maximum(hs, 1e-6)) ** 2)
    assert float(speed.max()) < 1e-6              # repos préservé (well-balanced)
    assert float(hs.min()) >= 0.0                 # positivité


def test_positivity_on_drying():
    """Une goutte qui s'étale sur un fond sec ne doit JAMAIS produire h<0."""
    yy, xx = np.mgrid[0:32, 0:32].astype(float)
    b = np.zeros((32, 32))
    h0 = np.maximum(0.5 * np.exp(-(((xx - 16) ** 2 + (yy - 16) ** 2) / 4.0)) - 0.05, 0.0)
    z = np.zeros_like(h0)
    times, hs, hus, hvs = simulate_wetdry(h0, z.copy(), z.copy(), b, GRID,
                                          cfl=0.4, t_end=1.0, dry_eps=1e-4)
    assert float(hs.min()) >= 0.0                 # positivité tout le temps
    assert np.isfinite(hs).all()                  # pas de NaN/Inf (désingularisation OK)
