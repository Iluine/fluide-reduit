import numpy as np
import pytest
from config import GridConfig, GRAVITY
from src.solver_wetdry import simulate_wetdry, simulate_wetdry_o2

GRID = GridConfig(H=48, W=48, dx=4.0 / 48, dy=4.0 / 48)


def test_o2_c_property_lake_at_rest_emergent():
    # WB au 2e ordre : reconstruire eta (pas h) DOIT garder le lac au repos -> vitesse ~0.
    # (Au repos eta=const -> pente minmod nulle -> se reduit au 1er ordre -> C-property heritee.)
    yy, xx = np.mgrid[0:48, 0:48].astype(float)
    b = 0.6 * np.exp(-(((xx - 24) ** 2 + (yy - 24) ** 2) / 30.0))
    eta0 = 0.4
    h0 = np.maximum(eta0 - b, 0.0)
    z = np.zeros_like(h0)
    _, hs, hus, hvs = simulate_wetdry_o2(h0, z.copy(), z.copy(), b, GRID, t_end=2.0)
    speed = np.sqrt((hus / np.maximum(hs, 1e-6)) ** 2 + (hvs / np.maximum(hs, 1e-6)) ** 2)
    assert float(speed.max()) < 1e-6                       # WB 2e ordre tenu (herite)
    assert float(hs.min()) >= 0.0                          # positivite


def test_o2_positivity_on_drying():
    yy, xx = np.mgrid[0:48, 0:48].astype(float)
    h0 = np.maximum(0.5 * np.exp(-(((xx - 24) ** 2 + (yy - 24) ** 2) / 20.0)) - 0.05, 0.0)
    z = np.zeros_like(h0)
    _, hs, _, _ = simulate_wetdry_o2(h0, z.copy(), z.copy(), np.zeros_like(h0), GRID, t_end=1.0)
    assert float(hs.min()) >= 0.0 and np.isfinite(hs).all()  # reconciliation eta/positivite OK


def test_o2_sharper_than_o1_on_dam_break():
    # le 2e ordre resout un front plus NET que le 1er ordre (zone de transition <= )
    xs = (np.arange(48) + 0.5) * GRID.dx
    h0 = np.where(xs[None, :] <= 2.0, 1.0, 0.1) * np.ones((48, 48))
    z = np.zeros((48, 48))

    def front_width(sim):
        _, hs, _, _ = sim(h0.copy(), z.copy(), z.copy(), z.copy(), GRID, t_end=0.3)
        row = hs[-1, 24, :]                                 # profil final
        return int(np.sum((row > 0.15) & (row < 0.9)))      # largeur de transition du choc

    assert front_width(simulate_wetdry_o2) <= front_width(simulate_wetdry)  # plus net (ou egal)


def test_mc_limiter_preserves_invariants_and_is_sharper():
    # Le limiteur MC doit AUSSI tenir C-property + positivite, et etre >= net que minmod.
    yy, xx = np.mgrid[0:48, 0:48].astype(float)
    b = 0.6 * np.exp(-(((xx - 24) ** 2 + (yy - 24) ** 2) / 30.0))
    h0 = np.maximum(0.4 - b, 0.0)
    z = np.zeros_like(h0)
    _, hs, hus, hvs = simulate_wetdry_o2(h0, z.copy(), z.copy(), b, GRID, t_end=2.0, limiter="mc")
    speed = np.sqrt((hus / np.maximum(hs, 1e-6)) ** 2 + (hvs / np.maximum(hs, 1e-6)) ** 2)
    assert float(speed.max()) < 1e-6 and float(hs.min()) >= 0.0   # WB + positivite (MC)

    xs = (np.arange(48) + 0.5) * GRID.dx
    h0d = np.where(xs[None, :] <= 2.0, 1.0, 0.1) * np.ones((48, 48))
    zd = np.zeros((48, 48))

    def width(lim):
        _, h, _, _ = simulate_wetdry_o2(h0d.copy(), zd.copy(), zd.copy(), zd.copy(),
                                        GRID, t_end=0.3, limiter=lim)
        r = h[-1, 24, :]
        return int(np.sum((r > 0.15) & (r < 0.9)))

    assert width("mc") <= width("minmod")           # MC au moins aussi net que minmod
    assert float(simulate_wetdry_o2(h0d.copy(), zd.copy(), zd.copy(), zd.copy(),
                                    GRID, t_end=0.3, limiter="mc")[1].min()) >= 0.0  # positivite bore
