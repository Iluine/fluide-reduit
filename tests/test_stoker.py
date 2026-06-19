import numpy as np
import pytest
from config import GRAVITY
from src.stoker import stoker_star_state, stoker_dam_break


def test_star_state_satisfies_rankine_hugoniot_and_riemann():
    # Le coeur de la validation flag #2 : un coefficient faux VIOLE ces residus.
    g = GRAVITY
    hl, hr = 1.0, 0.1
    h_m, u_m, s = stoker_star_state(hl, hr)
    assert hr < h_m < hl                                   # etat intermediaire encadre
    # Invariant de Riemann sur la rarefaction (gauche) :
    assert u_m + 2 * np.sqrt(g * h_m) == pytest.approx(2 * np.sqrt(g * hl), rel=1e-8)
    # Rankine-Hugoniot a travers le choc (etat (h_m,u_m) vs (hr,0)), residus ~0 :
    #   masse : s(h_m - hr) = h_m u_m - hr*0
    #   qdm   : s(h_m u_m - 0) = (h_m u_m^2 + g h_m^2/2) - (g hr^2/2)
    assert s * (h_m - hr) == pytest.approx(h_m * u_m, rel=1e-8)
    assert s * (h_m * u_m) == pytest.approx(
        h_m * u_m ** 2 + 0.5 * g * h_m ** 2 - 0.5 * g * hr ** 2, rel=1e-8)


def test_dam_break_zones_and_positivity():
    hl, hr, x0, t = 1.0, 0.1, 2.0, 0.2
    x = np.linspace(0, 4, 2001)
    h, u = stoker_dam_break(x, t, hl, hr, x0)
    assert (h > 0).all()                                   # lit mouille partout (h_r>0)
    assert h[x < x0 - t * np.sqrt(GRAVITY * hl)] == pytest.approx(hl)  # plateau gauche intact
    h_m, u_m, s = stoker_star_state(hl, hr)
    # loin devant le choc (x > x0 + s t) : etat non perturbe h_r
    assert h[x > x0 + s * t + 0.1] == pytest.approx(hr)
    # juste derriere le choc : l'etat intermediaire h_m (un plateau)
    mid = (x > x0 + 0.05) & (x < x0 + s * t - 0.05)
    assert np.median(h[mid]) == pytest.approx(h_m, rel=5e-2)


def test_dam_break_mass_monotone_front():
    # le choc (front net) se deplace : a deux instants, sa position avance de s*dt
    hl, hr, x0 = 1.0, 0.1, 2.0
    h_m, u_m, s = stoker_star_state(hl, hr)
    xs = np.linspace(0, 4, 4001)
    h1, _ = stoker_dam_break(xs, 0.10, hl, hr, x0)
    h2, _ = stoker_dam_break(xs, 0.20, hl, hr, x0)
    front1 = xs[h1 > 0.5 * (h_m + hr)].max()
    front2 = xs[h2 > 0.5 * (h_m + hr)].max()
    assert front2 - front1 == pytest.approx(s * 0.10, abs=2 * (4 / 4001))  # avance = s*dt
