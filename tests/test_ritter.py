import numpy as np
import pytest
from config import GRAVITY
from src.ritter import ritter_dam_break_dry


def test_ritter_front_position_and_profile():
    hl, x0, t = 0.005, 5.0, 6.0
    x = np.linspace(0, 10, 1001)
    h = ritter_dam_break_dry(x, t, hl, x0)
    c = np.sqrt(GRAVITY * hl)
    xA, xB = x0 - t * c, x0 + 2 * t * c           # bords amont/aval de la raréfaction
    np.testing.assert_allclose(h[x <= xA], hl)          # plateau amont intact
    assert (h[x >= xB] == 0.0).all()                     # sec en aval du front
    assert (h >= 0).all()
    # à la position du barrage, h = (4/9) hl (valeur de Ritter)
    i0 = np.argmin(np.abs(x - x0))
    assert h[i0] == pytest.approx(4.0 / 9.0 * hl, rel=1e-2)
