"""Solution analytique de Ritter : rupture de barrage sur lit SEC, sans friction.
Vérifiée dans SWASHES §4.1.2 (arXiv:1110.0288). Référence de validation du solveur
mouillé/sec (position et forme du front). h_l à gauche de x0, sec (0) à droite, u=0 à t=0."""
from __future__ import annotations

import numpy as np

from config import GRAVITY


def ritter_dam_break_dry(x: np.ndarray, t: float, hl: float = 0.005,
                         x0: float = 5.0) -> np.ndarray:
    """h(x,t) pour t>0. Trois zones : plateau (h_l), raréfaction (parabole), sec (0)."""
    x = np.asarray(x, dtype=np.float64)
    c = np.sqrt(GRAVITY * hl)
    if t <= 0:
        return np.where(x <= x0, hl, 0.0)
    xA = x0 - t * c
    xB = x0 + 2.0 * t * c
    h = np.empty_like(x)
    h[:] = 0.0
    h[x <= xA] = hl
    mid = (x > xA) & (x < xB)
    h[mid] = (4.0 / (9.0 * GRAVITY)) * (np.sqrt(GRAVITY * hl) - (x[mid] - x0) / (2.0 * t)) ** 2
    return h
