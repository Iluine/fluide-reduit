"""Solutions analytiques exactes de Thacker (shallow-water, trait d'eau MOBILE).

Vérifiées dans SWASHES (Delestre et al. 2016, arXiv:1110.0288, §4.2.1/§4.2.2).
Servent (a) d'oracle de représentation pour W0 (POD d'un front mobile, test n-width),
(b) de référence de validation pour le solveur mouillé/sec W1.

Indexation array[y, x]. h = hauteur d'eau (>=0). b = bathymétrie (z). Surface η = h + b."""
from __future__ import annotations

import numpy as np

from config import GRAVITY, GridConfig


def _centers(grid: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    """Coordonnées des centres de cellules (x, y), chacune (H, W). Domaine [0, L]²."""
    xs = (np.arange(grid.W) + 0.5) * grid.dx
    ys = (np.arange(grid.H) + 0.5) * grid.dy
    xx, yy = np.meshgrid(xs, ys)  # (H, W), indexation [y, x]
    return xx, yy


def thacker_radial_period(a: float = 1.0, h0: float = 0.1) -> float:
    """Période T = 2π/ω, ω = √(8 g h0)/a (paraboloïde radial, sans friction)."""
    omega = np.sqrt(8.0 * GRAVITY * h0) / a
    return 2.0 * np.pi / omega


def thacker_radial(grid: GridConfig, t: float, a: float = 1.0, h0: float = 0.1,
                   r0: float = 0.8, L: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Paraboloïde radial oscillant (SWASHES §4.2.2, sans friction). Shoreline = cercle
    de rayon variable (le front RESPIRE sur place). Retourne (h, b), h clampé >= 0."""
    xx, yy = _centers(grid)
    r2 = (xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2
    b = -h0 * (1.0 - r2 / a ** 2)                       # z(r), paraboloïde (bol)
    omega = np.sqrt(8.0 * GRAVITY * h0) / a
    A = (a ** 2 - r0 ** 2) / (a ** 2 + r0 ** 2)
    denom = 1.0 - A * np.cos(omega * t)
    eta = h0 * (np.sqrt(1.0 - A ** 2) / denom - 1.0
                - (r2 / a ** 2) * ((1.0 - A ** 2) / denom ** 2 - 1.0))
    h = np.maximum(eta - b, 0.0)
    return h, b
