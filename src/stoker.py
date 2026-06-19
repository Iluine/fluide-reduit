"""Solution exacte de Stoker (1957) : rupture de barrage sur lit MOUILLÉ (bore).
Structure : raréfaction gauche + état intermédiaire (h_m,u_m) + CHOC droit dans (h_r,0).
h_m n'est PAS en forme close -> bisection sur la transcendante de Rankine-Hugoniot.
Vérifié par les résidus R-H (cf. tests). Oracle du FRONT NET (vraie discontinuité)."""
from __future__ import annotations

import numpy as np

from config import GRAVITY


def stoker_star_state(hl: float, hr: float) -> tuple[float, float, float]:
    """(h_m, u_m, s) par bisection : raréfaction (u_m=2(√(ghl)−√(ghm))) = choc R-H."""
    g = GRAVITY

    def f(hm):  # raréfaction − choc, racine en h_m ∈ (hr, hl)
        u_rar = 2.0 * (np.sqrt(g * hl) - np.sqrt(g * hm))
        u_shk = (hm - hr) * np.sqrt(0.5 * g * (hm + hr) / (hm * hr))
        return u_rar - u_shk

    lo, hi = hr * (1.0 + 1e-9), hl * (1.0 - 1e-9)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    h_m = 0.5 * (lo + hi)
    u_m = 2.0 * (np.sqrt(g * hl) - np.sqrt(g * h_m))
    s = h_m * u_m / (h_m - hr)
    return float(h_m), float(u_m), float(s)


def stoker_dam_break(x, t: float, hl: float = 1.0, hr: float = 0.1,
                     x0: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """h(x,t), u(x,t) pour t>0 : 5 zones (h_l | raréfaction | h_m | choc | h_r)."""
    x = np.asarray(x, dtype=np.float64)
    g = GRAVITY
    if t <= 0:
        return np.where(x <= x0, hl, hr), np.zeros_like(x)
    h_m, u_m, s = stoker_star_state(hl, hr)
    cl = np.sqrt(g * hl)
    xi = (x - x0) / t                                  # variable d'auto-similarité
    x_raref_head = -cl                                 # tête de raréfaction (vers la gauche)
    x_raref_tail = u_m - np.sqrt(g * h_m)              # queue de raréfaction
    h = np.empty_like(x)
    u = np.empty_like(x)
    # zone 1 : plateau gauche
    m1 = xi <= x_raref_head
    h[m1] = hl
    u[m1] = 0.0
    # zone 2 : raréfaction (éventail) — h=(2cl−xi)^2/(9g), u=2(cl+xi)/3
    m2 = (xi > x_raref_head) & (xi < x_raref_tail)
    h[m2] = (2.0 * cl - xi[m2]) ** 2 / (9.0 * g)
    u[m2] = 2.0 * (cl + xi[m2]) / 3.0
    # zone 3 : état intermédiaire
    m3 = (xi >= x_raref_tail) & (xi < s)
    h[m3] = h_m
    u[m3] = u_m
    # zone 4 : aval du choc (non perturbé)
    m4 = xi >= s
    h[m4] = hr
    u[m4] = 0.0
    return h, u
