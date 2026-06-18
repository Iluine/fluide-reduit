"""M0 — Oracle shallow-water 2D (Saint-Venant) sur terrain.

Schéma : Lax-Friedrichs en forme conservative (flux d'interface global LF),
exactement conservatif en masse. Paroi réfléchissante (cellules fantômes :
hauteur en Neumann, composante normale de la quantité de mouvement négée).
Variables conservées internes : q1=h, q2=h*u, q3=h*v.

Indexation array[y, x] : axe 0 = y (H lignes), axe 1 = x (W colonnes).
u = vitesse selon x, v = vitesse selon y. b = bathymétrie (terrain) fixe."""
from __future__ import annotations

import numpy as np

from config import GRAVITY, GridConfig


def make_terrain(grid: GridConfig, kind: str = "bump") -> np.ndarray:
    """Bathymétrie fixe (H,W). 'flat'=plat ; 'bump'=bosse gaussienne centrale."""
    H, W = grid.H, grid.W
    if kind == "flat":
        return np.zeros((H, W), dtype=np.float64)
    if kind == "bump":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        sigma = min(H, W) / 6.0
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        return 0.4 * np.exp(-r2 / (2.0 * sigma ** 2))
    raise ValueError(f"terrain inconnu : {kind!r}")


def initial_condition_dam_break(grid: GridConfig, depth_left: float = 2.0,
                                depth_right: float = 1.0,
                                split_frac: float = 0.5):
    """Rupture de barrage : marche de hauteur en x. u=v=0."""
    H, W = grid.H, grid.W
    h = np.full((H, W), depth_right, dtype=np.float64)
    split = int(split_frac * W)
    h[:, :split] = depth_left
    return h, np.zeros((H, W)), np.zeros((H, W))


def initial_condition_gaussian_drop(grid: GridConfig, base: float = 1.0,
                                    amp: float = 0.5, cx_frac: float = 0.5,
                                    cy_frac: float = 0.5, width_frac: float = 0.1):
    """Goutte/bosse gaussienne sur la hauteur. u=v=0."""
    H, W = grid.H, grid.W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = cx_frac * (W - 1), cy_frac * (H - 1)
    sigma = width_frac * min(H, W)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    h = base + amp * np.exp(-r2 / (2.0 * sigma ** 2))
    return h, np.zeros((H, W)), np.zeros((H, W))


def cfl_dt(h: np.ndarray, u: np.ndarray, v: np.ndarray,
           grid: GridConfig, cfl: float) -> float:
    """Pas de temps limité par la CFL : cfl * min(dx,dy) / max(|u|+c, |v|+c)."""
    c = np.sqrt(GRAVITY * np.maximum(h, 0.0))
    max_speed = max(float((np.abs(u) + c).max()), float((np.abs(v) + c).max()), 1e-12)
    return cfl * min(grid.dx, grid.dy) / max_speed


def _pad_reflective(q1: np.ndarray, q2: np.ndarray, q3: np.ndarray):
    """Cellules fantômes pour paroi réfléchissante.

    q1=h : Neumann (edge). q2=h*u : composante normale négée aux bords x
    (gauche/droite). q3=h*v : composante normale négée aux bords y (haut/bas).
    Retourne des tableaux (H+2, W+2)."""
    p1 = np.pad(q1, 1, mode="edge")
    p2 = np.pad(q2, 1, mode="edge")
    p3 = np.pad(q3, 1, mode="edge")
    p2[:, 0] = -p2[:, 1]      # bord gauche (x) : négation de h*u
    p2[:, -1] = -p2[:, -2]    # bord droit (x)
    p3[0, :] = -p3[1, :]      # bord haut (y) : négation de h*v
    p3[-1, :] = -p3[-2, :]    # bord bas (y)
    return p1, p2, p3


def lax_friedrichs_step(h: np.ndarray, u: np.ndarray, v: np.ndarray,
                        b: np.ndarray, dt: float, grid: GridConfig,
                        min_depth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Un pas de temps Lax-Friedrichs conservatif (flux d'interface global LF).

    Conservatif en masse : le flux numérique de h aux parois est nul."""
    g = GRAVITY
    dx, dy = grid.dx, grid.dy

    q1, q2, q3 = h, h * u, h * v
    p1, p2, p3 = _pad_reflective(q1, q2, q3)
    hp = np.maximum(p1, min_depth)            # plancher pour les divisions
    up, vp = p2 / hp, p3 / hp

    # Flux physiques en chaque cellule (fantômes inclus)
    F1, F2, F3 = p2, p2 * up + 0.5 * g * p1 ** 2, p2 * vp     # flux selon x
    G1, G2, G3 = p3, p3 * up, p3 * vp + 0.5 * g * p1 ** 2     # flux selon y

    ax, ay = dx / (2.0 * dt), dy / (2.0 * dt)  # coefficients diffusifs LF global

    def fx(F, U):  # flux numérique aux interfaces x -> (H+2, W+1)
        return 0.5 * (F[:, :-1] + F[:, 1:]) - ax * (U[:, 1:] - U[:, :-1])

    def fy(G, U):  # flux numérique aux interfaces y -> (H+1, W+2)
        return 0.5 * (G[:-1, :] + G[1:, :]) - ay * (U[1:, :] - U[:-1, :])

    Fx1, Fx2, Fx3 = fx(F1, p1), fx(F2, p2), fx(F3, p3)
    Gy1, Gy2, Gy3 = fy(G1, p1), fy(G2, p2), fy(G3, p3)

    def divx(Fx):  # divergence en x sur les cellules réelles -> (H, W)
        return (Fx[1:-1, 1:] - Fx[1:-1, :-1]) / dx

    def divy(Gy):  # divergence en y sur les cellules réelles -> (H, W)
        return (Gy[1:, 1:-1] - Gy[:-1, 1:-1]) / dy

    new1 = q1 - dt * (divx(Fx1) + divy(Gy1))
    new2 = q2 - dt * (divx(Fx2) + divy(Gy2))
    new3 = q3 - dt * (divx(Fx3) + divy(Gy3))

    # Terme source de bathymétrie : -g h db/dx (sur h*u), -g h db/dy (sur h*v)
    dbdx = np.zeros_like(b)
    dbdy = np.zeros_like(b)
    dbdx[:, 1:-1] = (b[:, 2:] - b[:, :-2]) / (2.0 * dx)
    dbdy[1:-1, :] = (b[2:, :] - b[:-2, :]) / (2.0 * dy)
    new2 = new2 - dt * g * q1 * dbdx
    new3 = new3 - dt * g * q1 * dbdy

    h_safe = np.maximum(new1, min_depth)
    return new1, new2 / h_safe, new3 / h_safe
