"""Vocabulaire de transport mouillé/sec : CI (η + u co-directionnel) + scénarios.

PESÉ run-up (seul régime exerçant l'assèchement h>0->0, le cas opérateur le plus dur)
et positions étalées (axe V5 = position autant qu'orientation). Généré par le solveur
2e ordre validé. Sert deux verdicts : étendue (k combiné) + opérateur single-global.

Corrections de CI (mêmes faux verdicts que l'écrêtage DMD, mais côté données) :
  1. vitesse compagnon planchée à la profondeur AU CENTRE de la bosse (eau franche) ->
     pas de pic supersonique √(g/h)→∞ au trait d'eau ;
  2. bosses semées en zone MOUILLÉE (cf. vocabulary) ;
  3. îles ÉMERGENTES via still_surface>0 (base submergée, sommet émergent)."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from config import GridConfig, GRAVITY
from src.terrains import gaussian_terrain

_DIRS = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0),
         "diag": (np.sqrt(0.5), np.sqrt(0.5))}


def slope_bed(grid: GridConfig, slope: float, axis: str = "x") -> np.ndarray:
    """Plage inclinée centrée : b = slope·(coord_frac − 0.5) (traverse 0 -> mouillé/sec)."""
    H, W = grid.H, grid.W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    coord = (xx / (W - 1)) if axis == "x" else (yy / (H - 1))
    return slope * (coord - 0.5)


def island_bed(grid: GridConfig, amp: float, x0_frac: float, y0_frac: float,
               sigma: float) -> np.ndarray:
    """Île gaussienne (réutilise gaussian_terrain) — émergente si still_surface < amp."""
    return gaussian_terrain(grid, amp, x0_frac, y0_frac, sigma, slope=0.0)


def moving_wave_ic(grid: GridConfig, b: np.ndarray, still_surface: float, amp: float,
                   x0_frac: float, y0_frac: float, width_frac: float, direction: str
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bosse de surface MOBILE : η = amp·sech²(coord_dir/width), u co-directionnel.

    Sans u, une bosse à u=0 se scinde (d'Alembert) -> pas de run-up propre. La vitesse
    compagnon utilise une profondeur de référence PLANCHÉE à la profondeur au centre de
    la bosse (eau franche), scalaire -> bornée partout (pas de divergence au trait d'eau)."""
    H, W = grid.H, grid.W
    g = GRAVITY
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = x0_frac * (W - 1), y0_frac * (H - 1)
    nx, ny = _DIRS[direction]
    coord = (xx - cx) * nx + (yy - cy) * ny             # le long de la direction (cellules)
    width = width_frac * max(H, W)
    eta_bump = amp / np.cosh(coord / width) ** 2
    h_still = np.maximum(still_surface - b, 0.0)
    wet = h_still > 1e-6
    h = h_still + np.where(wet, eta_bump, 0.0)
    # profondeur de référence = profondeur AU CENTRE de la bosse (eau franche), planchée
    d_ref = max(float(still_surface - b[int(round(cy)), int(round(cx))]), 0.05)
    u_mag = np.sqrt(g / d_ref) * eta_bump               # scalaire d_ref -> vitesse bornée
    hu = np.where(wet, h * u_mag * nx, 0.0)
    hv = np.where(wet, h * u_mag * ny, 0.0)
    return h, hu, hv


def dam_break_ic(grid: GridConfig, b: np.ndarray, hl: float, hr: float, x0_frac: float,
                 orientation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dam-break (sec hr=0 / mouillé hr>0) orienté. Surfaces hl/hr au-dessus du lit b."""
    H, W = grid.H, grid.W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    if orientation == "x":
        coord = xx / (W - 1)
    elif orientation == "y":
        coord = yy / (H - 1)
    else:  # diag
        coord = 0.5 * (xx / (W - 1) + yy / (H - 1))
    surf = np.where(coord <= x0_frac, hl, hr)
    h = np.maximum(surf - b, 0.0)
    z = np.zeros((H, W))
    return h, z.copy(), z.copy()


@dataclass
class Scenario:
    name: str
    b: np.ndarray
    h0: np.ndarray
    hu0: np.ndarray
    hv0: np.ndarray
    t_end: float


def vocabulary(grid: GridConfig) -> list[Scenario]:
    """~16 scénarios pesés run-up (>=8), positions étalées, amplitude large (verdict
    d'étendue conservateur). Toutes les CI ont une zone mouillée non triviale et une
    perturbation réelle (cf. tests anti-dégénérescence)."""
    flat = np.zeros((grid.H, grid.W))
    # pentes : still_surface>0 -> zone mouillée profonde, trait d'eau intérieur
    sl_x = slope_bed(grid, slope=0.6, axis="x")        # monte vers +x ; mouillé côté x bas
    sl_y = slope_bed(grid, slope=0.6, axis="y")
    isl = island_bed(grid, amp=0.7, x0_frac=0.5, y0_frac=0.5, sigma=8.0)
    isl_off = island_bed(grid, amp=0.7, x0_frac=0.4, y0_frac=0.6, sigma=7.0)
    SS_SLOPE, SS_ISLAND = 0.15, 0.45                   # surfaces de repos (eau franche)
    sc: list[Scenario] = []

    # --- RUN-UP (>=8) : bosses en EAU FRANCHE (positions basses), montant vers le trait d'eau ---
    runup_specs = [
        ("runup_slopex_x20_dxp", sl_x, SS_SLOPE, 0.20, 0.5, "+x"),
        ("runup_slopex_x30_dxp", sl_x, SS_SLOPE, 0.30, 0.5, "+x"),
        ("runup_slopex_x35_dxp", sl_x, SS_SLOPE, 0.35, 0.5, "+x"),
        ("runup_slopey_x25_dyp", sl_y, SS_SLOPE, 0.25, 0.25, "+y"),
        ("runup_slopey_x35_dyp", sl_y, SS_SLOPE, 0.35, 0.35, "+y"),
        ("runup_island_x20_dxp", isl, SS_ISLAND, 0.20, 0.5, "+x"),
        ("runup_island_x25_diag", isl, SS_ISLAND, 0.25, 0.30, "diag"),
        ("runup_islandoff_x70_dxm", isl_off, SS_ISLAND, 0.70, 0.5, "-x"),
    ]
    for name, bed, ss, x0f, y0f, d in runup_specs:
        h, hu, hv = moving_wave_ic(grid, bed, still_surface=ss, amp=0.10,
                                   x0_frac=x0f, y0_frac=y0f, width_frac=0.08, direction=d)
        sc.append(Scenario(name, bed, h, hu, hv, t_end=1.2))

    # --- DAM-BREAK SEC (mouillage) : positions/orientations variées ---
    for name, bed, x0f, ori in [("dambreakdry_x30_x", flat, 0.30, "x"),
                                ("dambreakdry_x50_diag", flat, 0.50, "diag"),
                                ("dambreakdry_x60_y", flat, 0.60, "y")]:
        h, hu, hv = dam_break_ic(grid, bed, hl=0.5, hr=0.0, x0_frac=x0f, orientation=ori)
        sc.append(Scenario(name, bed, h, hu, hv, t_end=0.9))

    # --- BORE MOUILLÉ (Stoker) : positions/orientations variées ---
    for name, bed, x0f, ori in [("bore_x40_x", flat, 0.40, "x"),
                                ("bore_x50_diag", flat, 0.50, "diag"),
                                ("bore_x30_y", flat, 0.30, "y")]:
        h, hu, hv = dam_break_ic(grid, bed, hl=1.0, hr=0.1, x0_frac=x0f, orientation=ori)
        sc.append(Scenario(name, bed, h, hu, hv, t_end=0.7))

    return sc
