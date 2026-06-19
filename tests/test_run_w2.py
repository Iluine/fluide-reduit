"""Tests légers TDD pour run_w2_dambreak (W2 — n-width intra-échantillon, front dam-break).

Ces tests vérifient :
1. Le front dam-break se DÉPLACE (x mouillé max à la fin > au début).
2. dambreak_nwidth() retourne un front_l2 fini et strictement supérieur à global_l2
   (le front est plus dur que le bulk — même pattern discriminant que W0).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_dambreak_front_sweeps():
    """Le front wet/dry droit se déplace vers la droite au cours de la simulation."""
    from config import GridConfig
    from src.solver_wetdry import simulate_wetdry

    grid = GridConfig(H=32, W=32, dx=4.0 / 32, dy=4.0 / 32)
    xs = (np.arange(grid.W) + 0.5) * grid.dx
    x0 = 0.5
    hl = 0.1
    # Broadcasting: xs has shape (W,); broadcast over H rows
    h0 = np.where(xs[np.newaxis, :] <= x0, hl, 0.0) * np.ones((grid.H, grid.W))
    hu0 = np.zeros_like(h0)
    hv0 = np.zeros_like(h0)
    b = np.zeros((grid.H, grid.W))

    DRY_EPS = 1e-4

    # Position initiale du front : plus grand x avec h > DRY_EPS (rangée centrale)
    iy = grid.H // 2
    wet_start = np.where(h0[iy, :] > DRY_EPS)[0]
    x_front_start = float(xs[wet_start[-1]]) if len(wet_start) > 0 else 0.0

    # Simulation courte (t_end petit, assez pour que le front bouge)
    _, h_seq, _, _ = simulate_wetdry(h0, hu0, hv0, b, grid, cfl=0.4, t_end=0.3,
                                     dry_eps=DRY_EPS)

    wet_end = np.where(h_seq[-1][iy, :] > DRY_EPS)[0]
    x_front_end = float(xs[wet_end[-1]]) if len(wet_end) > 0 else 0.0

    assert x_front_end > x_front_start + grid.dx, (
        f"Le front n'a pas bougé : x_start={x_front_start:.4f}, x_end={x_front_end:.4f}"
    )


def test_dambreak_nwidth_returns_finite_front_harder_than_bulk():
    """dambreak_nwidth() : front_l2 fini et > global_l2 (front plus dur que le bulk)."""
    from scripts.run_w2_dambreak import dambreak_nwidth

    # Grille réduite pour la vitesse du test
    result = dambreak_nwidth(H=32, W=32, hl=0.1, x0=0.5, t_end=0.8,
                             energy_threshold=0.9999, max_modes=500)

    front_l2 = result["front_l2"]
    global_l2 = result["global_l2"]
    k = result["k"]

    assert np.isfinite(front_l2), f"front_l2 non fini : {front_l2}"
    assert np.isfinite(global_l2), f"global_l2 non fini : {global_l2}"
    assert k >= 1, f"k doit être >= 1, got {k}"
    assert front_l2 > global_l2, (
        f"Le front doit être plus dur que le bulk : front_l2={front_l2:.4f}, "
        f"global_l2={global_l2:.4f}"
    )
