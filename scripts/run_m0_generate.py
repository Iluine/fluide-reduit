"""M0 — Génère les rollouts oracle (vérité terrain) et une animation de contrôle.

Usage : .venv/bin/python scripts/run_m0_generate.py
Sorties : data/ground_truth/*.npz, outputs/m0_control_drop_center.(gif|png)
Interprétation attendue : masse conservée (~1e-9), ondes qui se propagent et
se réfléchissent sur les parois de façon physique."""
from __future__ import annotations

import sys
from pathlib import Path

# Rendre src/ et config importables quel que soit le CWD
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from config import GridConfig, SolverConfig
from src.solver import (make_terrain, simulate, initial_condition_dam_break,
                        initial_condition_gaussian_drop)
from src.io_utils import Dataset, save_dataset, save_animation
from src.metrics import mass_series

# ----------------------------- CONFIG ------------------------------------
GRID = GridConfig(H=64, W=64, dx=1.0, dy=1.0)
# n_steps relevé (et save_every=4) car la CFL 2D donne un dt plus petit : il faut
# ~800 pas pour que les ondes traversent le domaine (64 unités) et se réfléchissent.
# -> 201 frames par rollout, suffisant pour POD/DMD et l'évaluation long-horizon H2.
SOLVER = SolverConfig(cfl=0.45, n_steps=800, save_every=4, min_depth=1e-3)
TERRAIN_KIND = "bump"
OUT_DATA = ROOT / "data" / "ground_truth"
OUT_FIG = ROOT / "outputs"
# CI : (nom, fonction, est_test)
CASES = [
    ("drop_center", lambda g: initial_condition_gaussian_drop(g, cx_frac=0.5, cy_frac=0.5), False),
    ("drop_offset", lambda g: initial_condition_gaussian_drop(g, cx_frac=0.3, cy_frac=0.6), False),
    ("dam_break", lambda g: initial_condition_dam_break(g, 2.0, 1.0, 0.4), False),
    ("drop_test", lambda g: initial_condition_gaussian_drop(g, cx_frac=0.65, cy_frac=0.35, amp=0.6), True),
]
# -------------------------------------------------------------------------


def main() -> None:
    """Génère les rollouts oracle pour toutes les CI et sauvegarde les datasets."""
    b = make_terrain(GRID, TERRAIN_KIND)
    for name, ci_fn, is_test in CASES:
        h0, u0, v0 = ci_fn(GRID)
        hs, us, vs, dt = simulate(h0, u0, v0, b, GRID, SOLVER)
        masses = mass_series(hs, GRID.dx, GRID.dy)
        drift = float((np.abs(masses - masses[0]) / masses[0]).max())
        assert drift < 1e-8, f"{name}: dérive de masse {drift:.2e} trop élevée"
        meta = {"dx": GRID.dx, "dy": GRID.dy, "dt": dt, "ci": name,
                "schema": "lax-friedrichs", "cfl": SOLVER.cfl,
                "terrain": TERRAIN_KIND, "is_test": is_test}
        save_dataset(OUT_DATA / f"{name}.npz", Dataset(hs, us, vs, b, meta))
        print(f"[M0] {name:12s} T={hs.shape[0]:3d} dt={dt:.4e} dérive_masse={drift:.2e}")
        if name == "drop_center":
            written = save_animation(OUT_FIG / "m0_control_drop_center.gif", hs,
                                     fps=20, cmap="viridis", title="M0 — h(t) drop_center")
            print(f"[M0] animation de contrôle -> {written}")


if __name__ == "__main__":
    main()
