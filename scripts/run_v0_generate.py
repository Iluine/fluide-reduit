"""V0 — Génère le dataset v2 (famille de terrains × CI) et applique les garde-fous
d'oracle. La conservation de masse ne suffit pas : on vérifie aussi la positivité
avec marge (assèchement) et le résidu au repos (well-balancedness), et on rend les
animations oracle des terrains d'extrapolation pour le sanity visuel.

Usage : .venv/bin/python scripts/run_v0_generate.py
Sorties : data/v2/<terrain_id>__<ic_id>.npz, data/v2/split.json,
          outputs/v2/v0_oracle_<terrain>.gif (terrains extrap)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from config import GridConfig, SolverConfig
from src.terrains import (REST_SURFACE, DROP_ICS, sample_split,
                          make_terrain_from_params, rest_state_ic, rest_residual)
from src.solver import simulate
from src.io_utils import Dataset, save_dataset, save_animation
from src.metrics import mass_series
from src.render import surface_height

# ----------------------------- CONFIG ------------------------------------
GRID = GridConfig(H=64, W=64, dx=1.0, dy=1.0)
SOLVER = SolverConfig(cfl=0.45, n_steps=800, save_every=4, min_depth=1e-3)
OUT_DATA = ROOT / "data" / "v2"
OUT_FIG = ROOT / "outputs" / "v2"
POSITIVITY_MARGIN = 0.1   # min(h) doit rester > cette marge sur toute la trajectoire
# Bornes de SÉCURITÉ du résidu au repos (well-balancedness) : elles attrapent une
# vraie rupture (instabilité / courants parasites massifs qui invalident l'oracle),
# pas la non-well-balancedness légère. Le jugement fin « assez petit ? » se fait sur
# les VALEURS rapportées (report["rest_residual_*"]) + les animations oracle extrap.
REST_SURF_DEV_TOL = 0.15  # ~10% de REST_SURFACE
REST_SPEED_TOL = 0.5
# -------------------------------------------------------------------------


def generate_split(out_data_dir, out_fig_dir, grid: GridConfig, solver_cfg: SolverConfig,
                   entries, rest_surface: float = REST_SURFACE,
                   save_extrap_anim: bool = True) -> dict:
    """Génère un .npz par (terrain, CI), applique les garde-fous, écrit split.json,
    rend les animations oracle des terrains extrap. Retourne un rapport."""
    out_data_dir = Path(out_data_dir)
    out_fig_dir = Path(out_fig_dir)
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    report = {"max_mass_drift": 0.0, "min_depth": np.inf,
              "rest_residual_surf": {}, "rest_residual_speed": {}, "trajectories": []}
    split_entries = []

    for e in entries:
        b = make_terrain_from_params(grid, e.params)
        assert rest_surface - float(b.max()) >= 0.0, f"{e.terrain_id}: terrain non submergé"

        # garde-fou de well-balancedness (une fois par terrain)
        surf_dev, speed = rest_residual(grid, b, solver_cfg, rest_surface, n_steps=50)
        report["rest_residual_surf"][e.terrain_id] = surf_dev
        report["rest_residual_speed"][e.terrain_id] = speed
        assert surf_dev < REST_SURF_DEV_TOL, (
            f"{e.terrain_id}: résidu de surface au repos {surf_dev:.3e} "
            f">= {REST_SURF_DEV_TOL} (bathymétrie trop raide)")
        assert speed < REST_SPEED_TOL, (
            f"{e.terrain_id}: vitesse parasite au repos {speed:.3e} >= {REST_SPEED_TOL}")

        for ic_id in e.ic_ids:
            h0, u0, v0 = rest_state_ic(grid, b, **DROP_ICS[ic_id], rest_surface=rest_surface)
            hs, us, vs, dt = simulate(h0, u0, v0, b, grid, solver_cfg)

            masses = mass_series(hs, grid.dx, grid.dy)
            drift = float((np.abs(masses - masses[0]) / masses[0]).max())
            min_depth = float(hs.min())
            assert drift < 1e-7, f"{e.terrain_id}__{ic_id}: dérive masse {drift:.2e}"
            assert min_depth > POSITIVITY_MARGIN, (
                f"{e.terrain_id}__{ic_id}: min(h)={min_depth:.3f} <= "
                f"{POSITIVITY_MARGIN} (assèchement)")
            report["max_mass_drift"] = max(report["max_mass_drift"], drift)
            report["min_depth"] = min(report["min_depth"], min_depth)
            report["trajectories"].append(f"{e.terrain_id}__{ic_id}")

            meta = {"dx": grid.dx, "dy": grid.dy, "dt": dt, "schema": "lax-friedrichs",
                    "cfl": solver_cfg.cfl, "terrain_id": e.terrain_id, "ic_id": ic_id,
                    "regime": e.regime, "role": e.role, "rest_surface": rest_surface}
            ds = Dataset(hs, us, vs, b, meta)
            path = out_data_dir / f"{e.terrain_id}__{ic_id}.npz"
            # save_dataset n'écrit pas theta : on l'ajoute via un re-dump complet
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path, h=hs.astype(np.float64), u=us.astype(np.float64),
                v=vs.astype(np.float64), b=b.astype(np.float64),
                theta=e.params.to_vector(), meta_json=np.array(json.dumps(meta)))
            print(f"[V0] {e.terrain_id:24s} {ic_id:11s} regime={e.regime:16s} "
                  f"drift={drift:.2e} min_h={min_depth:.3f}")

            if save_extrap_anim and e.role == "holdout_extrap":
                eta = surface_height(hs, b)
                written = save_animation(
                    out_fig_dir / f"v0_oracle_{e.terrain_id}.gif", eta, fps=20,
                    cmap="viridis", title=f"V0 oracle η — {e.terrain_id} ({e.regime})")
                print(f"[V0] animation oracle extrap -> {written}")

        split_entries.append({"terrain_id": e.terrain_id, "role": e.role,
                              "regime": e.regime, "params": e.params.to_dict(),
                              "theta": e.params.to_vector().tolist(),
                              "ic_ids": list(e.ic_ids)})

    split = {"rest_surface": rest_surface, "grid": {"H": grid.H, "W": grid.W,
             "dx": grid.dx, "dy": grid.dy}, "entries": split_entries}
    (out_data_dir / "split.json").write_text(json.dumps(split, indent=2))
    report["min_depth"] = float(report["min_depth"])
    return report


def main() -> None:
    entries = sample_split(GRID)
    report = generate_split(OUT_DATA, OUT_FIG, GRID, SOLVER, entries)
    print(f"\n[V0] {len(report['trajectories'])} trajectoires générées.")
    print(f"[V0] dérive de masse max = {report['max_mass_drift']:.2e}")
    print(f"[V0] profondeur min sur tout le dataset = {report['min_depth']:.3f} "
          f"(marge {POSITIVITY_MARGIN})")
    print(f"[V0] résidu de surface au repos max = "
          f"{max(report['rest_residual_surf'].values()):.3e} (tol {REST_SURF_DEV_TOL})")


if __name__ == "__main__":
    main()
