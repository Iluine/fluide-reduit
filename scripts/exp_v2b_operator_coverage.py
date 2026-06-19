"""Expérience v2b — couverture-OPÉRATEUR : un A global tient-il la dynamique du canal
une fois qu'il l'a vue ? (analogue dynamique de v1b)

V2 a montré un cliff : rollout canal 157 % avec un opérateur DMD global fit sur
bosses+obstacles seuls (canal jamais dans le fit). v1b avait montré, côté
REPRÉSENTATION, que couvrir la topologie effondre le plafond (16→6 %). Question ici,
côté DYNAMIQUE : si on met des canaux dans le fit DMD (et dans la base, sinon le
plancher de représentation cape le rollout), le rollout du canal holdout chute-t-il ?

- chute vers le niveau topologie-vue (~5–9 %)  => opérateur coverage-limited : un seul
  A global tient le canal une fois vu => V3b NON requis.
- chute mais plafonne avec déphasage           => signature TRANSPORT dans la dynamique.
- reste haut malgré la couverture              => un seul A ne peut porter bosse ET
  canal (bathymétrie = terme source) => V3b (opérateur conditionné) justifié.

Le canal holdout (wall=1.0,y0=0.5,hw=8,soft=2) garde des params DIFFÉRENTS des canaux
d'entraînement (param-extrapolation dans la topologie canal).

Usage : .venv/bin/python scripts/exp_v2b_operator_coverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from config import GridConfig, SolverConfig
from src.terrains import (TerrainParams, make_terrain_from_params, rest_state_ic,
                          DROP_ICS, REST_SURFACE)
from src.solver import simulate
from scripts.run_v2_transfer import evaluate_transfer, _load

GRID = GridConfig(H=64, W=64, dx=1.0, dy=1.0)
SOLVER = SolverConfig(cfl=0.45, n_steps=800, save_every=4, min_depth=1e-3)


def main() -> None:
    H, W, base_train, eval_set = _load()  # train = bosses+obstacles ; eval = ref+holdouts

    # 4 canaux d'entraînement (params != holdout), 2 CI chacun
    train_channels = [
        TerrainParams("channel", amp=0.9, x0_frac=0.5, y0_frac=0.45, sigma=7.0, slope=2.0),
        TerrainParams("channel", amp=1.0, x0_frac=0.5, y0_frac=0.55, sigma=9.0, slope=3.0),
        TerrainParams("channel", amp=0.8, x0_frac=0.5, y0_frac=0.50, sigma=6.0, slope=2.0),
        TerrainParams("channel", amp=0.95, x0_frac=0.5, y0_frac=0.48, sigma=10.0, slope=3.0),
    ]
    chan_trajs = []  # [(canal0_ic0, canal0_ic1), (canal1_ic0, ...), ...]
    for p in train_channels:
        b = make_terrain_from_params(GRID, p)
        per = []
        for ic in ("drop_center", "drop_offset"):
            h0, u0, v0 = rest_state_ic(GRID, b, **DROP_ICS[ic], rest_surface=REST_SURFACE)
            hs, _, _, _ = simulate(h0, u0, v0, b, GRID, SOLVER)
            per.append(hs)
        chan_trajs.append(per)

    print("[v2b] couverture-opérateur : rollout du canal holdout vs n canaux dans le fit")
    print(f"{'n_canaux':>9} {'k':>4} {'rho':>6} {'train_ref':>10} {'interp':>8} "
          f"{'obstacle':>9} {'CANAL roll':>11} {'CANAL floor':>12}")
    for n in (0, 1, 2, 4):
        extra = [hs for per in chan_trajs[:n] for hs in per]  # n canaux x 2 CI
        out = evaluate_transfer(base_train + extra, eval_set, H, W, 0.9999, 2000)
        r = out["results"]
        ref = next(v for k, v in r.items() if k.startswith("train_ref"))
        print(f"{n:>9} {out['k']:>4} {out['rho']:>6.3f} {ref['rollout']:>10.4f} "
              f"{r['interp']['rollout']:>8.4f} {r['extrap_obstacle']['rollout']:>9.4f} "
              f"{r['extrap_channel']['rollout']:>11.4f} {r['extrap_channel']['floor']:>12.4f}")

    print("\nLecture : CANAL roll vers ~obstacle => opérateur coverage-limited (V3b non "
          "requis) ; plafonne au-dessus => résidu (transport / besoin de conditionnement "
          "V3b). CANAL floor = plancher de représentation (ce que la base permet).")


if __name__ == "__main__":
    main()
