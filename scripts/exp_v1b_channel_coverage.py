"""Expérience v1b — couverture vs capacité sur la topologie canal.

V1 a mesuré un plafond de 16.4 % sur le canal — MAIS le canal était holdout-only
(jamais en entraînement). Cette expérience tranche : ce plafond est-il une limite
de COUVERTURE (la base n'a pas de mode en forme de bande parce qu'aucun canal
n'était en train) ou de CAPACITÉ (la POD linéaire ne sait pas représenter un
canal) ?

Protocole : on reconstruit le MÊME canal holdout (1) avec la base d'origine
(bosses+obstacles seuls) puis (2) avec une base à laquelle on a ajouté quelques
canaux d'entraînement (paramètres DIFFÉRENTS du holdout). On surveille k.

Prédiction (hypothèse couverture) : le plafond canal s'effondre vers le niveau
obstacle (~2 %), et k ne croît que modérément. Si c'est le cas, la représentation
n'est pas le goulot dès que le train couvre le vocabulaire de topologies.

Usage : .venv/bin/python scripts/exp_v1b_channel_coverage.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from config import GridConfig, SolverConfig
from src.terrains import (TerrainParams, make_terrain_from_params, rest_state_ic,
                          DROP_ICS, REST_SURFACE)
from src.solver import simulate
from src.pod import fit_pod, encode, decode, stack_height, unstack_height
from src.io_utils import load_dataset
from src.metrics import relative_l2_error

GRID = GridConfig(H=64, W=64, dx=1.0, dy=1.0)
SOLVER = SolverConfig(cfl=0.45, n_steps=800, save_every=4, min_depth=1e-3)
DATA = ROOT / "data" / "v2"
ENERGY, MAX_MODES = 0.9999, 2000


def recon_err(basis, h_seq):
    X = stack_height(h_seq)
    seq_r = unstack_height(decode(basis, encode(basis, X)), GRID.H, GRID.W)
    return relative_l2_error(seq_r, h_seq)


def main() -> None:
    split = json.loads((DATA / "split.json").read_text())

    # 1. trajectoires train d'origine (bosses + obstacles), hauteur seule
    train_h = []
    for e in split["entries"]:
        if e["role"] == "train":
            for ic in e["ic_ids"]:
                train_h.append(load_dataset(DATA / f"{e['terrain_id']}__{ic}.npz").h)

    # cibles holdout (vérité)
    ch = load_dataset(DATA / "holdout_extrap_channel__drop_new.npz").h
    ob = load_dataset(DATA / "holdout_extrap_obstacle__drop_new.npz").h
    itp = load_dataset(DATA / "holdout_interp__drop_new.npz").h

    # 2. BASELINE : base d'origine (sans canal en train) — doit reproduire V1
    X0 = np.concatenate([stack_height(h) for h in train_h], axis=1)
    b0 = fit_pod(X0, ENERGY, MAX_MODES, n_channels=1)
    print(f"[baseline sans canal en train] k={b0.Phi.shape[1]}")
    print(f"  interp={recon_err(b0, itp):.4f}  extrap_obstacle={recon_err(b0, ob):.4f}"
          f"  CANAL={recon_err(b0, ch):.4f}")

    # 3. canaux d'ENTRAÎNEMENT (params DIFFÉRENTS du holdout wall=1.0,y0=0.5,hw=8,
    #    soft=2), submergés, 2 CI chacun (pour égaler la couverture obstacle :
    #    l'obstacle holdout se reconstruit à 1.9 % avec 4 obstacles x 2 CI en train)
    train_channels = [
        TerrainParams("channel", amp=0.9, x0_frac=0.5, y0_frac=0.45, sigma=7.0, slope=2.0),
        TerrainParams("channel", amp=1.0, x0_frac=0.5, y0_frac=0.55, sigma=9.0, slope=3.0),
        TerrainParams("channel", amp=0.8, x0_frac=0.5, y0_frac=0.50, sigma=6.0, slope=2.0),
        TerrainParams("channel", amp=0.95, x0_frac=0.5, y0_frac=0.48, sigma=10.0, slope=3.0),
    ]
    # trajectoires des canaux train : 2 CI chacun (drop_center, drop_offset)
    chan_trajs = []  # liste de (h_seq) par (canal, CI), ordonnée canal par canal
    for p in train_channels:
        b = make_terrain_from_params(GRID, p)
        assert REST_SURFACE - float(b.max()) >= 0.2, "canal train non submergé"
        per_chan = []
        for ic in ("drop_center", "drop_offset"):
            h0, u0, v0 = rest_state_ic(GRID, b, **DROP_ICS[ic], rest_surface=REST_SURFACE)
            hs, _, _, _ = simulate(h0, u0, v0, b, GRID, SOLVER)
            per_chan.append(hs)
        chan_trajs.append(per_chan)

    # 4. balayage de couverture : 0, 1, 2, 4 canaux (x2 CI) en train
    print("\n[balayage de couverture canal] (n_canaux x 2 CI)")
    print(f"{'n_canaux':>9} {'k':>4} {'interp':>8} {'obstacle':>9} {'CANAL':>8}")
    for n in (0, 1, 2, 4):
        extra = [hs for pc in chan_trajs[:n] for hs in pc]  # n canaux x 2 CI
        X = np.concatenate([stack_height(h) for h in train_h + extra], axis=1)
        bN = fit_pod(X, ENERGY, MAX_MODES, n_channels=1)
        print(f"{n:>9} {bN.Phi.shape[1]:>4} {recon_err(bN, itp):>8.4f} "
              f"{recon_err(bN, ob):>9.4f} {recon_err(bN, ch):>8.4f}")
    print("\nLecture : si CANAL chute monotone vers ~obstacle (couverture égalée) => "
          "COUVERTURE ; s'il plafonne au-dessus => coût de représentation de la bande.")


if __name__ == "__main__":
    main()
