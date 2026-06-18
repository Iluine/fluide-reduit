# tests/test_run_v0.py
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import GridConfig, SolverConfig
from src.io_utils import load_dataset
from src.terrains import sample_split


def test_generate_split_writes_contract(tmp_path):
    from scripts.run_v0_generate import generate_split
    grid = GridConfig(H=64, W=64, dx=1.0, dy=1.0)
    # config réduite pour un test rapide : peu de pas, sous-ensemble de terrains
    solver = SolverConfig(cfl=0.45, n_steps=12, save_every=4, min_depth=1e-3)
    entries = [sample_split(grid)[0],                       # 1 train (bump)
               [e for e in sample_split(grid) if e.regime == "extrap_channel"][0]]
    data_dir = tmp_path / "data"
    fig_dir = tmp_path / "fig"
    report = generate_split(data_dir, fig_dir, grid, solver, entries,
                            save_extrap_anim=False)

    # contrat fichiers : un .npz par (terrain, CI)
    train0 = sample_split(grid)[0]
    npz = data_dir / f"{train0.terrain_id}__{train0.ic_ids[0]}.npz"
    assert npz.exists()
    ds = load_dataset(npz)
    assert ds.h.shape[1:] == (64, 64)
    assert ds.meta["regime"] == "train"
    assert "theta" in np.load(npz).files  # theta présent dans le .npz

    # split.json présent et structuré
    split = json.loads((data_dir / "split.json").read_text())
    assert "entries" in split and "rest_surface" in split
    roles = {e["regime"] for e in split["entries"]}
    assert {"train", "extrap_channel"} <= roles

    # garde-fous reportés et respectés
    assert report["max_mass_drift"] < 1e-7
    assert report["min_depth"] > 0.0
    assert all(np.isfinite(v) for v in report["rest_residual_surf"].values())
