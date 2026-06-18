"""M7 — Rendu de l'état prédit (POD+DMD) : heatmap de h et surface eta=h+b.

Usage : .venv/bin/python scripts/run_m7_render.py
Interprétation attendue : l'animation prédite est visuellement plausible et
temporellement cohérente."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.io_utils import load_dataset
from src.pod import PODBasis, encode, decode, stack_snapshots, unstack
from src.dmd import rollout
from src.render import surface_height, render_rollout

# ----------------------------- CONFIG ------------------------------------
CASE = "drop_center"
DATA = ROOT / "data"; GT = DATA / "ground_truth"; OUT = ROOT / "outputs"
# -------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA / "pod_basis.npz")
    basis = PODBasis(d["mean"], d["scale"], d["Phi"], d["singular_values"])
    H, W = int(d["H"]), int(d["W"])
    A = np.load(DATA / "dmd_A.npz")["A"]

    ds = load_dataset(GT / f"{CASE}.npz")
    z_true = encode(basis, stack_snapshots(ds.h, ds.u, ds.v))
    z_pred = rollout(A, z_true[:, 0], z_true.shape[1] - 1)
    h_pred, _, _ = unstack(decode(basis, z_pred), H, W)

    out1 = render_rollout(OUT / "m7_height_heatmap.gif", h_pred, cmap="viridis",
                          title=f"M7 — h prédit ({CASE})")
    out2 = render_rollout(OUT / "m7_surface_eta.gif", surface_height(h_pred, ds.b),
                          cmap="terrain", title=f"M7 — surface eta=h+b ({CASE})")
    print(f"[M7] rendus -> {out1} ; {out2}")


if __name__ == "__main__":
    main()
