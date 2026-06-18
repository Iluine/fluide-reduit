"""M2 — DMD : fit sur les trajectoires latentes, rollout court vs vérité terrain.

Usage : .venv/bin/python scripts/run_m2_dmd.py
Interprétation attendue : à court horizon, le champ reconstruit par DMD suit la
vérité (baseline de référence). Le rayon spectral indique la tendance long-terme."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.io_utils import load_dataset, save_animation
from src.pod import PODBasis, encode, decode, stack_snapshots, unstack
from src.dmd import fit_dmd, rollout, spectral_radius

# ----------------------------- CONFIG ------------------------------------
TRAIN_CASES = ["drop_center", "drop_offset", "dam_break"]
DEMO_CASE = "drop_center"   # rollout démontré
SHORT_HORIZON = 40          # nombre de pas du rollout court
DATA = ROOT / "data"; GT = DATA / "ground_truth"; OUT = ROOT / "outputs"
# -------------------------------------------------------------------------


def load_basis() -> tuple[PODBasis, int, int]:
    d = np.load(DATA / "pod_basis.npz")
    basis = PODBasis(d["mean"], d["scale"], d["Phi"], d["singular_values"])
    return basis, int(d["H"]), int(d["W"])


def main() -> None:
    basis, H, W = load_basis()
    z_list = []
    for name in TRAIN_CASES:
        ds = load_dataset(GT / f"{name}.npz")
        z_list.append(encode(basis, stack_snapshots(ds.h, ds.u, ds.v)))
    A = fit_dmd(z_list)
    np.savez_compressed(DATA / "dmd_A.npz", A=A)
    rho = spectral_radius(A)
    print(f"[M2] rayon spectral de A = {rho:.4f} ({'stable' if rho <= 1.0 else 'CROISSANT'})")

    ds = load_dataset(GT / f"{DEMO_CASE}.npz")
    z_true = encode(basis, stack_snapshots(ds.h, ds.u, ds.v))
    n = min(SHORT_HORIZON, z_true.shape[1] - 1)
    z_pred = rollout(A, z_true[:, 0], n)
    h_pred, _, _ = unstack(decode(basis, z_pred), H, W)
    h_true = ds.h[:n + 1]

    # Animation côte à côte (concat horizontale vérité | prédiction)
    side = np.concatenate([h_true, h_pred], axis=2)  # (n+1, H, 2W)
    written = save_animation(OUT / "m2_dmd_vs_truth.gif", side, fps=15,
                             title=f"M2 — {DEMO_CASE} : vérité | DMD")
    print(f"[M2] animation -> {written}")


if __name__ == "__main__":
    main()
