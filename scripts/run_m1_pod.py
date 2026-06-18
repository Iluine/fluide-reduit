"""M1 — POD : énergie cumulée, erreur de reconstruction vs k, modes spatiaux.

Usage : .venv/bin/python scripts/run_m1_pod.py
Interprétation attendue : l'énergie atteint 99 % avec peu de modes (quelques
dizaines) ; l'erreur de reconstruction est faible à k petit -> H1 (volet POD) OK."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from config import PODConfig
from src.io_utils import load_dataset
from src.pod import PODBasis, stack_snapshots, fit_pod, encode, decode, cumulative_energy

# ----------------------------- CONFIG ------------------------------------
POD = PODConfig(energy_threshold=0.99, max_modes=128)
TRAIN_CASES = ["drop_center", "drop_offset", "dam_break"]  # CI vues
DATA = ROOT / "data" / "ground_truth"
OUT = ROOT / "outputs"
# -------------------------------------------------------------------------


def main() -> None:
    """Calcule la base POD, sauvegarde pod_basis.npz et génère les figures d'énergie."""
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    cols = []
    H = W = None
    for name in TRAIN_CASES:
        ds = load_dataset(DATA / f"{name}.npz")
        H, W = ds.h.shape[1], ds.h.shape[2]
        cols.append(stack_snapshots(ds.h, ds.u, ds.v))
    X = np.concatenate(cols, axis=1)
    print(f"[M1] X shape = {X.shape} (n_features, n_snapshots)")

    basis = fit_pod(X, POD.energy_threshold, POD.max_modes)
    k = basis.Phi.shape[1]
    energy = cumulative_energy(basis.singular_values)
    print(f"[M1] k retenu pour {POD.energy_threshold:.0%} d'énergie = {k}")

    # Sauvegarde de la base pour M2/M3
    np.savez_compressed(ROOT / "data" / "pod_basis.npz",
                        mean=basis.mean, scale=basis.scale, Phi=basis.Phi,
                        singular_values=basis.singular_values, H=H, W=W)

    # Figure 1 : énergie cumulée
    plt.figure(figsize=(5, 4))
    plt.plot(np.arange(1, len(energy) + 1), energy, marker=".")
    plt.axhline(POD.energy_threshold, color="r", ls="--", label=f"{POD.energy_threshold:.0%}")
    plt.axvline(k, color="g", ls="--", label=f"k={k}")
    plt.xlabel("nombre de modes")
    plt.ylabel("énergie cumulée")
    plt.legend()
    plt.title("M1 — énergie cumulée POD")
    plt.tight_layout()
    plt.savefig(OUT / "m1_energy.png", dpi=120)
    plt.close()

    # Figure 2 : erreur de reconstruction vs k
    ks = [k_ for k_ in (1, 2, 4, 8, 16, 32, 64, 128) if k_ <= basis.Phi.shape[1]]
    errs = []
    for k_ in ks:
        b_k = PODBasis(mean=basis.mean, scale=basis.scale,
                       Phi=basis.Phi[:, :k_], singular_values=basis.singular_values)
        errs.append(np.linalg.norm(decode(b_k, encode(b_k, X)) - X) / np.linalg.norm(X))
    plt.figure(figsize=(5, 4))
    plt.semilogy(ks, errs, marker="o")
    plt.xlabel("k (nombre de modes)")
    plt.ylabel("erreur L2 relative")
    plt.title("M1 — erreur de reconstruction vs k")
    plt.tight_layout()
    plt.savefig(OUT / "m1_recon_error_vs_k.png", dpi=120)
    plt.close()
    print("[M1] erreur recon par k :", {k_: round(e, 4) for k_, e in zip(ks, errs)})

    # Figure 3 : 4 premiers modes spatiaux (canal h)
    hw = H * W
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for m in range(min(4, basis.Phi.shape[1])):
        mode_h = basis.Phi[:hw, m].reshape(H, W)
        axes[m].imshow(mode_h, cmap="RdBu_r", origin="lower")
        axes[m].set_title(f"mode {m} (h)")
        axes[m].axis("off")
    fig.suptitle("M1 — premiers modes spatiaux POD (canal h)")
    fig.tight_layout()
    fig.savefig(OUT / "m1_modes.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
