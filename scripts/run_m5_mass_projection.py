"""M5 — Évaluation de la projection de masse comme garde-fou de sortie.

Compare le rollout DMD (clipped, ρ=1.0) avec et sans projection de masse :
  - OFF : h_pred brut (dérive masse ~2 %, croissante)
  - ON  : h_proj = project_mass_series(h_pred, m_target, dx, dy)
            → dérive ramenée à la précision machine

Hypothèse BC : parois réfléchissantes → masse vraie exactement conservée
(vérifiée par un assert sur la dérive relative < 1e-6).

Usage : .venv/bin/python scripts/run_m5_mass_projection.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.io_utils import load_dataset
from src.pod import PODBasis, encode, decode, stack_snapshots, unstack
from src.dmd import rollout
from src.metrics import total_mass, mass_series, error_growth
from src.mass_projection import project_mass_series

# ----------------------------- CONFIG ------------------------------------
SEEN_CASE = "drop_center"
TEST_CASE  = "drop_test"
DATA = ROOT / "data"
GT   = DATA / "ground_truth"
OUT  = ROOT / "outputs"
TMP  = Path("/tmp/sdd-pocphysicator")
# -------------------------------------------------------------------------


def load_basis() -> tuple[PODBasis, int, int]:
    """Recharge la base POD depuis data/pod_basis.npz."""
    with np.load(DATA / "pod_basis.npz") as d:
        basis = PODBasis(d["mean"], d["scale"], d["Phi"], d["singular_values"])
        H, W = int(d["H"]), int(d["W"])
    return basis, H, W


def evaluate_case(name: str, basis: PODBasis, A: np.ndarray, H: int, W: int) -> dict:
    """Rollout + projection de masse pour un scénario donné."""
    ds = load_dataset(GT / f"{name}.npz")
    dx = float(ds.meta["dx"])
    dy = float(ds.meta["dy"])

    # --- Vérification BC : masse vraie exactement conservée (parois réfléchissantes)
    m_true_series = mass_series(ds.h, dx, dy)
    m_mean = float(m_true_series.mean())
    true_drift = float((m_true_series.max() - m_true_series.min()) / m_mean)
    assert true_drift < 1e-6, (
        f"[M5/{name}] dérive masse vraie = {true_drift:.2e} >= 1e-6 — "
        "les BCs ne semblent pas réfléchissantes ; il faudrait projeter sur la masse vraie instantanée."
    )
    print(f"[M5/{name}] vérification BC : dérive masse vraie = {true_drift:.2e}  (< 1e-6 ✓)")

    m_target = total_mass(ds.h[0], dx, dy)
    T = ds.h.shape[0]

    # Rollout
    X_true = stack_snapshots(ds.h, ds.u, ds.v)
    z      = encode(basis, X_true)
    z_pred = rollout(A, z[:, 0], T - 1)
    X_pred = decode(basis, z_pred)
    h_pred, _, _ = unstack(X_pred, H, W)

    # Projection de masse
    h_proj = project_mass_series(h_pred, m_target, dx, dy)

    # Dérive de masse relative
    drift_off = (mass_series(h_pred, dx, dy) - m_target) / m_target
    drift_on  = (mass_series(h_proj, dx, dy) - m_target) / m_target

    # Erreur hauteur
    eh_off = error_growth(h_pred, ds.h)
    eh_on  = error_growth(h_proj, ds.h)

    return {
        "drift_off": drift_off,
        "drift_on":  drift_on,
        "eh_off":    eh_off,
        "eh_on":     eh_on,
        "true_drift": true_drift,
    }


def main() -> None:
    """Évalue M5 : projection de masse sur le rollout DMD clipped (ρ=1.0)."""
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    basis, H, W = load_basis()
    with np.load(DATA / "dmd_A.npz") as d:
        A = d["A"]

    results = {}
    for name in (SEEN_CASE, TEST_CASE):
        results[name] = evaluate_case(name, basis, A, H, W)

    # --- Figure : dérive de masse OFF vs ON (seen + test)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, (SEEN_CASE, TEST_CASE)):
        r = results[name]
        ax.plot(r["drift_off"] * 100, label="OFF (brut)", color="tab:red")
        ax.plot(r["drift_on"]  * 100, label="ON (projeté)", color="tab:blue", lw=1.5)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_xlabel("pas de temps")
        ax.set_ylabel("dérive de masse (%)")
        ax.set_title(f"{name}")
        ax.legend()
    fig.suptitle("M5 — Dérive de masse : OFF (DMD brut) vs ON (projection uniforme)")
    fig.tight_layout()
    fig_path = OUT / "m5_mass_drift.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"[M5] figure sauvegardée : {fig_path}")

    # Copie pour le contrôleur
    import shutil
    shutil.copy(fig_path, TMP / "m5_check.png")
    print(f"[M5] copie contrôleur  : {TMP / 'm5_check.png'}")

    # --- Verdicts chiffrés
    print("\n[M5] verdict par scénario :")
    for name in (SEEN_CASE, TEST_CASE):
        r = results[name]
        d_off_final = float(r["drift_off"][-1]) * 100
        d_on_final  = float(r["drift_on"][-1])  * 100
        eh_off_fin  = float(r["eh_off"][-1])
        eh_on_fin   = float(r["eh_on"][-1])
        eh_off_max  = float(r["eh_off"].max())
        eh_on_max   = float(r["eh_on"].max())
        print(
            f"   {name:12s}  "
            f"mass_drift_final OFF={d_off_final:+.3f}%  ON={d_on_final:+.2e}%  |  "
            f"h_rel_final OFF={eh_off_fin:.4f}  ON={eh_on_fin:.4f}  |  "
            f"h_rel_max   OFF={eh_off_max:.4f}  ON={eh_on_max:.4f}"
        )


if __name__ == "__main__":
    main()
