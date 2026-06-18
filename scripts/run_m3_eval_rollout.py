"""M3 — H2 : dérive du rollout DMD sur long horizon (CI vue ET CI de test).

Usage : .venv/bin/python scripts/run_m3_eval_rollout.py
Interprétation attendue : caractériser si/quand le rollout dérive ou explose, et
si la masse totale prédite dérive. Résultat valable même s'il est négatif.
L'erreur L2 relative est calculée sur l'état empilé [h,u,v] et est donc dominée par le canal h (|u|,|v| ~ 0.1 vs h ~ 1)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.io_utils import load_dataset, save_animation
from src.pod import PODBasis, encode, decode, stack_snapshots, unstack
from src.dmd import rollout
from src.metrics import error_growth, mass_series

# ----------------------------- CONFIG ------------------------------------
SEEN_CASE = "drop_center"   # CI vue à l'entraînement
TEST_CASE = "drop_test"     # CI mise de côté (généralisation)
DATA = ROOT / "data"; GT = DATA / "ground_truth"; OUT = ROOT / "outputs"
# -------------------------------------------------------------------------


def load_basis():
    """Recharge la base POD (data/pod_basis.npz) en PODBasis + (H, W)."""
    d = np.load(DATA / "pod_basis.npz")
    return PODBasis(d["mean"], d["scale"], d["Phi"], d["singular_values"]), int(d["H"]), int(d["W"])


def evaluate(name, basis, A, H, W):
    """Rollout long-horizon vs vérité ; retourne (err_curve, mass_pred, mass_true, h_true, h_pred, dx, dy)."""
    ds = load_dataset(GT / f"{name}.npz")
    dx, dy = ds.meta["dx"], ds.meta["dy"]
    X_true = stack_snapshots(ds.h, ds.u, ds.v)
    z_true = encode(basis, X_true)
    T = z_true.shape[1]
    z_pred = rollout(A, z_true[:, 0], T - 1)
    X_pred = decode(basis, z_pred)
    h_pred, _, _ = unstack(X_pred, H, W)
    err = error_growth(X_pred.T.reshape(T, -1), X_true.T.reshape(T, -1))
    return (err, mass_series(h_pred, dx, dy), mass_series(ds.h, dx, dy),
            ds.h, h_pred, dx, dy)


def main() -> None:
    """Évalue H2 : croissance d'erreur + dérive de masse (CI vue et CI test)."""
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    basis, H, W = load_basis()
    A = np.load(DATA / "dmd_A.npz")["A"]

    results = {name: evaluate(name, basis, A, H, W) for name in (SEEN_CASE, TEST_CASE)}

    # Figure 1 : croissance d'erreur (vue + test)
    plt.figure(figsize=(6, 4))
    for name in (SEEN_CASE, TEST_CASE):
        err = results[name][0]
        plt.plot(err, label=f"{name} (final={err[-1]:.3f})")
    plt.xlabel("pas de temps"); plt.ylabel("erreur L2 relative")
    plt.title("M3 — H2 : croissance d'erreur du rollout"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "m3_error_growth.png", dpi=120); plt.close()

    # Figure 2 : dérive de masse (prédite vs vérité)
    plt.figure(figsize=(6, 4))
    for name in (SEEN_CASE, TEST_CASE):
        _, m_pred, m_true, *_ = results[name]
        plt.plot((m_pred - m_true[0]) / m_true[0], label=f"{name} prédit")
        plt.plot((m_true - m_true[0]) / m_true[0], ls="--", label=f"{name} vérité")
    plt.xlabel("pas de temps"); plt.ylabel("dérive relative de masse")
    plt.title("M3 — H2 : dérive de la masse totale"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "m3_mass_drift.png", dpi=120); plt.close()

    # Animations long-horizon côte à côte + verdict chiffré
    verdicts = {}
    for name in (SEEN_CASE, TEST_CASE):
        err, m_pred, m_true, h_true, h_pred, *_ = results[name]
        side = np.concatenate([h_true, h_pred], axis=2)
        save_animation(OUT / f"m3_longhorizon_{name}.gif", side, fps=15,
                       title=f"M3 — {name} : vérité | DMD (long horizon)")
        verdicts[name] = {
            "err_final": float(err[-1]),
            "err_max": float(err.max()),
            "exploded": bool(err.max() > 5.0 or not np.isfinite(err).all()),
            "mass_drift_final": float((m_pred[-1] - m_true[0]) / m_true[0]),
        }
    np.savez_compressed(DATA / "m3_eval.npz",
                        **{f"{n}_err": results[n][0] for n in results})
    print("[M3] verdict H2 :")
    for name, vd in verdicts.items():
        print(f"   {name:12s} err_final={vd['err_final']:.3f} err_max={vd['err_max']:.3f} "
              f"explosé={vd['exploded']} dérive_masse_finale={vd['mass_drift_final']:.2e}")


if __name__ == "__main__":
    main()
