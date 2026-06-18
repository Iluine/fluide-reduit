"""M3 — H2 : dérive du rollout DMD sur long horizon (CI vue ET CI de test).

Usage : .venv/bin/python scripts/run_m3_eval_rollout.py
Interprétation attendue : erreur relative de HAUTEUR (h) bornée, vitesses
rapportées en RMS absolu (non explosif quand ‖u‖→0), dérive masse ~2 %."""
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
from src.metrics import error_growth, rms_growth, mass_series

# ----------------------------- CONFIG ------------------------------------
SEEN_CASE = "drop_center"   # CI vue à l'entraînement
TEST_CASE = "drop_test"     # CI mise de côté (généralisation)
DATA = ROOT / "data"
GT = DATA / "ground_truth"
OUT = ROOT / "outputs"
# -------------------------------------------------------------------------


def load_basis():
    """Recharge la base POD (data/pod_basis.npz) en PODBasis + (H, W)."""
    with np.load(DATA / "pod_basis.npz") as d:
        basis = PODBasis(d["mean"], d["scale"], d["Phi"], d["singular_values"])
        H, W = int(d["H"]), int(d["W"])
    return basis, H, W


def evaluate(name, basis, A, H, W):
    """Rollout long-horizon vs vérité ; retourne dict de métriques par canal."""
    ds = load_dataset(GT / f"{name}.npz")
    dx, dy = ds.meta["dx"], ds.meta["dy"]
    X_true = stack_snapshots(ds.h, ds.u, ds.v)
    z_true = encode(basis, X_true)
    T = z_true.shape[1]
    z_pred = rollout(A, z_true[:, 0], T - 1)
    X_pred = decode(basis, z_pred)
    h_pred, u_pred, v_pred = unstack(X_pred, H, W)
    eh = error_growth(h_pred, ds.h)       # erreur L2 relative HEIGHT par frame
    ru = rms_growth(u_pred, ds.u)         # RMS absolu vitesse u par frame
    rv = rms_growth(v_pred, ds.v)         # RMS absolu vitesse v par frame
    # Reference velocity: max-over-time per-frame RMS of TRUE velocity
    u_ref = float(np.max(np.sqrt(np.mean(ds.u**2, axis=(1, 2)))))
    v_ref = float(np.max(np.sqrt(np.mean(ds.v**2, axis=(1, 2)))))
    return {
        "eh": eh,
        "ru": ru,
        "rv": rv,
        "u_ref": u_ref,
        "v_ref": v_ref,
        "mass_pred": mass_series(h_pred, dx, dy),
        "mass_true": mass_series(ds.h, dx, dy),
        "h_true": ds.h,
        "h_pred": h_pred,
    }


def main() -> None:
    """Évalue H2 par canal : erreur hauteur (relative) + vitesses (RMS absolu) + dérive de masse."""
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    basis, H, W = load_basis()
    with np.load(DATA / "dmd_A.npz") as _d:
        A = _d["A"]

    results = {name: evaluate(name, basis, A, H, W) for name in (SEEN_CASE, TEST_CASE)}

    # Figure 1 : erreur relative de HAUTEUR (h) — vue + test
    plt.figure(figsize=(6, 4))
    for name in (SEEN_CASE, TEST_CASE):
        eh = results[name]["eh"]
        plt.plot(eh, label=f"{name} (final={eh[-1]:.3f})")
    plt.xlabel("pas de temps")
    plt.ylabel("erreur L2 relative")
    plt.title("H2 — erreur relative de HAUTEUR (h) vs temps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "m3_error_growth.png", dpi=120)
    plt.close()

    # Figure 2 : RMS absolu vitesses u et v — vue + test
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in (SEEN_CASE, TEST_CASE):
        ru = results[name]["ru"]
        rv = results[name]["rv"]
        u_ref = results[name]["u_ref"]
        v_ref = results[name]["v_ref"]
        axes[0].plot(ru, label=f"{name} (réf={u_ref:.3f} m/s)")
        axes[1].plot(rv, label=f"{name} (réf={v_ref:.3f} m/s)")
    axes[0].set_title("RMS absolu u")
    axes[0].set_xlabel("pas de temps")
    axes[0].set_ylabel("RMS [m/s]")
    axes[0].legend()
    axes[1].set_title("RMS absolu v")
    axes[1].set_xlabel("pas de temps")
    axes[1].set_ylabel("RMS [m/s]")
    axes[1].legend()
    fig.suptitle("H2 — erreur RMS absolue des vitesses (u, v) vs temps")
    fig.tight_layout()
    fig.savefig(OUT / "m3_velocity_rms.png", dpi=120)
    plt.close(fig)

    # Figure 3 : dérive de masse (prédite vs vérité)
    plt.figure(figsize=(6, 4))
    for name in (SEEN_CASE, TEST_CASE):
        m_pred = results[name]["mass_pred"]
        m_true = results[name]["mass_true"]
        plt.plot((m_pred - m_true[0]) / m_true[0], label=f"{name} prédit")
        plt.plot((m_true - m_true[0]) / m_true[0], ls="--", label=f"{name} vérité")
    plt.xlabel("pas de temps")
    plt.ylabel("dérive relative de masse")
    plt.title("M3 — H2 : dérive de la masse totale")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "m3_mass_drift.png", dpi=120)
    plt.close()

    # Animations long-horizon côte à côte + verdict chiffré
    verdicts = {}
    for name in (SEEN_CASE, TEST_CASE):
        r = results[name]
        h_true, h_pred = r["h_true"], r["h_pred"]
        side = np.concatenate([h_true, h_pred], axis=2)
        save_animation(OUT / f"m3_longhorizon_{name}.gif", side, fps=15,
                       title=f"M3 — {name} : vérité | DMD (long horizon)")
        m_pred = r["mass_pred"]
        m_true = r["mass_true"]
        verdicts[name] = {
            "h_rel_final": float(r["eh"][-1]),
            "h_rel_max": float(r["eh"].max()),
            "u_rms_final": float(r["ru"][-1]),
            "v_rms_final": float(r["rv"][-1]),
            "u_ref": float(r["u_ref"]),
            "v_ref": float(r["v_ref"]),
            "mass_drift_final": float((m_pred[-1] - m_true[0]) / m_true[0]),
            "exploded": bool(r["eh"].max() > 5.0 or not np.isfinite(r["eh"]).all()),
        }

    np.savez_compressed(DATA / "m3_eval.npz",
                        **{f"{n}_eh": results[n]["eh"] for n in results},
                        **{f"{n}_ru": results[n]["ru"] for n in results},
                        **{f"{n}_rv": results[n]["rv"] for n in results})

    print("[M3] verdict H2 par canal :")
    for name, vd in verdicts.items():
        u_pct = 100.0 * vd["u_rms_final"] / vd["u_ref"] if vd["u_ref"] > 0 else float("nan")
        v_pct = 100.0 * vd["v_rms_final"] / vd["v_ref"] if vd["v_ref"] > 0 else float("nan")
        print(f"   {name:12s} "
              f"h_rel_final={vd['h_rel_final']:.3f} h_rel_max={vd['h_rel_max']:.3f} "
              f"u_rms={vd['u_rms_final']:.4f} (={u_pct:.0f}% de la réf {vd['u_ref']:.3f} m/s) "
              f"v_rms={vd['v_rms_final']:.4f} (={v_pct:.0f}% de la réf {vd['v_ref']:.3f} m/s) "
              f"dérive_masse_finale={vd['mass_drift_final']:.2e} explosé={vd['exploded']}")


if __name__ == "__main__":
    main()
