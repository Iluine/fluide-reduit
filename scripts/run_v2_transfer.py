"""V2 — Transfert naïf : un opérateur DMD GLOBAL prédit-il le rollout sur un terrain
nouveau ?

V1+v1b ont montré que la REPRÉSENTATION (base POD hauteur) tient sur terrain nouveau
dès que le train couvre le vocabulaire. V2 isole la question suivante : la DYNAMIQUE.
On ajuste un seul opérateur latent linéaire A (DMD écrêté) sur TOUS les terrains
d'entraînement, puis on déroule sur les terrains holdout depuis leur CI vraie. La
bathymétrie entre dans la dynamique comme terme source — un A global l'ignore, donc on
s'attend à une dégradation. La question fine (foresight v1b) : la dégradation est-elle
ADVECTIVE (features déplacées, floutées, en retard de phase) = signature transport ?

Décomposition propre de l'erreur : pour chaque terrain on rapporte le PLANCHER de
représentation (encode-décode du h vrai, ce que la base peut faire) et l'erreur de
ROLLOUT (depuis la CI, via A). L'écart rollout − plancher = erreur d'OPÉRATEUR.

Usage : .venv/bin/python scripts/run_v2_transfer.py
Sorties : outputs/v2/v2_error_growth.png, outputs/v2/v2_rollout_<terrain>.gif,
          docs/v2_V2_transfer.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.pod import fit_pod, encode, decode, stack_height, unstack_height
from src.dmd import fit_dmd, rollout, clip_eigenvalues, spectral_radius
from src.metrics import relative_l2_error, error_growth
from src.io_utils import load_dataset, save_animation

DATA = ROOT / "data" / "v2"
OUT_FIG = ROOT / "outputs" / "v2"
OUT_DOC = ROOT / "docs" / "v2_V2_transfer.md"
ENERGY_THRESHOLD = 0.9999
MAX_MODES = 2000


def evaluate_transfer(train_h_seqs, eval_h_seqs: dict, H: int, W: int,
                      energy_threshold: float, max_modes: int) -> dict:
    """Base POD hauteur + opérateur DMD global écrêté sur train ; pour chaque
    trajectoire d'éval, plancher de représentation et erreur de rollout depuis la CI.
    Retourne k, rayon spectral, et par éval {floor, rollout, rollout_max, gap}."""
    X_train = np.concatenate([stack_height(h) for h in train_h_seqs], axis=1)
    basis = fit_pod(X_train, energy_threshold, max_modes, n_channels=1)
    z_list = [encode(basis, stack_height(h)) for h in train_h_seqs]
    A = clip_eigenvalues(fit_dmd(z_list))

    results = {}
    seqs = {}  # (truth, pred) par éval, pour les figures
    for name, h_seq in eval_h_seqs.items():
        X = stack_height(h_seq)
        z_true = encode(basis, X)
        floor = unstack_height(decode(basis, z_true), H, W)          # plancher repr.
        z_pred = rollout(A, z_true[:, 0], z_true.shape[1] - 1)       # rollout depuis CI
        pred = unstack_height(decode(basis, z_pred), H, W)
        results[name] = {
            "floor": relative_l2_error(floor, h_seq),
            "rollout": relative_l2_error(pred, h_seq),
            "rollout_max": float(error_growth(pred, h_seq).max()),
        }
        results[name]["gap"] = results[name]["rollout"] - results[name]["floor"]
        seqs[name] = (h_seq, pred)
    return {"k": basis.Phi.shape[1], "rho": spectral_radius(A),
            "results": results, "seqs": seqs}


def _load():
    split = json.loads((DATA / "split.json").read_text())
    H, W = split["grid"]["H"], split["grid"]["W"]
    train, eval_set, ref_train = [], {}, None
    for e in split["entries"]:
        for ic in e["ic_ids"]:
            ds = load_dataset(DATA / f"{e['terrain_id']}__{ic}.npz")
            if e["role"] == "train":
                train.append(ds.h)
                if ref_train is None:  # 1re trajectoire train comme référence in-sample
                    ref_train = (f"train_ref ({e['terrain_id']}/{ic})", ds.h)
            else:
                eval_set[e["regime"]] = ds.h
    # référence in-sample (opérateur sur terrain VU) en tête de l'éval
    eval_with_ref = {ref_train[0]: ref_train[1], **eval_set}
    return H, W, train, eval_with_ref


def _render(res: dict, H: int, W: int):
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    # courbes d'erreur par frame (croissance du rollout) — signature temporelle
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, (truth, pred) in res["seqs"].items():
        ax.plot(error_growth(pred, truth), label=name)
    ax.set_xlabel("frame"); ax.set_ylabel("erreur L2 relative (h)")
    ax.set_title(f"V2 — croissance d'erreur du rollout DMD global (k={res['k']}, ρ={res['rho']:.3f})")
    ax.legend(fontsize=8); fig.tight_layout()
    growth_path = OUT_FIG / "v2_error_growth.png"
    fig.savefig(growth_path, dpi=120); plt.close(fig)

    # animations vérité | prédiction | |erreur| pour les terrains holdout
    for name, (truth, pred) in res["seqs"].items():
        if name.startswith("train_ref"):
            continue
        triptych = np.concatenate([truth, pred, np.abs(pred - truth)], axis=2)
        save_animation(OUT_FIG / f"v2_rollout_{name}.gif", triptych, fps=20,
                       cmap="viridis", title=f"V2 {name} — vérité | prédiction | |erreur|")
    return growth_path


def main() -> None:
    H, W, train, eval_set = _load()
    out = evaluate_transfer(train, eval_set, H, W, ENERGY_THRESHOLD, MAX_MODES)
    growth_path = _render(out, H, W)

    print(f"[V2] k={out['k']}  rayon spectral (écrêté) ρ={out['rho']:.4f}")
    print(f"[V2] {'terrain':28s} {'plancher':>9} {'rollout':>9} {'rollout_max':>12} {'gap(opérateur)':>15}")
    for name, m in out["results"].items():
        print(f"[V2] {name:28s} {m['floor']:>9.4f} {m['rollout']:>9.4f} "
              f"{m['rollout_max']:>12.4f} {m['gap']:>15.4f}")

    # lecture automatique : l'opérateur global transfère-t-il sur terrain nouveau ?
    ref = next(m for n, m in out["results"].items() if n.startswith("train_ref"))
    holdouts = {n: m for n, m in out["results"].items() if not n.startswith("train_ref")}
    worst = max(holdouts.values(), key=lambda m: m["rollout"])
    worst_name = max(holdouts, key=lambda n: holdouts[n]["rollout"])
    if worst["rollout"] < 0.10 and worst["gap"] < 2 * ref["rollout"]:
        verdict = ("L'opérateur DMD global TRANSFÈRE : rollout borné sur terrain nouveau, "
                   "écart à la référence in-sample modéré. La baseline statique suffit ; "
                   "V3b (opérateur conditionné) non requis pour ces régimes.")
    else:
        verdict = (f"L'opérateur global ne transfère pas sur le pire holdout "
                   f"({worst_name} rollout={worst['rollout']:.3f} vs plancher "
                   f"{worst['floor']:.3f}, gap {worst['gap']:.3f} ; in-sample "
                   f"{ref['rollout']:.3f}). MAIS ce cliff est sur une topologie "
                   f"HOLDOUT-ONLY (absente du fit DMD). Le test couverture-opérateur "
                   f"(v2b, docs/v2_v2b_operator_coverage.md) montre qu'il est de la "
                   f"COUVERTURE (157%->11.8% en mettant des canaux au fit, gap opérateur "
                   f"alors ~comparable à l'obstacle) -> un A GLOBAL suffit une fois la "
                   f"topologie vue, V3b NON requis ; le résidu est le plancher de "
                   f"représentation (n-width, v1b), pas l'opérateur.")

    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# V2 — Transfert naïf (opérateur DMD global) sur terrain nouveau", "",
             f"Base POD hauteur (k={out['k']}) + opérateur DMD global écrêté "
             f"(ρ={out['rho']:.4f}) ajusté sur tous les terrains d'entraînement, "
             f"déroulé sur les terrains holdout depuis leur CI vraie.", "",
             "Décomposition : plancher = erreur de représentation (encode-décode du h "
             "vrai) ; rollout = erreur depuis la CI via l'opérateur ; gap = rollout − "
             "plancher = erreur imputable à l'OPÉRATEUR. (Note : `fit_dmd` est homogène "
             "mais la POD soustrait une moyenne -> la dynamique centrée est affine ; une "
             "part du gap, même in-sample, est cet angle mort affine, cf. v2b.)", "",
             "| terrain | plancher (repr.) | rollout | rollout_max | gap (opérateur) |",
             "|---|---|---|---|---|"]
    for name, m in out["results"].items():
        lines.append(f"| {name} | {m['floor']:.4f} | {m['rollout']:.4f} | "
                     f"{m['rollout_max']:.4f} | {m['gap']:.4f} |")
    lines += ["", "## Verdict", "", verdict, "",
              "## Signature transport (foresight v1b)", "",
              "DMD est linéaire ; le transport est là où les opérateurs linéaires "
              "souffrent le plus. Si la dégradation est advective (fronts d'onde "
              "déplacés / floutés / déphasés), c'est la même limite n-width que le "
              "résidu canal de V1, mais dans la dynamique. Voir les animations "
              "`outputs/v2/v2_rollout_*.gif` (vérité | prédiction | |erreur|) : l'erreur "
              "se concentre-t-elle sur les fronts en mouvement ?", "",
              "## Couverture-opérateur (v2b)", "",
              "Le cliff canal ci-dessus est mesuré topologie HOLDOUT-ONLY (absente du "
              "fit DMD). v2b (`docs/v2_v2b_operator_coverage.md`) montre qu'il est dominé "
              "par la COUVERTURE : 157 % → 11.8 % en mettant des canaux au fit, gap "
              "opérateur alors comparable à l'obstacle → un A global suffit une fois la "
              "topologie vue, **V3b non requis**. Le résidu (~6 %) est le plancher de "
              "représentation (n-width, v1b), pas l'opérateur.", "",
              f"Figure : `{growth_path.relative_to(ROOT)}`."]
    OUT_DOC.write_text("\n".join(lines) + "\n")
    print(f"[V2] figure -> {growth_path}")
    print(f"[V2] note   -> {OUT_DOC}")
    print(f"[V2] VERDICT : {verdict}")


if __name__ == "__main__":
    main()
