"""V1 — Plafond de représentation inter-terrain (le pas décisif).

Base POD de HAUTEUR SEULE construite sur les terrains d'entraînement, puis
encode-décode du h VRAI des terrains holdout (interp + extrap). Aucune dynamique.
Tranche : la base statique span-t-elle un terrain nouveau ? Plafond bas (<~10%) ->
conditionner la dynamique (V2/V3b). Plafond haut (>~30%, surtout extrap) ->
conditionner la représentation (V3a, voire encodeur V5).

Usage : .venv/bin/python scripts/run_v1_representation.py
Sorties : outputs/v2/v1_representation_ceiling.png, docs/v2_V1_representation_ceiling.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.pod import (fit_pod, encode, decode, stack_height, unstack_height,
                     cumulative_energy)
from src.metrics import relative_l2_error, error_growth
from src.io_utils import load_dataset

DATA = ROOT / "data" / "v2"
OUT_FIG = ROOT / "outputs" / "v2"
OUT_DOC = ROOT / "docs" / "v2_V1_representation_ceiling.md"
ENERGY_THRESHOLD = 0.9999
MAX_MODES = 2000  # cap volontairement élevé : laisser le seuil d'énergie gouverner k


def representation_ceiling(train_h_seqs, holdout_h_seqs, H: int, W: int,
                           energy_threshold: float, max_modes: int) -> dict:
    """Construit la base POD hauteur sur les trajectoires train, encode-décode chaque
    holdout. Retourne k, énergie, erreur train (plancher) et erreur par régime."""
    X_train = np.concatenate([stack_height(s) for s in train_h_seqs], axis=1)
    basis = fit_pod(X_train, energy_threshold, max_modes, n_channels=1)
    k = basis.Phi.shape[1]
    energy = cumulative_energy(basis.singular_values)
    energy_at_k = float(energy[k - 1])

    # plancher in-sample : reconstruction des données d'entraînement
    train_err = relative_l2_error(decode(basis, encode(basis, X_train)), X_train)

    regimes = {}
    for regime, h_seq in holdout_h_seqs.items():
        X = stack_height(h_seq)
        Xr = decode(basis, encode(basis, X))
        seq_r = unstack_height(Xr, H, W)
        regimes[regime] = {
            "err": relative_l2_error(seq_r, h_seq),
            "err_max": float(error_growth(seq_r, h_seq).max()),
        }
    return {"k": k, "energy_at_k": energy_at_k, "train_err": float(train_err),
            "regimes": regimes}


def _load_split():
    split = json.loads((DATA / "split.json").read_text())
    H, W = split["grid"]["H"], split["grid"]["W"]
    train, holdout = [], {}
    for e in split["entries"]:
        for ic in e["ic_ids"]:
            ds = load_dataset(DATA / f"{e['terrain_id']}__{ic}.npz")
            if e["role"] == "train":
                train.append(ds.h)
            else:
                holdout[e["regime"]] = ds.h  # un holdout par régime (1 CI chacun)
    return H, W, train, holdout


def _render_figure(res: dict, H: int, W: int, holdout_h: dict, basis_seqs: dict):
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    regimes = list(res["regimes"].keys())
    fig = plt.figure(figsize=(4 + 3 * len(regimes), 6))
    # barres : plancher train + un bar par régime
    ax = fig.add_subplot(2, 1, 1)
    labels = ["train (réf)"] + regimes
    vals = [res["train_err"]] + [res["regimes"][r]["err"] for r in regimes]
    ax.bar(labels, vals, color=["#888"] + ["#3a7"] * len(regimes))
    ax.set_ylabel("erreur L2 relative (h)")
    ax.set_title(f"V1 — plafond de représentation hauteur (k={res['k']}, "
                 f"énergie={res['energy_at_k']:.5f})")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    # champs résiduels : dernière frame, un régime extrap représentatif
    show = [r for r in regimes if r.startswith("extrap")][:2] or regimes[:1]
    for j, r in enumerate(show):
        true = holdout_h[r][-1]
        rec = basis_seqs[r][-1]
        for col, (field, ttl) in enumerate(
                [(true, "vérité"), (rec, "reconstruit"), (np.abs(rec - true), "|résidu|")]):
            a = fig.add_subplot(2, len(show) * 3, len(show) * 3 + j * 3 + col + 1)
            a.imshow(field, origin="lower", cmap="viridis")
            a.set_title(f"{r}\n{ttl}", fontsize=8); a.axis("off")
    fig.tight_layout()
    out = OUT_FIG / "v1_representation_ceiling.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def main() -> None:
    H, W, train, holdout = _load_split()
    res = representation_ceiling(train, holdout, H, W, ENERGY_THRESHOLD, MAX_MODES)

    # reconstruire les séquences holdout pour la figure (résidus)
    X_train = np.concatenate([stack_height(s) for s in train], axis=1)
    basis = fit_pod(X_train, ENERGY_THRESHOLD, MAX_MODES, n_channels=1)
    basis_seqs = {r: unstack_height(decode(basis, encode(basis, stack_height(h))), H, W)
                  for r, h in holdout.items()}
    fig_path = _render_figure(res, H, W, holdout, basis_seqs)

    print(f"[V1] k={res['k']} (énergie {res['energy_at_k']:.6f}), "
          f"erreur train (plancher) = {res['train_err']:.4f}")
    for r, m in res["regimes"].items():
        print(f"[V1] {r:18s} err={m['err']:.4f}  err_max={m['err_max']:.4f}")

    # verdict automatique sur le seuil de représentation
    extrap_errs = [m["err"] for r, m in res["regimes"].items() if r.startswith("extrap")]
    worst_extrap = max(extrap_errs) if extrap_errs else 0.0
    ch = res["regimes"].get("extrap_channel", {}).get("err")  # signal porteur (topo neuve)
    if worst_extrap < 0.10:
        verdict = ("Plafond BAS (<10% même en extrapolation) : la base statique span "
                   "le terrain nouveau dans le régime submergé. Défaut probable = la "
                   "DYNAMIQUE -> conditionner l'opérateur (V2 baseline puis V3b).")
    elif worst_extrap < 0.30:
        verdict = ("Plafond INTERMÉDIAIRE (10–30% en extrapolation) : la base tient "
                   "l'interpolation mais peine en extrapolation -> évaluer V2, puis "
                   "conditionner la représentation (V3a : b en canal) si nécessaire.")
    else:
        verdict = ("Plafond HAUT (>30% en extrapolation). NE PAS conclure tout de suite "
                   "'base fondamentalement terrain-spécifique -> encodeur appris'. Avec "
                   "~9 terrains la famille est sous-échantillonnée (cf. plafond vitesse "
                   "du POC, en partie data-limité). CHECK BON MARCHÉ D'ABORD : le plafond "
                   "baisse-t-il en ajoutant des terrains d'entraînement ? S'il baisse -> "
                   "data-limité (ajouter des terrains, garder la base statique). S'il "
                   "tient -> structurel (alors V3a / encodeur V5 justifié).")

    # corollaire : extrap_channel (topologie jamais vue) donne ses dents au diagnostic ;
    # interp/extrap_obstacle restent du régime de réfraction douce.
    if ch is not None:
        verdict += (f" Signal porteur : extrap_channel (topologie jamais vue) = {ch:.4f}"
                    f" ; ne pas conclure 'ça généralise' sur les seuls cas de réfraction"
                    f" douce (interp/extrap_obstacle).")

    # portée du résultat (toujours écrite, pour calibrer le ✅)
    scope = ("Tous les terrains sont SUBMERGÉS : la dépendance au terrain passe par la "
             "réfraction (contraste de célérité ~1.7×, réel mais modéré). Un V1 qui "
             "passe certifie la généralisation DANS le régime submergé/réfraction — pas "
             "sur tout terrain de jeu. Le sec / les îles (sillages, séparation) sont un "
             "régime distinct plus dur, reporté en v2.5 (solveur mouillé/sec "
             "positivity-preserving & well-balanced).")

    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# V1 — Plafond de représentation inter-terrain (hauteur)", "",
             f"Base POD hauteur seule (n_channels=1), seuil d'énergie {ENERGY_THRESHOLD}.",
             "",
             f"- **k = {res['k']}** (énergie cumulée {res['energy_at_k']:.6f}). "
             f"Pour mémoire, le POC mono-terrain donnait k≈43 ; un k plus grand ici "
             f"est attendu et mesure la complexité accrue de la famille.",
             f"- Erreur de reconstruction train (plancher in-sample) : "
             f"{res['train_err']:.4f}.", "",
             "## Plafond par régime (erreur L2 relative de h)", "",
             "| régime | err | err_max (par frame) |", "|---|---|---|"]
    for r, m in res["regimes"].items():
        lines.append(f"| {r} | {m['err']:.4f} | {m['err_max']:.4f} |")
    lines += ["", "## Verdict", "", verdict, "", "## Portée du résultat (calibrage du ✅)",
              "", scope, "", f"Figure : `{fig_path.relative_to(ROOT)}`."]
    OUT_DOC.write_text("\n".join(lines))
    print(f"[V1] figure -> {fig_path}")
    print(f"[V1] note   -> {OUT_DOC}")
    print(f"[V1] VERDICT : {verdict}")
    print(f"[V1] PORTÉE  : {scope}")


if __name__ == "__main__":
    main()
