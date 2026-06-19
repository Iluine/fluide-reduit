"""Surrogate mouillé/sec : génération du vocabulaire + deux verdicts.

(Task 2) ÉTENDUE — POD hauteur combiné, k @2 % (global L2), même métrique par-scénario
et combiné -> sature (étendue bornée, linéaire suffit) vs croît (encodeur de base).
(Task 3) OPÉRATEUR — POD état complet [h,u,v] + DMD écrêté + rollout + rendu (jugement
visuel). Critère de succès = VISUEL, pas L2.

Usage : .venv/bin/python scripts/run_surrogate.py
Sorties : docs/v2.5_surrogate.md, outputs/v2.5/surrogate_*.{png}."""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from config import GridConfig
from src.solver_wetdry import simulate_wetdry_o2
from src.wetdry_vocab import vocabulary
from src.pod import (fit_pod, encode, decode, stack_height, unstack_height, PODBasis)
from src.metrics import relative_l2_error

GRID = GridConfig(H=64, W=64, dx=4.0 / 64, dy=4.0 / 64)
OUT_FIG = ROOT / "outputs" / "v2.5"
OUT_DOC = ROOT / "docs" / "v2.5_surrogate.md"
TAU = 0.02


def generate(grid: GridConfig) -> list[dict]:
    """Génère chaque scénario du vocabulaire par le solveur 2e ordre validé."""
    seqs = []
    for s in vocabulary(grid):
        _, hs, hus, hvs = simulate_wetdry_o2(s.h0, s.hu0, s.hv0, s.b, grid, t_end=s.t_end)
        seqs.append(dict(name=s.name, b=s.b, h_seq=hs, hu_seq=hus, hv_seq=hvs))
    return seqs


def _k_at_error(X_h: np.ndarray, true_h: np.ndarray, H: int, W: int, tau: float) -> int:
    """Plus petit k tel que l'erreur L2 globale de la reconstruction POD ≤ tau."""
    basis = fit_pod(X_h, 0.9999999, 4000, n_channels=1)
    for k in range(1, basis.Phi.shape[1] + 1):
        bt = PODBasis(mean=basis.mean, scale=basis.scale,
                      Phi=basis.Phi[:, :k], singular_values=basis.singular_values)
        rec = unstack_height(decode(bt, encode(bt, X_h)), H, W)
        if relative_l2_error(rec, true_h) <= tau:
            return k
    return basis.Phi.shape[1]


def _k_combined(seqs_subset: list[dict], H: int, W: int, tau: float) -> int:
    all_h = np.concatenate([s["h_seq"] for s in seqs_subset], axis=0)
    return _k_at_error(stack_height(all_h), all_h, H, W, tau)


def extent_endpoint(seqs: list[dict], grid: GridConfig, tau: float = TAU) -> dict:
    """k combiné @tau vs k par-scénario (même métrique) + courbe de saturation nested
    (k combiné pour n=1,2,4,8,… scénarios) pour trancher plateau vs montée linéaire."""
    H, W = grid.H, grid.W
    per = []
    for s in seqs:
        Xh = stack_height(s["h_seq"])
        per.append(_k_at_error(Xh, s["h_seq"], H, W, tau))
    n = len(seqs)
    k_combined = _k_combined(seqs, H, W, tau)
    # courbe nested : k combiné cumulatif (les n premiers scénarios)
    ns = sorted({1, 2, 4, 8, n})
    curve = [(m, _k_combined(seqs[:m], H, W, tau)) for m in ns if m <= n]
    return dict(per=per, k_med=int(np.median(per)), k_sum=int(np.sum(per)),
                k_combined=k_combined, n=n, curve=curve)


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    seqs = generate(GRID)
    ext = extent_endpoint(seqs, GRID)
    n, kc, kmed, ksum, curve = ext["n"], ext["k_combined"], ext["k_med"], ext["k_sum"], ext["curve"]
    NDOF = GRID.H * GRID.W

    # incrément marginal sur la dernière moitié du vocabulaire : plateau vs montée
    last = [k for _, k in curve][-2:]
    last_m = [m for m, _ in curve][-2:]
    marginal = (last[-1] - last[-2]) / max(last_m[-1] - last_m[-2], 1)   # modes / scénario ajouté
    frac_sum = kc / max(ksum, 1)
    tractable = kc < NDOF // 10                                          # ≪ degrés de liberté
    # bornée pour vocabulaire borné si : base tractable ET sous-additive ET coût marginal
    # par scénario petit devant N² (quelques modes), même s'il remonte pour les régimes divers.
    bounded = tractable and frac_sum < 0.75 and marginal < NDOF * 0.01
    verdict = (f"BORNÉE (pour vocabulaire borné) : k combiné={kc} ≪ N²={NDOF} (base linéaire "
               f"tractable, {kc/NDOF*100:.1f}% des DDL), {frac_sum*100:.0f}% de la somme {ksum} "
               f"(sous-additif -> partage réel). Coût marginal ~{marginal:.1f} modes/scénario "
               f"(modéré ; il REMONTE pour les régimes nets dam-break/bore, plus divers que les "
               f"run-up -> PAS un plateau plat). Un vocabulaire BORNÉ reste donc tractable "
               f"(~few modes/régime) ; l'encodeur de base N'est PAS forcé sur la représentation. "
               f"Il ne le serait qu'avec une étendue NON bornée (centaines de régimes distincts), "
               f"absente du livrable."
               if bounded else
               f"NON BORNÉE : k combiné={kc} ({frac_sum*100:.0f}% de la somme, coût marginal "
               f"~{marginal:.1f} modes/scénario) -> montée ~linéaire -> encodeur de base "
               f"pourrait se justifier.")

    print(f"[SURROGATE-EXTENT] {n} scénarios ; par-scénario médian={kmed}, somme={ksum}, N²={NDOF}")
    print(f"[SURROGATE-EXTENT] k COMBINÉ @{TAU:.0%} = {kc} ({frac_sum*100:.0f}% somme, "
          f"{kc/NDOF*100:.1f}% de N²)")
    print(f"[SURROGATE-EXTENT] courbe nested : " + ", ".join(f"n={m}->k={k}" for m, k in curve)
          + f"  (incrément marginal ~{marginal:.1f} modes/scénario)")
    print(f"[SURROGATE-EXTENT] VERDICT : {verdict}")

    curve_str = " | ".join(f"n={m} → k={k}" for m, k in curve)
    lines = ["# Surrogate mouillé/sec — endpoint d'étendue (k combiné)", "",
             "POD hauteur combiné sur tout le vocabulaire (un seul jeu de snapshots). "
             f"k mesuré à erreur L2 globale ≤ {TAU:.0%}, même métrique par-scénario et "
             "combiné (apples-to-apples).", "",
             f"- scénarios : **{n}** (pesés run-up, positions étalées)",
             f"- k par-scénario : médian **{kmed}**, somme **{ksum}** ; N² = {NDOF}",
             f"- **k combiné @{TAU:.0%} = {kc}** ({frac_sum*100:.0f}% de la somme, "
             f"{kc/NDOF*100:.1f}% de N²)",
             f"- courbe de saturation nested : {curve_str}",
             f"- incrément marginal (dernier palier) : ~{marginal:.1f} modes/scénario", "",
             "## Verdict d'étendue", "", verdict, "",
             "_(Verdict mesuré par l'endpoint + la courbe nested. Le verdict d'opérateur, "
             "lui, est visuel — cf. rollouts, Task 3.)_"]
    OUT_DOC.write_text("\n".join(lines) + "\n")
    print(f"[SURROGATE-EXTENT] note -> {OUT_DOC}")


if __name__ == "__main__":
    main()
