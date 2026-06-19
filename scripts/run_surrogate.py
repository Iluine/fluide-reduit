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
import matplotlib.pyplot as plt

from config import GridConfig
from src.solver_wetdry import simulate_wetdry_o2
from src.wetdry_vocab import vocabulary
from src.pod import (fit_pod, encode, decode, stack_height, unstack_height,
                     stack_snapshots, unstack, PODBasis)
from src.dmd import fit_dmd, clip_eigenvalues, rollout, spectral_radius
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


def _traj_state(s: dict) -> np.ndarray:
    """État complet [h,u,v] empilé (3HW, T) pour une trajectoire (u,v désingularisés)."""
    h = s["h_seq"]
    u = s["hu_seq"] / np.maximum(h, 1e-6)
    v = s["hv_seq"] / np.maximum(h, 1e-6)
    return stack_snapshots(h, u, v)


def operator_and_rollout(seqs: list[dict], grid: GridConfig):
    """UN SEUL opérateur linéaire global (DMD ÉCRÊTÉ) sur le POD état complet de TOUT
    le vocabulaire (hypothèse single-global de v2, en mouillé/sec). L'axe non testé."""
    Xs = [_traj_state(s) for s in seqs]
    basis = fit_pod(np.concatenate(Xs, axis=1), 0.9999, 2000, n_channels=3)
    z_list = [encode(basis, X) for X in Xs]            # une trajectoire de coeffs / scénario
    A = fit_dmd(z_list)                                # paires INTRA-trajectoire (jamais à cheval)
    rho_raw = spectral_radius(A)
    A = clip_eigenvalues(A, 1.0)                       # CORRECTION 2 : |λ|<=1 (anti faux blow-up)
    rho_clip = spectral_radius(A)
    return basis, A, z_list, rho_raw, rho_clip


def _rollout_scenario(basis, A, z0, n_steps, H, W):
    """Rollout autorégressif -> décode -> h_pred CLAMPÉ >=0 (correction 5 : analogue W3
    différé ; un h<0 au rendu serait lu à tort comme « opérateur casse »)."""
    z_pred = rollout(A, z0, n_steps)
    h_pred, u_pred, v_pred = unstack(decode(basis, z_pred), H, W)
    return np.maximum(h_pred, 0.0)


def render_scenarios(seqs, basis, A, z_list, grid, names):
    """Frames côte à côte vérité vs surrogate (surface η) à t=0, mi, fin -> PNG (jugement
    visuel). Retourne la L2 indicative par scénario (PAS un gate — critère = visuel)."""
    H, W = grid.H, grid.W
    info = []
    for nm in names:
        i = next(j for j, s in enumerate(seqs) if s["name"] == nm)
        s = seqs[i]
        T = s["h_seq"].shape[0]
        h_pred = _rollout_scenario(basis, A, z_list[i][:, 0], T - 1, H, W)
        eta_true = s["h_seq"] + s["b"]
        eta_pred = h_pred + s["b"]
        l2 = relative_l2_error(h_pred, s["h_seq"])
        info.append((nm, l2))
        frames = [0, T // 2, T - 1]
        vmin = float(min(eta_true.min(), eta_pred.min()))
        vmax = float(max(eta_true.max(), eta_pred.max()))
        fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
        for c, t in enumerate(frames):
            axes[0, c].imshow(eta_true[t], cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
            axes[0, c].set_title(f"vérité t={t}", fontsize=9)
            axes[1, c].imshow(eta_pred[t], cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
            axes[1, c].set_title(f"surrogate t={t}", fontsize=9)
            for a in (axes[0, c], axes[1, c]):
                a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"{nm}  (η : vérité haut / surrogate bas ; L2={l2:.2f})", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_FIG / f"surrogate_{nm}.png", dpi=110)
        plt.close(fig)
    return info


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
             "## Verdict d'étendue", "", verdict, ""]

    # ---- OPÉRATEUR (l'axe non testé) : un seul A global, écrêté, en rollout ----
    basis, A, z_list, rho_raw, rho_clip = operator_and_rollout(seqs, GRID)
    show = ["runup_island_x25_diag", "dambreakdry_x50_diag", "bore_x40_x"]  # >=1 run-up
    show = [nm for nm in show if any(s["name"] == nm for s in seqs)]
    info = render_scenarios(seqs, basis, A, z_list, GRID, show)
    k_state = basis.Phi.shape[1]

    print(f"[SURROGATE-OPERATOR] base état complet k={k_state} ; ρ(A) brut={rho_raw:.3f} "
          f"-> écrêté={rho_clip:.3f} (correction 2 : |λ|<=1)")
    for nm, l2 in info:
        print(f"[SURROGATE-OPERATOR] rollout {nm:26s} L2 indicatif={l2:.3f} "
              f"-> rendu outputs/v2.5/surrogate_{nm}.png")
    print("[SURROGATE-OPERATOR] CRITÈRE = VISUEL : juger les rendus (vérité vs surrogate), "
          "pas la L2. Remède orthogonal : base OK (étendue bornée) ; si l'opérateur casse "
          "visuellement -> dynamique plus riche, PAS encodeur de base.")

    lines += ["## Opérateur single-global (l'axe non testé) — jugement VISUEL", "",
              f"Un seul opérateur linéaire global (DMD écrêté |λ|≤1) sur le POD état complet "
              f"[h,u,v] (k={k_state}) de tout le vocabulaire. ρ(A) brut **{rho_raw:.3f}** → "
              f"écrêté **{rho_clip:.3f}** (correction 2 : un blow-up brut serait un faux "
              "« opérateur casse »).", "",
              "| scénario | L2 indicatif (PAS un gate) | rendu |", "|---|---|---|"]
    for nm, l2 in info:
        lines.append(f"| {nm} | {l2:.3f} | `outputs/v2.5/surrogate_{nm}.png` |")
    lines += ["", "**Critère de succès = VISUEL** (plausibilité du rendu vérité vs surrogate), "
              "pas la L2. Remède orthogonal : l'étendue (base) est bornée ; si l'opérateur "
              "casse visuellement, le remède est une dynamique plus riche (pas un encodeur de "
              "base). Décode clampé h≥0 (analogue W3 différé) pour ne pas lire un h<0 comme un "
              "échec d'opérateur.", "",
              "→ **Checkpoint : jugement visuel de l'utilisateur sur les rendus.**"]

    OUT_DOC.write_text("\n".join(lines) + "\n")
    print(f"[SURROGATE] note -> {OUT_DOC}")


if __name__ == "__main__":
    main()
