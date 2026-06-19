"""W2 — n-width intra-échantillon : un dam-break BALAYE le domaine → la base POD
linéaire représente-t-elle le front du SOLVEUR VALIDÉ (oracle Ritter, W1) ?

Framing :
- Mesure INTRA-SAMPLE (à la W0) : un seul run contient le front à toutes les positions.
- Solveur 1er ordre → diffuse chaque front → c'est le BARREAU FACILE (front diffusé,
  translatent, comme W0-planaire mais sur la vraie dynamique numérique du solveur).
- Oracle = Ritter dry-bed (position de front validée W1).
- Pas de confound de couverture : le front est partout dans le run.

Gate :
  front_l2 > 0.20 → DÉCISIF (la base linéaire casse même sur le front diffusé)
  front_l2 ≤ 0.20 → HOLD (provisoire — le front 2e ordre MUSCL reste à tester)

Usage : .venv/bin/python scripts/run_w2_dambreak.py
Sorties : outputs/v2.5/w2_dambreak.png, docs/v2.5_W2_dambreak.md
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from config import GridConfig
from src.solver_wetdry import simulate_wetdry
from src.analytic_thacker import front_band_mask
from src.pod import fit_pod, encode, decode, stack_height, unstack_height
from src.metrics import relative_l2_error

# Réutilise la métrique front-localisée de W0 (DRY — ne pas réimplémenter)
from scripts.run_w0_representation import band_l2_error

# ── Configuration ─────────────────────────────────────────────────────────────
GRID = GridConfig(H=64, W=64, dx=4.0 / 64, dy=4.0 / 64)
DRY_EPS = 1e-4
OUT_FIG = ROOT / "outputs" / "v2.5"
OUT_DOC = ROOT / "docs" / "v2.5_W2_dambreak.md"

ENERGY = 0.9999
MAX_MODES = 2000

# Dam-break : eau à gauche (x ≤ x0), lit sec à droite
HL = 0.1       # hauteur initiale côté mouillé (m)
X0 = 0.5      # position du barrage (m), dans un domaine [0, 4]

# t_end choisi pour que le front Ritter atteigne ~3.0 m (intérieur, pas le mur à 4 m)
# Vitesse de front Ritter : x_B(t) = x0 + 2·c·t  avec c = sqrt(g·hl)
# x_B = 0.5 + 2*sqrt(9.81*0.1)*t ≈ 0.5 + 1.981*t
# À t=1.3 → x_B ≈ 0.5 + 2.575 = 3.075  (marge ≈ 1 cellule = 0.0625 avant le mur)
T_END = 1.3


def _cell_centers_x(W: int, dx: float) -> np.ndarray:
    """Coordonnées x des centres de cellules (W,)."""
    return (np.arange(W) + 0.5) * dx


def dambreak_nwidth(H: int = 64, W: int = 64,
                    hl: float = HL, x0: float = X0,
                    t_end: float = T_END,
                    energy_threshold: float = ENERGY,
                    max_modes: int = MAX_MODES) -> dict:
    """Cœur du diagnostic W2 : simule le dam-break, POD h_seq, retourne les métriques.

    Paramètres exposés pour permettre les tests sur grilles réduites.

    Returns
    -------
    dict avec clés : k, global_l2, front_l2, k_sweep (dict k_try→front_l2),
                     h_seq (T,H,W), times (list), basis (PODBasis).
    """
    grid = GridConfig(H=H, W=W, dx=4.0 / W, dy=4.0 / H)
    xs = _cell_centers_x(W, grid.dx)

    # Condition initiale : barrage à x0, lit sec à droite
    h0 = np.where(xs[np.newaxis, :] <= x0, hl, 0.0) * np.ones((H, W))
    hu0 = np.zeros((H, W))
    hv0 = np.zeros((H, W))
    b = np.zeros((H, W))

    # Simulation
    times, h_seq, _hu_seq, _hv_seq = simulate_wetdry(
        h0, hu0, hv0, b, grid, cfl=0.4, t_end=t_end, dry_eps=DRY_EPS
    )

    # POD sur les snapshots de hauteur (n_channels=1)
    X = stack_height(h_seq)
    basis = fit_pod(X, energy_threshold, max_modes, n_channels=1)
    recon = unstack_height(decode(basis, encode(basis, X)), H, W)

    k = basis.Phi.shape[1]
    global_l2 = relative_l2_error(recon, h_seq)
    front_l2 = band_l2_error(recon, h_seq)  # métrique gate W2

    # k-sweep : erreur de front pour k_try modes tronqués
    k_sweep: dict[int, float] = {}
    for k_try in [5, 10, 20, 40]:
        if k_try > k:
            # Moins de modes disponibles : utiliser tous
            k_eff = k
        else:
            k_eff = k_try
        # Tronquer la base aux k_eff premiers modes
        from src.pod import PODBasis
        basis_trunc = PODBasis(
            mean=basis.mean,
            scale=basis.scale,
            Phi=basis.Phi[:, :k_eff],
            singular_values=basis.singular_values,
        )
        recon_t = unstack_height(decode(basis_trunc, encode(basis_trunc, X)), H, W)
        k_sweep[k_try] = band_l2_error(recon_t, h_seq)

    return {
        "k": k,
        "global_l2": global_l2,
        "front_l2": front_l2,
        "k_sweep": k_sweep,
        "h_seq": h_seq,
        "times": times,
        "basis": basis,
        "recon": recon,
        "grid": grid,
    }


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    import math
    # Vérifie que le front Ritter ne touche pas le mur droit à t_end
    import config as _cfg
    c = math.sqrt(_cfg.GRAVITY * HL)
    x_B_tend = X0 + 2.0 * c * T_END
    print(f"[W2] Ritter : c={c:.4f} m/s, x_B(t={T_END}) = {x_B_tend:.4f} m "
          f"(mur droit à 4.0 m, marge = {4.0 - x_B_tend:.4f} m)")
    if x_B_tend >= 3.9:
        print(f"[W2] ATTENTION : le front Ritter approche du mur — envisager de réduire t_end")

    print("[W2] Simulation du dam-break (barrage solveur validé W1)…")
    result = dambreak_nwidth()

    k = result["k"]
    global_l2 = result["global_l2"]
    front_l2 = result["front_l2"]
    k_sweep = result["k_sweep"]
    h_seq = result["h_seq"]
    times = result["times"]
    recon = result["recon"]
    grid = result["grid"]

    print(f"[W2] Simulation : {len(times)} frames, h_seq shape {h_seq.shape}")
    print(f"[W2] k={k}  global_l2={global_l2:.4f}  FRONT_l2={front_l2:.4f}")
    print(f"[W2] k-sweep (front_l2 par nb modes tronqués) :")
    for k_try, fe in sorted(k_sweep.items()):
        print(f"     k_try={k_try:3d} → front_l2={fe:.4f}")
    print(f"[W2] Référence W0 smooth transport : ~7.1% (Thacker planaire_diag, "
          f"front analytique lisse)")

    # ── Figure ────────────────────────────────────────────────────────────────
    xs = _cell_centers_x(GRID.W, GRID.dx)
    iy = GRID.H // 2  # rangée centrale

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Évolution de h(x, y=centre) dans le temps (quelques frames)
    n_frames = h_seq.shape[0]
    frame_indices = [0,
                     n_frames // 4,
                     n_frames // 2,
                     3 * n_frames // 4,
                     n_frames - 1]
    ax = axes[0]
    for fi in frame_indices:
        t = times[fi]
        ax.plot(xs, h_seq[fi, iy, :], label=f"t={t:.2f}s", alpha=0.75)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("h (m)")
    ax.set_title("h(x, y=centre) — dam-break sweep")
    ax.legend(fontsize=7)

    # Erreur de reconstruction (vérité vs POD) — frame finale
    ax2 = axes[1]
    residual = recon[-1, iy, :] - h_seq[-1, iy, :]
    ax2.plot(xs, h_seq[-1, iy, :], "k-", lw=2, label="vérité (t_fin)")
    ax2.plot(xs, recon[-1, iy, :], "r--", lw=1.5, label=f"POD recon k={k}")
    ax2.fill_between(xs, h_seq[-1, iy, :], recon[-1, iy, :], alpha=0.3, color="r",
                     label="erreur")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("h (m)")
    ax2.set_title(f"Reconstruction POD (t_fin)  FRONT={front_l2:.4f}")
    ax2.legend(fontsize=7)

    # k-sweep : décroissance de l'erreur front avec k
    ax3 = axes[2]
    ks_list = sorted(k_sweep.keys())
    fe_list = [k_sweep[kk] for kk in ks_list]
    ax3.semilogy(ks_list, fe_list, "o-", color="#c33", label="front_l2 vs k_try")
    ax3.axhline(0.20, ls="--", color="gray", label="gate 0.20")
    ax3.axhline(0.071, ls=":", color="blue", label="W0 smooth transport ~7.1%")
    ax3.set_xlabel("k (modes tronqués)")
    ax3.set_ylabel("front_l2 (log)")
    ax3.set_title("W2 — Décroissance erreur front vs nb modes")
    ax3.legend(fontsize=7)

    fig.suptitle(f"W2 — Dam-break n-width  k={k}  global={global_l2:.4f}  FRONT={front_l2:.4f}",
                 fontsize=10)
    fig.tight_layout()
    fig_path = OUT_FIG / "w2_dambreak.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"[W2] Figure -> {fig_path}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    W0_SMOOTH_TRANSPORT = 0.071  # 7.1% — W0 Thacker planaire (oracle lisse, analytic)

    if front_l2 > 0.20:
        verdict = (
            f"DÉCISIF : la base linéaire ne représente pas le front du solveur même "
            f"intra-échantillon (capacité) → encodeur justifié. "
            f"(front_l2={front_l2:.4f} > 0.20 ; vs W0 smooth {W0_SMOOTH_TRANSPORT:.3f})"
        )
        verdict_bucket = "DÉCISIF-encodeur"
    else:
        verdict = (
            f"HOLD (provisoire) : la base tient le front 1er-ordre DIFFUSÉ "
            f"(barreau facile). front_l2={front_l2:.4f} ≤ 0.20. "
            f"Le front genuinement net (bore) requiert le solveur 2e ordre MUSCL "
            f"— re-test gaté, prochain pas. PAS les îles. "
            f"(vs W0 smooth transport {W0_SMOOTH_TRANSPORT:.3f})"
        )
        verdict_bucket = "HOLD-provisoire"

    print(f"[W2] VERDICT ({verdict_bucket}) : {verdict}")

    # ── Doc ───────────────────────────────────────────────────────────────────
    k_sweep_rows = "\n".join(
        f"| {kk} | {k_sweep[kk]:.4f} |" for kk in sorted(k_sweep.keys())
    )
    doc_lines = [
        "# W2 — Dam-break n-width intra-échantillon (solveur validé W1)",
        "",
        "**Encoder forcing-test** : la base POD linéaire représente-t-elle le front "
        "mouillé/sec du SOLVEUR (oracle Ritter, W1 validé) ? Mesure INTRA-SAMPLE : "
        "un run dam-break BALAYE le domaine → le front est à toutes les positions "
        "dans un seul run (pas de confound de couverture).",
        "",
        "## Setup",
        "",
        f"- Grille : 64×64, dx=dy={GRID.dx:.5f} m, domaine [0,4]²",
        f"- Dam-break lit sec : hl={HL}, dam à x0={X0} m, t_end={T_END} s",
        f"- Front Ritter à t_end : x_B ≈ {x_B_tend:.4f} m (marge avant mur : {4.0 - x_B_tend:.4f} m)",
        f"- Simulation : {len(times)} frames",
        f"- POD : energy_threshold={ENERGY}, max_modes={MAX_MODES}",
        "",
        "## Résultats",
        "",
        f"| k (modes) | L2 globale | L2 FRONT (gate) |",
        f"|---|---|---|",
        f"| **{k}** | {global_l2:.4f} | **{front_l2:.4f}** |",
        "",
        "### k-sweep (décroissance de l'erreur front)",
        "",
        "| k_try (modes tronqués) | front_l2 |",
        "|---|---|",
        k_sweep_rows,
        "",
        "### Référence W0",
        "",
        f"- W0 Thacker planaire_diag (front analytique LISSE, transport) : ~{W0_SMOOTH_TRANSPORT:.1%}",
        "- Le front Ritter est un touchdown LISSE (h→0 paraboliquement, 1er ordre "
        "diffuse davantage) → c'est le barreau FACILE pour la POD.",
        "",
        "## Framing",
        "",
        "- **Oracle** : Ritter dry-bed (position de front validée W1 ±3 cellules).",
        "- **1er ordre diffuse le front** : chaque interface numérique étale le front "
        "→ front plus large → plus facile pour la POD linéaire (barreau facile).",
        "- **front_l2 > 20%** → la base linéaire NE TIENT PAS même le front diffusé "
        "(décisif, intra-sample, oracle validé, pas de confound) → ENCODEUR JUSTIFIÉ.",
        "- **front_l2 ≤ 20%** → HOLD provisoire : le front diffusé passe, "
        "le front net (bore MUSCL 2e ordre) reste à tester.",
        "",
        "## Verdict",
        "",
        f"**{verdict_bucket}** : {verdict}",
        "",
        f"Figure : `outputs/v2.5/w2_dambreak.png`",
    ]
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.write_text("\n".join(doc_lines) + "\n")
    print(f"[W2] Note -> {OUT_DOC}")


if __name__ == "__main__":
    main()
