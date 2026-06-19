"""W1 — 4 validations analytiques du solveur mouillé/sec (porte d'acceptation de l'oracle).

1. Thacker radial (shoreline mobile) : erreur L2 bornée dans le temps.
2. Ritter dam-break sur lit sec (1D-en-2D) : position du front.
3. C-property × 2 (île émergente + lit incliné mouillé) : vitesse parasite.
4. Positivité (h >= 0 sur toutes les frames des cas 1–3).

Usage : .venv/bin/python scripts/run_w1_validate.py
Sorties : outputs/v2.5/w1_*.png, docs/v2.5_W1_validation.md
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from config import GRAVITY, GridConfig
from src.analytic_thacker import thacker_radial, thacker_radial_period
from src.ritter import ritter_dam_break_dry
from src.solver_wetdry import simulate_wetdry, desingularize_velocity
from src.metrics import relative_l2_error

# ── Grid identique pour toutes les validations ──────────────────────────────
GRID = GridConfig(H=64, W=64, dx=4.0 / 64, dy=4.0 / 64)
DRY_EPS = 1e-4
OUT_FIG = ROOT / "outputs" / "v2.5"
OUT_DOC = ROOT / "docs" / "v2.5_W1_validation.md"

# ── Helpers ──────────────────────────────────────────────────────────────────

def _cell_centers():
    """Coordonnées (xx, yy) des centres de cellules (64×64), domaine [0,4]²."""
    xs = (np.arange(GRID.W) + 0.5) * GRID.dx  # (W,)
    ys = (np.arange(GRID.H) + 0.5) * GRID.dy  # (H,)
    xx, yy = np.meshgrid(xs, ys)               # (H, W)
    return xx, yy


def _max_speed_from_seq(hus, hvs, hs):
    """Vitesse max sur toutes les frames; desingularisation cohérente."""
    # évite division par zéro : utiliser desingularize_velocity frame-by-frame
    speeds = []
    for k in range(hs.shape[0]):
        u = desingularize_velocity(hs[k], hus[k], DRY_EPS)
        v = desingularize_velocity(hs[k], hvs[k], DRY_EPS)
        speeds.append(float(np.max(np.sqrt(u**2 + v**2))))
    return max(speeds)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION 1 : Thacker radial
# ─────────────────────────────────────────────────────────────────────────────

def validate_thacker() -> dict:
    """Thacker radial : shoreline mobile oscillante, 1 période complète."""
    print("[V1-Thacker] initialisation…")
    h0, b = thacker_radial(GRID, t=0.0)
    hu0 = np.zeros_like(h0)
    hv0 = np.zeros_like(h0)

    T_period = thacker_radial_period()  # ≈ 1.41 s (g=9.81, a=1, h0=0.1)
    t_end = T_period                    # exactement 1 période

    print(f"  Période Thacker = {T_period:.4f} s → simulation jusqu'à t={t_end:.4f} s")
    times, h_seq, hu_seq, hv_seq = simulate_wetdry(
        h0, hu0, hv0, b, GRID, cfl=0.4, t_end=t_end, dry_eps=DRY_EPS
    )
    print(f"  Simulation OK : {len(times)} pas, h_seq shape {h_seq.shape}")

    # Aire mouillée initiale (cellules avec h > DRY_EPS)
    wet_start = int(np.sum(h_seq[0] > DRY_EPS))

    # Erreur L2 relative à 8 instants échantillonnés (indices réguliers dans times)
    n_times = len(times)
    sample_indices = [int(round(i * (n_times - 1) / 7)) for i in range(8)]
    l2_errors = []
    for idx in sample_indices:
        t_sim = times[idx]
        h_analytic, _ = thacker_radial(GRID, t=t_sim)
        err = relative_l2_error(h_seq[idx], h_analytic)
        l2_errors.append(err)

    l2_errors_arr = np.array(l2_errors)
    max_l2 = float(l2_errors_arr.max())

    # Aire mouillée à la fin (dernier pas = ≈ 1 période)
    wet_end = int(np.sum(h_seq[-1] > DRY_EPS))

    # Erreur non-monotone ? (oscillation dans l'erreur au cours du temps)
    # Critère : l'erreur ne doit pas croître à chaque pas de la séquence
    diffs = np.diff(l2_errors_arr)
    monotone_increasing = bool(np.all(diffs > 0))

    min_h = float(h_seq.min())

    print(f"  L2 relative à 8 instants : {[f'{e:.4f}' for e in l2_errors_arr]}")
    print(f"  max L2 = {max_l2:.4f}  (seuil < 0.30)")
    print(f"  Erreur monotone croissante = {monotone_increasing} (doit être False)")
    print(f"  min(h) = {min_h:.2e}  (doit être >= 0)")
    print(f"  Aire mouillée : début={wet_start}, fin={wet_end}")

    # ── Assertions ───────────────────────────────────────────────────────────
    assert max_l2 < 0.30, (
        f"ÉCHEC V1-Thacker : max L2 = {max_l2:.4f} >= 0.30 "
        f"(solveur ne suit pas la shoreline)"
    )
    assert not monotone_increasing, (
        f"ÉCHEC V1-Thacker : erreur croît MONOTONEMENT → blowup ou divergence "
        f"(errors={l2_errors_arr})"
    )
    assert min_h >= 0.0, f"ÉCHEC V1-Thacker : min(h) = {min_h:.2e} < 0"

    # ── Figure ───────────────────────────────────────────────────────────────
    h_final_analytic, _ = thacker_radial(GRID, t=times[-1])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    cmin = min(h_seq[-1].min(), h_final_analytic.min())
    cmax = max(h_seq[-1].max(), h_final_analytic.max())
    im0 = axes[0].imshow(h_final_analytic, origin="lower", vmin=cmin, vmax=cmax)
    axes[0].set_title("Analytique (t=T_fin)")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(h_seq[-1], origin="lower", vmin=cmin, vmax=cmax)
    axes[1].set_title("Simulé (t=T_fin)")
    plt.colorbar(im1, ax=axes[1])
    residu = h_seq[-1] - h_final_analytic
    im2 = axes[2].imshow(residu, origin="lower", cmap="RdBu_r",
                          vmax=abs(residu).max(), vmin=-abs(residu).max())
    axes[2].set_title("Résidu simulé − analytique")
    plt.colorbar(im2, ax=axes[2])
    ax_err = axes[0].inset_axes([0.0, -0.45, 3.0, 0.35])
    sample_times = [times[idx] for idx in sample_indices]
    ax_err.plot(sample_times, l2_errors_arr, "o-", color="#c33")
    ax_err.axhline(0.30, ls="--", color="gray", label="seuil 0.30")
    ax_err.set_xlabel("t (s)"); ax_err.set_ylabel("L2 rel.")
    ax_err.set_title("Erreur L2 vs temps (Thacker radial)")
    ax_err.legend()
    fig.suptitle(f"V1-Thacker  max_L2={max_l2:.3f}  wet_start={wet_start}  wet_end={wet_end}")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "w1_thacker.png", dpi=120)
    plt.close(fig)
    print("  [V1-Thacker] PASS")

    return {
        "max_l2": max_l2,
        "l2_errors": l2_errors_arr.tolist(),
        "wet_start": wet_start,
        "wet_end": wet_end,
        "min_h": min_h,
        "h_seq": h_seq,
        "hu_seq": hu_seq,
        "hv_seq": hv_seq,
    }


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION 2 : Ritter dam-break sur lit sec (1D-en-2D)
# ─────────────────────────────────────────────────────────────────────────────

def validate_ritter() -> dict:
    """Ritter dam-break sec, 1D dans la direction x, comparé à la solution analytique."""
    print("[V2-Ritter] initialisation…")
    xx, yy = _cell_centers()

    hl = 0.1
    x0 = 2.0
    t_ritter = 0.5

    b = np.zeros((GRID.H, GRID.W))
    h0 = np.where(xx <= x0, hl, 0.0)
    hu0 = np.zeros_like(h0)
    hv0 = np.zeros_like(h0)

    # Position théorique du front Ritter
    c = np.sqrt(GRAVITY * hl)
    x_B = x0 + 2.0 * t_ritter * c   # ≈ 2.99 m
    x_A = x0 - t_ritter * c          # ≈ 1.51 m (tête de raréfaction)

    print(f"  x_B (front analytique) = {x_B:.4f} m, x_A (tête raréfaction) = {x_A:.4f} m")
    print(f"  Simulation jusqu'à t={t_ritter} s…")

    times, h_seq, hu_seq, hv_seq = simulate_wetdry(
        h0, hu0, hv0, b, GRID, cfl=0.4, t_end=t_ritter, dry_eps=DRY_EPS
    )
    print(f"  Simulation OK : {len(times)} pas")

    # Rangée centrale en y
    iy = GRID.H // 2
    x_centers = (np.arange(GRID.W) + 0.5) * GRID.dx
    h_num = h_seq[-1][iy, :]
    h_analytic = ritter_dam_break_dry(x_centers, t_ritter, hl=hl, x0=x0)

    # Position du front numérique : plus grand x avec h > DRY_EPS
    wet_cells = np.where(h_num > DRY_EPS)[0]
    if len(wet_cells) == 0:
        x_front_num = 0.0
    else:
        x_front_num = x_centers[wet_cells[-1]]

    front_error_m = abs(x_front_num - x_B)
    cell_size = GRID.dx
    front_error_cells = front_error_m / cell_size

    # Plateau amont : cellules avec x < x_A - quelques cellules de marge
    plateau_cells = x_centers < (x_A - 2 * cell_size)
    h_plateau = float(h_num[plateau_cells].mean()) if plateau_cells.any() else float("nan")
    plateau_rel_err = abs(h_plateau - hl) / hl if not np.isnan(h_plateau) else float("inf")

    min_h = float(h_seq.min())

    # Vérification cellules parasites au-delà du front
    beyond_front = x_centers > x_B + cell_size
    spurious_beyond = float(h_num[beyond_front].max()) if beyond_front.any() else 0.0

    print(f"  Front numérique x_front = {x_front_num:.4f} m vs x_B = {x_B:.4f} m "
          f"(écart {front_error_cells:.2f} cellules)")
    print(f"  Plateau amont h_moy = {h_plateau:.5f} m (erreur relative {plateau_rel_err:.3f}, seuil 10%)")
    print(f"  Cellule parasite au-delà du front : max(h) = {spurious_beyond:.2e}")
    print(f"  min(h) = {min_h:.2e}")

    # ── Assertions ───────────────────────────────────────────────────────────
    assert front_error_cells <= 3.0, (
        f"ÉCHEC V2-Ritter : front numérique {x_front_num:.4f} m vs x_B={x_B:.4f} m "
        f"(écart {front_error_cells:.2f} > 3 cellules)"
    )
    assert plateau_rel_err <= 0.10, (
        f"ÉCHEC V2-Ritter : plateau amont {h_plateau:.5f} m écarte de {plateau_rel_err:.3f} "
        f"(>10%) de hl={hl}"
    )
    assert min_h >= 0.0, f"ÉCHEC V2-Ritter : min(h) = {min_h:.2e} < 0"
    assert spurious_beyond < DRY_EPS, (
        f"ÉCHEC V2-Ritter : cellule parasite au-delà du front (h={spurious_beyond:.2e} "
        f">= DRY_EPS={DRY_EPS})"
    )

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_centers, h_analytic, "k-", lw=2, label=f"Ritter analytique (t={t_ritter})")
    ax.plot(x_centers, h_num, "r--", lw=1.5, label="Simulé (rangée centrale y)")
    ax.axvline(x_B, color="gray", ls=":", label=f"x_B={x_B:.3f}")
    ax.axvline(x_A, color="blue", ls=":", label=f"x_A={x_A:.3f}")
    ax.axvline(x_front_num, color="red", ls="--", alpha=0.5, label=f"front num={x_front_num:.3f}")
    ax.set_xlabel("x (m)"); ax.set_ylabel("h (m)")
    ax.set_title(f"V2-Ritter  front_err={front_error_cells:.2f} cells")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "w1_ritter.png", dpi=120)
    plt.close(fig)
    print("  [V2-Ritter] PASS")

    return {
        "x_front_num": x_front_num,
        "x_B": x_B,
        "front_error_cells": front_error_cells,
        "h_plateau": h_plateau,
        "plateau_rel_err": plateau_rel_err,
        "min_h": min_h,
        "h_seq": h_seq,
        "hu_seq": hu_seq,
        "hv_seq": hv_seq,
    }


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION 3 : C-property × 2
# ─────────────────────────────────────────────────────────────────────────────

def validate_c_property() -> dict:
    """Well-balanced : (a) île émergente, (b) lit incliné mouillé."""
    xx, yy = _cell_centers()
    L = 4.0

    # ── (a) île émergente ────────────────────────────────────────────────────
    print("[V3a-Cprop] île émergente…")
    cx, cy = L / 2, L / 2
    r2 = (xx - cx)**2 + (yy - cy)**2
    b_island = 0.6 * np.exp(-r2 / 0.18)   # sommet ≈ 0.6 > eta0=0.4 → émergente
    eta0_island = 0.4
    h0_island = np.maximum(eta0_island - b_island, 0.0)
    hu0 = np.zeros_like(h0_island)
    hv0 = np.zeros_like(h0_island)

    times_a, h_seq_a, hu_seq_a, hv_seq_a = simulate_wetdry(
        h0_island, hu0, hv0, b_island, GRID, cfl=0.4, t_end=2.0, dry_eps=DRY_EPS
    )
    spurious_speed_island = _max_speed_from_seq(hu_seq_a, hv_seq_a, h_seq_a)
    min_h_a = float(h_seq_a.min())
    print(f"  (a) vitesse parasite max = {spurious_speed_island:.2e}  (seuil < 1e-6)")
    print(f"  (a) min(h) = {min_h_a:.2e}")

    # ── (b) lit incliné, entièrement mouillé ─────────────────────────────────
    print("[V3b-Cprop] lit incliné mouillé…")
    b_slope = 0.3 * (xx / L)   # rampe douce de 0 à 0.3 sur [0,4]
    eta0_slope = 1.0            # superficie libre > max(b_slope)=0.3 partout
    h0_slope = eta0_slope - b_slope  # toujours > 0 (min ≈ 0.7)
    hu0_s = np.zeros_like(h0_slope)
    hv0_s = np.zeros_like(h0_slope)

    times_b, h_seq_b, hu_seq_b, hv_seq_b = simulate_wetdry(
        h0_slope, hu0_s, hv0_s, b_slope, GRID, cfl=0.4, t_end=2.0, dry_eps=DRY_EPS
    )
    spurious_speed_slope = _max_speed_from_seq(hu_seq_b, hv_seq_b, h_seq_b)
    min_h_b = float(h_seq_b.min())
    print(f"  (b) vitesse parasite max = {spurious_speed_slope:.2e}  (seuil < 1e-6)")
    print(f"  (b) min(h) = {min_h_b:.2e}")

    # ── Assertions ───────────────────────────────────────────────────────────
    assert spurious_speed_island < 1e-6, (
        f"ÉCHEC V3a-Cprop (île) : vitesse parasite {spurious_speed_island:.2e} >= 1e-6 "
        f"(solveur non well-balanced)"
    )
    assert min_h_a >= 0.0, f"ÉCHEC V3a-Cprop : min(h) = {min_h_a:.2e} < 0"

    assert spurious_speed_slope < 1e-6, (
        f"ÉCHEC V3b-Cprop (lit incliné) : vitesse parasite {spurious_speed_slope:.2e} >= 1e-6 "
        f"(erreur typique dans le terme source d'Audusse sur lit mouillé)"
    )
    assert min_h_b >= 0.0, f"ÉCHEC V3b-Cprop : min(h) = {min_h_b:.2e} < 0"

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # (a) isle
    u_a = desingularize_velocity(h_seq_a[-1], hu_seq_a[-1], DRY_EPS)
    v_a = desingularize_velocity(h_seq_a[-1], hv_seq_a[-1], DRY_EPS)
    speed_a_field = np.sqrt(u_a**2 + v_a**2)
    im = axes[0, 0].imshow(h_seq_a[-1], origin="lower")
    axes[0, 0].set_title("(a) h final — île émergente")
    plt.colorbar(im, ax=axes[0, 0])
    im2 = axes[0, 1].imshow(speed_a_field, origin="lower", cmap="hot")
    axes[0, 1].set_title(f"(a) |u| max={spurious_speed_island:.1e}")
    plt.colorbar(im2, ax=axes[0, 1])
    # (b) slope
    u_b = desingularize_velocity(h_seq_b[-1], hu_seq_b[-1], DRY_EPS)
    v_b = desingularize_velocity(h_seq_b[-1], hv_seq_b[-1], DRY_EPS)
    speed_b_field = np.sqrt(u_b**2 + v_b**2)
    im3 = axes[1, 0].imshow(h_seq_b[-1], origin="lower")
    axes[1, 0].set_title("(b) h final — lit incliné")
    plt.colorbar(im3, ax=axes[1, 0])
    im4 = axes[1, 1].imshow(speed_b_field, origin="lower", cmap="hot")
    axes[1, 1].set_title(f"(b) |u| max={spurious_speed_slope:.1e}")
    plt.colorbar(im4, ax=axes[1, 1])
    fig.suptitle("V3-C-property : vitesse parasite au repos")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "w1_cproperty.png", dpi=120)
    plt.close(fig)
    print("  [V3-Cprop] PASS")

    return {
        "spurious_island": spurious_speed_island,
        "spurious_slope": spurious_speed_slope,
        "min_h_island": min_h_a,
        "min_h_slope": min_h_b,
        "h_seq_a": h_seq_a,
        "hu_seq_a": hu_seq_a,
        "hv_seq_a": hv_seq_a,
        "h_seq_b": h_seq_b,
        "hu_seq_b": hu_seq_b,
        "hv_seq_b": hv_seq_b,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RAPPORT
# ─────────────────────────────────────────────────────────────────────────────

def _write_report(r_thacker, r_ritter, r_cprop):
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# W1 — Validations analytiques du solveur mouillé/sec",
        "",
        "Porte d'acceptation de l'oracle : 4 validations analytiques passées.",
        "",
        "## Grille",
        f"64×64, dx=dy={GRID.dx:.5f} m, domaine [0,4]²",
        "",
        "## V1 — Thacker radial (shoreline mobile)",
        "",
        f"- Période Thacker : T = {thacker_radial_period():.4f} s",
        f"- **max L2 relative (8 instants) = {r_thacker['max_l2']:.4f}** (seuil < 0.30)",
        f"- Erreur non monotone : PASS",
        f"- Aire mouillée : début = {r_thacker['wet_start']}, fin = {r_thacker['wet_end']}",
        f"- min(h) = {r_thacker['min_h']:.2e}",
        f"- L2 par instant : {[f'{e:.4f}' for e in r_thacker['l2_errors']]}",
        "- Statut : **PASS**",
        "",
        "## V2 — Ritter dam-break sur lit sec (1D-en-2D)",
        "",
        f"- hl=0.1, x0=2.0, t=0.5 s",
        f"- x_B analytique = {r_ritter['x_B']:.4f} m",
        f"- **x_front numérique = {r_ritter['x_front_num']:.4f} m** "
        f"(écart {r_ritter['front_error_cells']:.2f} cellules, seuil ≤ 3)",
        f"- Plateau amont h_moy = {r_ritter['h_plateau']:.5f} m "
        f"(erreur relative {r_ritter['plateau_rel_err']:.3f}, seuil ≤ 10%)",
        f"- min(h) = {r_ritter['min_h']:.2e}",
        "- Statut : **PASS**",
        "",
        "## V3 — C-property (well-balanced)",
        "",
        f"- **(a) île émergente** : vitesse parasite max = {r_cprop['spurious_island']:.2e} (seuil < 1e-6)",
        f"- **(b) lit incliné mouillé** : vitesse parasite max = {r_cprop['spurious_slope']:.2e} (seuil < 1e-6)",
        "- Statut : **PASS**",
        "",
        "## V4 — Positivité",
        "",
        f"- min(h) Thacker = {r_thacker['min_h']:.2e}",
        f"- min(h) Ritter = {r_ritter['min_h']:.2e}",
        f"- min(h) C-prop île = {r_cprop['min_h_island']:.2e}",
        f"- min(h) C-prop pente = {r_cprop['min_h_slope']:.2e}",
        "- Statut : **PASS**",
        "",
        "## Verdict",
        "",
        "Les 4 validations analytiques passent. **L'oracle W1 est accepté.**",
        "W2 peut être planifié (POD + mesure du plafond front-localisé sur dynamique réelle).",
        "",
        "Figures : `outputs/v2.5/w1_thacker.png`, `outputs/v2.5/w1_ritter.png`, "
        "`outputs/v2.5/w1_cproperty.png`",
    ]
    OUT_DOC.write_text("\n".join(lines) + "\n")
    print(f"[W1] Note -> {OUT_DOC}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("W1 — Validations analytiques du solveur mouillé/sec")
    print("=" * 60)

    # ── V1 : Thacker ─────────────────────────────────────────────────────────
    r_thacker = validate_thacker()

    # ── V2 : Ritter ──────────────────────────────────────────────────────────
    r_ritter = validate_ritter()

    # ── V3 : C-property ──────────────────────────────────────────────────────
    r_cprop = validate_c_property()

    # ── V4 : Positivité globale (résumé) ─────────────────────────────────────
    print("[V4-Positivité] vérification globale min(h) >= 0…")
    global_min_h = min(
        r_thacker["min_h"],
        r_ritter["min_h"],
        r_cprop["min_h_island"],
        r_cprop["min_h_slope"],
    )
    assert global_min_h >= 0.0, (
        f"ÉCHEC V4-Positivité : min(h) global = {global_min_h:.2e} < 0"
    )
    print(f"  global min(h) = {global_min_h:.2e}  PASS")

    # ── Tableau récapitulatif ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("RÉSUMÉ W1 — PORTE D'ACCEPTATION DE L'ORACLE")
    print("=" * 60)
    print(f"  V1 Thacker max L2          = {r_thacker['max_l2']:.4f}  (< 0.30)  PASS")
    print(f"     aire mouillée (début/fin)= {r_thacker['wet_start']} / {r_thacker['wet_end']}")
    print(f"  V2 Ritter front             = {r_ritter['x_front_num']:.4f} m "
          f"vs x_B={r_ritter['x_B']:.4f} m  "
          f"(±{r_ritter['front_error_cells']:.2f} cells ≤3)  PASS")
    print(f"  V3a C-prop île émergente    = {r_cprop['spurious_island']:.2e}  (< 1e-6)  PASS")
    print(f"  V3b C-prop lit incliné      = {r_cprop['spurious_slope']:.2e}  (< 1e-6)  PASS")
    print(f"  V4 Positivité global min(h) = {global_min_h:.2e}  (>= 0)  PASS")
    print()
    print("  ORACLE W1 : ACCEPTÉ — W2 peut être planifié.")
    print("=" * 60)

    # ── Écriture du doc ───────────────────────────────────────────────────────
    _write_report(r_thacker, r_ritter, r_cprop)


if __name__ == "__main__":
    main()
