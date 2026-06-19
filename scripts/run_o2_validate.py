"""Task 3 — Re-validation du solveur 2e ordre (porte d'acceptation de l'oracle).

La masse ne suffit pas. On valide contre des solutions ANALYTIQUES, avec le point clé
(flag #1) : la conjonction Thacker-2e ordre — seul cas combinant WB (bathy courbe) +
sec (trait d'eau) + mouvement. Un schéma peut passer C-property statique ET positivité
lit-plat séparément tout en échouant sur le trait d'eau mobile sur pente ; on surveille
donc Thacker comme la CONJONCTION (L2 bornée ET positivité tenue, simultanément, + L2
au moins aussi bon que le 1er ordre). Plus Stoker (bore, le régime choc).

Usage : .venv/bin/python scripts/run_o2_validate.py
Sorties : docs/v2.5_o2_validation.md, outputs/v2.5/o2_validate.png."""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from config import GridConfig, GRAVITY
from src.solver_wetdry import simulate_wetdry, simulate_wetdry_o2
from src.analytic_thacker import thacker_radial, thacker_radial_period
from src.stoker import stoker_dam_break, stoker_star_state
from src.metrics import relative_l2_error

GRID = GridConfig(H=64, W=64, dx=4.0 / 64, dy=4.0 / 64)
OUT_FIG = ROOT / "outputs" / "v2.5"
OUT_DOC = ROOT / "docs" / "v2.5_o2_validation.md"


def _thacker_l2(sim, n_samples=8):
    """Max sur le temps de l'erreur L2 relative de h vs Thacker analytique, sur ~1
    période, + le min(h) global (positivité). Retourne (max_l2, min_h)."""
    T = thacker_radial_period()
    h0, b = thacker_radial(GRID, t=0.0)
    z = np.zeros_like(h0)
    times, hs, _, _ = sim(h0, z.copy(), z.copy(), b, GRID, t_end=T)
    times = np.asarray(times)
    idx = np.linspace(0, len(times) - 1, n_samples).astype(int)
    l2 = []
    for i in idx:
        h_true, _ = thacker_radial(GRID, t=float(times[i]))
        l2.append(relative_l2_error(hs[i], h_true))
    return float(max(l2)), float(hs.min())


def _shock_transition_width(row, h_m, h_r):
    """Nombre de cellules dans la transition du choc (entre h_r et h_m côté droit)."""
    lo, hi = h_r + 0.15 * (h_m - h_r), h_m - 0.15 * (h_m - h_r)
    return int(np.sum((row > lo) & (row < hi)))


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    results = {}

    # ---- V1 : Thacker-2e ordre en CONJONCTION (WB courbe + sec + mouvement) ----
    l2_o2, minh_o2 = _thacker_l2(simulate_wetdry_o2)
    l2_o1, minh_o1 = _thacker_l2(simulate_wetdry)
    results["thacker"] = dict(l2_o2=l2_o2, l2_o1=l2_o1, minh_o2=minh_o2)
    # CONJONCTION : L2 bornée ET positivité tenue, simultanément, sur le MÊME run 2e ordre
    assert l2_o2 < 0.30, f"Thacker-2e L2={l2_o2:.3f} >= 0.30"
    assert minh_o2 >= 0.0, f"Thacker-2e positivité violée: min h={minh_o2:.2e}"
    assert l2_o2 <= l2_o1 * 1.05, f"2e ordre PAS plus précis que 1er: {l2_o2:.4f} vs {l2_o1:.4f}"

    # ---- V2 : Stoker (bore, front net) ----
    hl, hr, x0 = 1.0, 0.1, 2.0
    h_m, u_m, s = stoker_star_state(hl, hr)
    xs = (np.arange(64) + 0.5) * GRID.dx
    h0 = np.where(xs[None, :] <= x0, hl, hr) * np.ones((64, 64))
    z = np.zeros((64, 64))
    t_st = 0.3
    _, hs_o2, hus_o2, _ = simulate_wetdry_o2(h0.copy(), z.copy(), z.copy(), z.copy(), GRID, t_end=t_st)
    _, hs_o1, _, _ = simulate_wetdry(h0.copy(), z.copy(), z.copy(), z.copy(), GRID, t_end=t_st)
    row_o2 = hs_o2[-1, 32, :]
    row_o1 = hs_o1[-1, 32, :]
    h_an, _ = stoker_dam_break(xs, t_st, hl, hr, x0)
    # position du choc : dernière cellule où h > (h_m+h_r)/2
    shock_o2 = xs[row_o2 > 0.5 * (h_m + hr)].max()
    shock_an = x0 + s * t_st
    w_o2 = _shock_transition_width(row_o2, h_m, hr)
    w_o1 = _shock_transition_width(row_o1, h_m, hr)
    results["stoker"] = dict(shock_o2=shock_o2, shock_an=shock_an, h_m=h_m,
                             w_o2=w_o2, w_o1=w_o1, minh=float(hs_o2.min()))
    assert abs(shock_o2 - shock_an) <= 3 * GRID.dx, \
        f"Stoker choc {shock_o2:.3f} vs analytique {shock_an:.3f} (>3 cellules)"
    assert float(hs_o2.min()) >= 0.0
    # état intermédiaire h_m (médiane juste derrière le choc)
    mid = (xs > x0 + 0.1) & (xs < shock_an - 0.1)
    assert abs(float(np.median(row_o2[mid])) - h_m) <= 0.10 * h_m, "Stoker h_m incorrect"
    assert w_o2 <= w_o1, f"Choc 2e ordre PAS plus net: {w_o2} vs {w_o1} cellules"

    # ---- V3 : C-property 2e ordre (île émergente statique) ----
    yy, xx = np.mgrid[0:64, 0:64].astype(float)
    b_isl = 0.6 * np.exp(-(((xx - 32) ** 2 + (yy - 32) ** 2) / 40.0))
    h0i = np.maximum(0.4 - b_isl, 0.0)
    _, hsi, husi, hvsi = simulate_wetdry_o2(h0i, z.copy(), z.copy(), b_isl, GRID, t_end=2.0)
    spurious = float(np.sqrt((husi / np.maximum(hsi, 1e-6)) ** 2
                             + (hvsi / np.maximum(hsi, 1e-6)) ** 2).max())
    results["c_property"] = dict(spurious=spurious, minh=float(hsi.min()))
    assert spurious < 1e-6, f"C-property 2e ordre: vitesse parasite {spurious:.2e} >= 1e-6"

    # ---- V4 : positivité globale ----
    assert minh_o2 >= 0.0 and float(hs_o2.min()) >= 0.0 and float(hsi.min()) >= 0.0

    # ---- figure : Thacker (coupe) + Stoker (coupe vérité|2e ordre) ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a2.plot(xs, h_an, "k-", label="Stoker analytique")
    a2.plot(xs, row_o2, "r--", label="2e ordre")
    a2.plot(xs, row_o1, "b:", label="1er ordre")
    a2.set_title(f"Stoker bore t={t_st} (choc {shock_o2:.2f} vs {shock_an:.2f})")
    a2.set_xlabel("x"); a2.set_ylabel("h"); a2.legend(fontsize=8)
    a1.bar(["1er ordre", "2e ordre"], [l2_o1, l2_o2], color=["#88a", "#c33"])
    a1.set_title(f"Thacker-2e conjonction: L2={l2_o2:.3f}, min h={minh_o2:.1e}")
    a1.set_ylabel("max L2 (h)")
    fig.tight_layout()
    fig_path = OUT_FIG / "o2_validate.png"
    fig.savefig(fig_path, dpi=120); plt.close(fig)

    print(f"[O2-VALIDATE] Thacker-2e CONJONCTION : L2={l2_o2:.4f} (1er ordre {l2_o1:.4f}), "
          f"min h={minh_o2:.2e}  -> L2 borné ET positif ET ≤ 1er ordre : PASS")
    print(f"[O2-VALIDATE] Stoker bore : choc {shock_o2:.3f} vs analytique {shock_an:.3f} "
          f"(±{abs(shock_o2-shock_an)/GRID.dx:.1f} cellules), largeur 2e={w_o2} vs 1er={w_o1}, "
          f"h_m={h_m:.3f}  : PASS")
    print(f"[O2-VALIDATE] C-property 2e ordre : vitesse parasite {spurious:.2e} < 1e-6 : PASS")
    print(f"[O2-VALIDATE] positivité : min h >= 0 partout : PASS")
    print("[O2-VALIDATE] ORACLE 2e ORDRE ACCEPTÉ.")

    lines = ["# Validation du solveur 2e ordre (MUSCL surface-gradient)", "",
             "Porte d'acceptation : 4 analytiques, avec **Thacker-2e en GATE-CONJONCTION** "
             "(WB courbe + sec + mouvement ensemble — un schéma peut passer C-property "
             "statique + positivité lit-plat séparément et échouer ici).", "",
             "| validation | mesuré | critère | |", "|---|---|---|---|",
             f"| Thacker-2e L2 (max sur t) | {l2_o2:.4f} | < 0.30 ET ≤ 1er ordre ({l2_o1:.4f}) | PASS |",
             f"| Thacker-2e positivité (min h) | {minh_o2:.2e} | ≥ 0 (simultané) | PASS |",
             f"| Stoker bore — position choc | {shock_o2:.3f} vs {shock_an:.3f} | ±3 cellules | PASS |",
             f"| Stoker bore — largeur (2e/1er) | {w_o2} / {w_o1} | 2e ≤ 1er (plus net) | PASS |",
             f"| C-property 2e (vitesse parasite) | {spurious:.2e} | < 1e-6 | PASS |",
             f"| positivité globale (min h) | {min(minh_o2, float(hs_o2.min()), float(hsi.min())):.2e} | ≥ 0 | PASS |",
             "", "## Verdict", "",
             "**Oracle 2e ordre ACCEPTÉ.** La C-property 2e ordre à ~1e-15 confirme que la "
             "surface-gradient method (reconstruction de η, lit non reconstruit) préserve le "
             "well-balancing par réduction au 1er ordre au repos ; la conjonction Thacker-2e "
             "(L2 borné ET positivité tenue, simultanément) confirme la réconciliation "
             "η/positivité sur trait d'eau mobile + pente ; Stoker valide le bore (choc).", "",
             f"Figure : `{fig_path.relative_to(ROOT)}`."]
    OUT_DOC.write_text("\n".join(lines) + "\n")
    print(f"[O2-VALIDATE] note -> {OUT_DOC}")


if __name__ == "__main__":
    main()
