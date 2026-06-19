"""W0 — Test forçant n-width : une base POD LINÉAIRE représente-t-elle un trait d'eau
MOBILE ? Oracle analytique (Thacker, sans solveur). Signal PARALLÈLE qui dit quel réduit
aval (DMD linéaire vs encodeur) accoler à l'oracle mouillé/sec.

Raffinement clé : le GATE se lit sur l'erreur FRONT-LOCALISÉE, pas la L2 globale (le bulk
lisse axisymétrique est bas-rang et masquerait la difficulté du front). Deux oracles :
radial (front courbe qui respire = forme) et planaire diagonal (front qui translate =
transport). Les faire tourner désambiguïse forme vs transport.

Usage : .venv/bin/python scripts/run_w0_representation.py
Sorties : outputs/v2.5/w0_nwidth.png, docs/v2.5_W0_nwidth.md."""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from config import GridConfig
from src.analytic_thacker import (thacker_radial, thacker_planar_diag,
                                  thacker_radial_period, thacker_planar_period,
                                  front_band_mask)
from src.pod import fit_pod, encode, decode, stack_height, unstack_height
from src.metrics import relative_l2_error

GRID = GridConfig(H=64, W=64, dx=4.0 / 64, dy=4.0 / 64)
OUT_FIG = ROOT / "outputs" / "v2.5"
OUT_DOC = ROOT / "docs" / "v2.5_W0_nwidth.md"
N_FRAMES = 80
ENERGY, MAX_MODES = 0.9999, 2000


def band_l2_error(recon_seq, true_seq, eps: float = 1e-3, width: int = 2) -> float:
    """Erreur L2 relative restreinte aux cellules de bande-front (par frame, agrégée)."""
    num = den = 0.0
    for rec, tru in zip(recon_seq, true_seq):
        band = front_band_mask(tru, eps, width)
        if band.any():
            num += float(np.sum((rec[band] - tru[band]) ** 2))
            den += float(np.sum(tru[band] ** 2))
    return float(np.sqrt(num / (den + 1e-12)))


def nwidth_ceiling(true_seq, H: int, W: int, energy_threshold: float = ENERGY,
                   max_modes: int = MAX_MODES) -> dict:
    """POD hauteur (n_channels=1) sur la séquence ; erreur de reconstruction globale ET
    front-localisée (le gate se lit sur band_err)."""
    X = stack_height(true_seq)
    basis = fit_pod(X, energy_threshold, max_modes, n_channels=1)
    recon = unstack_height(decode(basis, encode(basis, X)), H, W)
    return {"k": basis.Phi.shape[1],
            "global_err": relative_l2_error(recon, true_seq),
            "band_err": band_l2_error(recon, true_seq)}


def _seq(kind: str):
    if kind == "radial":
        T = thacker_radial_period()
        return np.stack([thacker_radial(GRID, t=ph * T)[0]
                         for ph in np.linspace(0, 1, N_FRAMES)])
    T = thacker_planar_period()
    return np.stack([thacker_planar_diag(GRID, t=ph * T)[0]
                     for ph in np.linspace(0, 1, N_FRAMES)])


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    results = {}
    for kind in ("radial", "planar_diag"):
        seq = _seq(kind)
        results[kind] = nwidth_ceiling(seq, 64, 64)
        m = results[kind]
        print(f"[W0] {kind:12s} k={m['k']:4d}  global={m['global_err']:.4f}  "
              f"FRONT={m['band_err']:.4f}")

    # figure : barres global vs front, par oracle
    fig, ax = plt.subplots(figsize=(7, 4))
    labels, glob, band = list(results), [], []
    for kk in labels:
        glob.append(results[kk]["global_err"]); band.append(results[kk]["band_err"])
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w / 2, glob, w, label="L2 globale (trompeuse)")
    ax.bar(x + w / 2, band, w, label="L2 FRONT-localisée (gate)", color="#c33")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("erreur L2 relative (h)")
    ax.set_title("W0 — plafond n-width : front mobile vs base POD linéaire")
    ax.legend(); fig.tight_layout()
    fig_path = OUT_FIG / "w0_nwidth.png"
    fig.savefig(fig_path, dpi=120); plt.close(fig)

    worst_band = max(results[k]["band_err"] for k in results)
    if worst_band > 0.20:
        verdict = (f"La base POD linéaire NE TIENT PAS le trait d'eau mobile (erreur "
                   f"front max {worst_band:.2f} > 20%). C'est le test forçant n-width "
                   f"attendu : l'approche linéaire (POD+DMD) casse en sec -> l'encodeur "
                   f"non-linéaire est JUSTIFIÉ avec preuve. Le réduit aval à accoler à "
                   f"l'oracle W1 est l'encodeur, pas DMD. (Décision gate sur le FRONT, "
                   f"pas le global — voir l'écart dans la figure.)")
    else:
        verdict = (f"La base POD linéaire tient le front mobile (erreur front max "
                   f"{worst_band:.2f} <= 20%) — résultat peu attendu mais décisif : "
                   f"DMD linéaire reste candidat sur le régime sec. Continuer W2 avec "
                   f"l'oracle W1 et le réduit linéaire. ATTENTION (portée) : le front de "
                   f"Thacker est un touchdown LISSE (h→0 paraboliquement), le meilleur cas "
                   f"pour la POD. Un front de SOLVEUR (W1 : dam-break, sillages, "
                   f"raidissement numérique) peut être bien plus dur. W0 qui passe est "
                   f"NÉCESSAIRE, PAS SUFFISANT — le vrai test est W2 sur la dynamique "
                   f"réelle ; le transport (planaire) plus dur que la forme (radial) "
                   f"préfigure où ça pourrait mordre.")

    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# W0 — Test forçant n-width (Thacker, sans solveur)", "",
             "POD hauteur (n_channels=1) sur un trait d'eau MOBILE analytique. **Gate sur "
             "l'erreur FRONT-localisée** (le bulk lisse dilue la L2 globale).", "",
             "| oracle | k | L2 globale | L2 FRONT (gate) |", "|---|---|---|---|"]
    for kk in results:
        m = results[kk]
        lines.append(f"| {kk} | {m['k']} | {m['global_err']:.4f} | **{m['band_err']:.4f}** |")
    lines += ["", "Radial = front courbe qui *respire* (forme) ; planar_diag = front qui "
              "*translate* (transport, tueur n-width). L'écart global vs front mesure le "
              "masquage de régime évité.", "", "## Verdict", "", verdict, "",
              f"Figure : `{fig_path.relative_to(ROOT)}`."]
    OUT_DOC.write_text("\n".join(lines) + "\n")
    print(f"[W0] note -> {OUT_DOC}")
    print(f"[W0] VERDICT : {verdict}")


if __name__ == "__main__":
    main()
