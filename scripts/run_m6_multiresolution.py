"""M6 — H3 : cohérence de la couture quand la fenêtre fine se déplace.

Usage : .venv/bin/python scripts/run_m6_multiresolution.py
Interprétation attendue : quantifier le saut résiduel au bord de la fenêtre au
fil du temps (collage dur vs fondu) -> caractériser le 'popping'."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from config import GridConfig
from src.io_utils import load_dataset, save_animation
from src.multiresolution import compose_multiresolution, window_trajectory
from src.metrics import seam_jump

# ----------------------------- CONFIG ------------------------------------
SOURCE_CASE = "drop_center"
COARSE_FACTOR = 4    # 64 -> 16 grossier
WINDOW_SIZE = 16
BLEND_WIDTH = 3      # largeur d'anneau de fondu (variante atténuée)
AXIS = "x"
DATA = ROOT / "data" / "ground_truth"
OUT = ROOT / "outputs"
# -------------------------------------------------------------------------


def main() -> None:
    """Évalue la cohérence de couture multiresolution (H3) et génère les figures."""
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATA / f"{SOURCE_CASE}.npz")
    h = ds.h                      # (T,H,W) — on observe le canal hauteur
    T, H, W = h.shape
    grid = GridConfig(H=H, W=W, dx=ds.meta["dx"], dy=ds.meta["dy"])
    wins = window_trajectory(grid, WINDOW_SIZE, T, axis=AXIS)

    jumps_hard = np.zeros(T)
    jumps_soft = np.zeros(T)
    composed_frames = np.zeros((T, H, W))
    for t in range(T):
        w = wins[t]
        hard = compose_multiresolution(h[t], w, COARSE_FACTOR, blend_width=0)
        soft = compose_multiresolution(h[t], w, COARSE_FACTOR, blend_width=BLEND_WIDTH)
        jumps_hard[t] = seam_jump(hard, w.i0, w.j0, w.size)
        jumps_soft[t] = seam_jump(soft, w.i0, w.j0, w.size)
        composed_frames[t] = soft

    # Figure : saut de couture vs temps
    plt.figure(figsize=(6, 4))
    plt.plot(jumps_hard, label=f"collage dur (moy={jumps_hard.mean():.3f})")
    plt.plot(jumps_soft, label=f"fondu w={BLEND_WIDTH} (moy={jumps_soft.mean():.3f})")
    plt.xlabel("pas de temps (fenêtre en déplacement)")
    plt.ylabel("saut de couture moyen")
    plt.title("M6 — H3 : saut à la couture vs temps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "m6_seam_jump.png", dpi=120)
    plt.close()

    written = save_animation(OUT / "m6_window_moving.gif", composed_frames, fps=15,
                             title="M6 — fenêtre fine mobile (fond grossier)")
    print(f"[M6] verdict H3 : saut moyen collage_dur={jumps_hard.mean():.4f} "
          f"max={jumps_hard.max():.4f} | fondu={jumps_soft.mean():.4f} max={jumps_soft.max():.4f}")
    print(f"[M6] animation -> {written}")


if __name__ == "__main__":
    main()
