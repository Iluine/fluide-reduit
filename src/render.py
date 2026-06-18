"""M7 — Rendu comme relevé de l'état : heatmap et surface de hauteur.

L'image est un relevé direct du champ simulé/prédit (« ce que tu vois est ce
que tu simules »)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.io_utils import save_animation


def surface_height(h: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Surface libre eta = h + b (b diffusé sur l'axe temporel si besoin)."""
    return h + b


def render_rollout(path: str | Path, field_seq: np.ndarray, *, cmap: str = "viridis",
                   fps: int = 20, title: str = "") -> str:
    """Exporte une séquence (T,H,W) en animation (délègue à io_utils)."""
    return save_animation(path, field_seq, fps=fps, cmap=cmap, title=title)
