"""Lecture/écriture des datasets .npz et export d'animations.

Sans dépendance aux autres modules src. Indexation array[y, x]."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Dataset:
    """Rollout oracle. h,u,v : (T,H,W) float64 ; b : (H,W) float64 ; meta : dict."""
    h: np.ndarray
    u: np.ndarray
    v: np.ndarray
    b: np.ndarray
    meta: dict


def save_dataset(path: str | Path, ds: Dataset) -> None:
    """Écrit le dataset en .npz (meta sérialisé en JSON sous la clé 'meta_json')."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        h=np.asarray(ds.h, dtype=np.float64),
        u=np.asarray(ds.u, dtype=np.float64),
        v=np.asarray(ds.v, dtype=np.float64),
        b=np.asarray(ds.b, dtype=np.float64),
        meta_json=np.array(json.dumps(ds.meta)),
    )


def load_dataset(path: str | Path) -> Dataset:
    """Relit un dataset écrit par save_dataset."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        return Dataset(
            h=data["h"].astype(np.float64),
            u=data["u"].astype(np.float64),
            v=data["v"].astype(np.float64),
            b=data["b"].astype(np.float64),
            meta=meta,
        )


def _save_montage(path: Path, frames: np.ndarray, cmap: str, title: str,
                  vmin: float | None, vmax: float | None) -> str:
    """Fallback sans Pillow : grille de jusqu'à 12 frames échantillonnées."""
    T = frames.shape[0]
    n = min(T, 12)
    idx = np.linspace(0, T - 1, n).astype(int)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for k, t in enumerate(idx):
        axes[k].imshow(frames[t], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        axes[k].set_title(f"t={t}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out = path.with_suffix(".png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return str(out)


def save_animation(path: str | Path, frames: np.ndarray, *, fps: int = 20,
                   cmap: str = "viridis", title: str = "",
                   vmin: float | None = None, vmax: float | None = None) -> str:
    """Exporte frames (T,H,W) en GIF (Pillow) ou, à défaut, en montage PNG.

    Retourne le chemin réellement écrit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames, dtype=np.float64)
    if vmin is None:
        vmin = float(frames.min())
    if vmax is None:
        vmax = float(frames.max())

    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
        import PIL  # noqa: F401  (vérifie la présence de Pillow)
    except ImportError:
        return _save_montage(path, frames, cmap, title, vmin, vmax)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(frames[0], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)

    def update(t: int):
        im.set_data(frames[t])
        return (im,)

    anim = FuncAnimation(fig, update, frames=frames.shape[0], blit=True)
    out = path.with_suffix(".gif")
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return str(out)
