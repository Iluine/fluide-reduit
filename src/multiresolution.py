"""M6 — Représentation à deux niveaux : grossier global + fenêtre fine mobile.

Proxy d'observateur/caméra : fin dans la fenêtre, grossier dehors. On mesure la
discontinuité à la couture (cf. metrics.seam_jump) quand la fenêtre se déplace.
Indexation array[y, x] ; fenêtre carrée [i0:i0+size, j0:j0+size]."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import GridConfig


@dataclass(frozen=True)
class Window:
    i0: int  # ligne (y) du coin haut-gauche
    j0: int  # colonne (x) du coin haut-gauche
    size: int


def downsample(field: np.ndarray, factor: int) -> np.ndarray:
    """Moyenne de blocs factor x factor. H et W doivent être divisibles par factor."""
    H, W = field.shape
    if H % factor or W % factor:
        raise ValueError(f"shape {field.shape} non divisible par {factor}")
    return field.reshape(H // factor, factor, W // factor, factor).mean(axis=(1, 3))


def upsample(coarse: np.ndarray, factor: int) -> np.ndarray:
    """Sur-échantillonnage voisin-le-plus-proche (blocs constants)."""
    return np.kron(coarse, np.ones((factor, factor)))


def compose_multiresolution(field: np.ndarray, window: Window,
                            coarse_factor: int, blend_width: int = 0) -> np.ndarray:
    """Fond grossier (down+up) + fenêtre fine. blend_width>0 = fondu fine->grossier."""
    background = upsample(downsample(field, coarse_factor), coarse_factor)
    out = background.copy()
    i0, j0, s = window.i0, window.j0, window.size
    i1, j1 = i0 + s, j0 + s

    if blend_width <= 0:
        out[i0:i1, j0:j1] = field[i0:i1, j0:j1]
        return out

    # Poids de fondu : 1 au cœur de la fenêtre, descend vers 0 sur l'anneau extérieur.
    # On fond vers la valeur background du voisin extérieur (pas intérieur) pour garantir
    # la continuité même quand le bord tombe sur une frontière de bloc grossier.
    H, W = field.shape
    yy, xx = np.mgrid[i0:i1, j0:j1]
    dist = np.minimum.reduce([yy - i0, i1 - 1 - yy, xx - j0, j1 - 1 - xx]).astype(float)
    w = np.clip((dist + 1.0) / (blend_width + 1.0), 0.0, 1.0)

    # Cible de fondu : valeur du fond au voisin extérieur le plus proche (clampé au bord)
    ii_out = np.clip(np.where(yy - i0 < i1 - 1 - yy, i0 - 1, i1), 0, H - 1)
    jj_out = np.clip(np.where(xx - j0 < j1 - 1 - xx, j0 - 1, j1), 0, W - 1)
    # Choisir la direction dominante (la plus proche du bord)
    dy = np.minimum(yy - i0, i1 - 1 - yy)
    dx = np.minimum(xx - j0, j1 - 1 - xx)
    use_y = dy <= dx
    target_i = np.where(use_y, ii_out, yy)
    target_j = np.where(use_y, jj_out, jj_out)  # pour les coins, y prend le dessus
    target_i = np.where(use_y, ii_out, yy)
    target_j = np.where(use_y, xx, jj_out)
    blend_target = background[target_i, target_j]

    out[i0:i1, j0:j1] = w * field[i0:i1, j0:j1] + (1.0 - w) * blend_target
    return out


def window_trajectory(grid: GridConfig, size: int, n_frames: int,
                      axis: str = "x", margin: int = 4) -> list[Window]:
    """Fenêtre carrée qui translate linéairement (axe 'x' = colonnes, 'y' = lignes)."""
    if axis == "x":
        i0 = (grid.H - size) // 2
        lo, hi = margin, grid.W - size - margin
        js = np.linspace(lo, hi, n_frames).astype(int)
        return [Window(int(i0), int(j), size) for j in js]
    if axis == "y":
        j0 = (grid.W - size) // 2
        lo, hi = margin, grid.H - size - margin
        iss = np.linspace(lo, hi, n_frames).astype(int)
        return [Window(int(i), int(j0), size) for i in iss]
    raise ValueError(f"axis inconnu : {axis!r}")
