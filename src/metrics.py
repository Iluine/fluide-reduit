"""Diagnostics chiffrés : conservation de masse, erreurs de rollout, saut de couture.

Module feuille : ne dépend que de numpy. Indexation array[y, x]."""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def total_mass(h: np.ndarray, dx: float, dy: float) -> float:
    """Masse totale d'un champ de hauteur (H,W) : somme(h) * dx * dy."""
    return float(np.sum(h) * dx * dy)


def mass_series(h_seq: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Masse totale par frame d'une séquence (T,H,W) -> (T,)."""
    return np.sum(h_seq, axis=(1, 2)) * dx * dy


def relative_l2_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Erreur L2 relative globale ‖pred-true‖ / (‖true‖ + eps)."""
    num = float(np.linalg.norm(pred.ravel() - true.ravel()))
    den = float(np.linalg.norm(true.ravel())) + _EPS
    return num / den


def error_growth(pred_seq: np.ndarray, true_seq: np.ndarray) -> np.ndarray:
    """Erreur L2 relative par frame (T,...) vs (T,...) -> (T,)."""
    T = pred_seq.shape[0]
    return np.array([relative_l2_error(pred_seq[t], true_seq[t]) for t in range(T)])


def seam_jump(field: np.ndarray, i0: int, j0: int, size: int) -> float:
    """Saut absolu moyen à travers les 4 bords d'une fenêtre carrée.

    Compare la valeur juste à l'intérieur du bord et juste à l'extérieur.
    Les bords qui sortent de la grille sont ignorés. field : (H,W)."""
    H, W = field.shape
    i1, j1 = i0 + size, j0 + size
    diffs: list[np.ndarray] = []
    # bord haut : ligne intérieure i0 vs ligne extérieure i0-1
    if i0 - 1 >= 0:
        diffs.append(np.abs(field[i0, j0:j1] - field[i0 - 1, j0:j1]))
    # bord bas : ligne intérieure i1-1 vs ligne extérieure i1
    if i1 < H:
        diffs.append(np.abs(field[i1 - 1, j0:j1] - field[i1, j0:j1]))
    # bord gauche : colonne intérieure j0 vs colonne extérieure j0-1
    if j0 - 1 >= 0:
        diffs.append(np.abs(field[i0:i1, j0] - field[i0:i1, j0 - 1]))
    # bord droit : colonne intérieure j1-1 vs colonne extérieure j1
    if j1 < W:
        diffs.append(np.abs(field[i0:i1, j1 - 1] - field[i0:i1, j1]))
    if not diffs:
        return 0.0
    return float(np.mean(np.concatenate([d.ravel() for d in diffs])))
