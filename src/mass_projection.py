"""M5 — Projection de masse (conservation dure) par offset uniforme additif.

Garde-fou de SORTIE : on projette le champ de hauteur décodé sur la variété à
masse constante, sans rétroaction sur la dynamique latente. La correction de
norme minimale pour une contrainte intégrale (somme de h) est un décalage
uniforme, qui préserve exactement la forme spatiale. Module feuille (numpy)."""
from __future__ import annotations

import numpy as np


def project_mass(h: np.ndarray, target_mass: float, dx: float, dy: float) -> np.ndarray:
    """Décale h d'un offset uniforme pour que sum(h)*dx*dy == target_mass.

    h : champ (H, W). Lève AssertionError si la projection rend une profondeur
    négative (non physique) — il faudrait alors une projection à contrainte de
    positivité (qui sacrifierait la conservation exacte)."""
    H, W = h.shape
    current = float(np.sum(h) * dx * dy)
    delta = (target_mass - current) / (H * W * dx * dy)
    h_corr = h + delta
    assert (h_corr > 0.0).all(), (
        f"project_mass: profondeur négative après projection "
        f"(min={float(h_corr.min()):.3g}, delta={delta:.3g}). Dérive trop forte "
        f"pour une projection uniforme : une projection à contrainte de positivité serait requise.")
    return h_corr


def project_mass_series(h_seq: np.ndarray, target_mass: float, dx: float, dy: float) -> np.ndarray:
    """Applique project_mass à chaque frame d'une séquence (T, H, W) -> (T, H, W)."""
    return np.stack([project_mass(h_seq[t], target_mass, dx, dy) for t in range(h_seq.shape[0])])
