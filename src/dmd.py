"""M2 — Dynamique latente linéaire (DMD).

Identifie A (k,k) tel que z[:,t+1] ≈ A @ z[:,t], par moindres carrés sur les
paires consécutives internes à chaque trajectoire (jamais à cheval entre deux
rollouts)."""
from __future__ import annotations

import numpy as np


def fit_dmd(z_list: list[np.ndarray]) -> np.ndarray:
    """A = Z2 @ pinv(Z1) sur les paires (z_t, z_{t+1}) de chaque trajectoire."""
    z1_cols, z2_cols = [], []
    for z in z_list:
        if z.shape[1] < 2:
            continue
        z1_cols.append(z[:, :-1])
        z2_cols.append(z[:, 1:])
    Z1 = np.concatenate(z1_cols, axis=1)
    Z2 = np.concatenate(z2_cols, axis=1)
    return Z2 @ np.linalg.pinv(Z1)


def rollout(A: np.ndarray, z0: np.ndarray, n_steps: int) -> np.ndarray:
    """Rollout autorégressif z_pred (k, n_steps+1) depuis z0 (k,)."""
    k = A.shape[0]
    z = np.zeros((k, n_steps + 1))
    z[:, 0] = z0
    for t in range(n_steps):
        z[:, t + 1] = A @ z[:, t]
    return z


def spectral_radius(A: np.ndarray) -> float:
    """Rayon spectral max|eig(A)| (diagnostic de stabilité du rollout)."""
    return float(np.max(np.abs(np.linalg.eigvals(A))))
