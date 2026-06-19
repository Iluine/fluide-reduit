"""M1 — Base réduite POD par SVD des snapshots.

Contrat (cf. spec §5, avec extension documentée 'scale') :
    X ≈ scale[:,None] * (Phi @ z) + mean[:,None]
Ordre des canaux dans X : [h, u, v]. n_features = 3*H*W.
Le 'scale' (écart-type par canal, diffusé sur les features) équilibre les
amplitudes de h (~1) et u,v (~0.1) pour que la SVD ne soit pas dominée par h."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass
class PODBasis:
    mean: np.ndarray            # (n_features,)
    scale: np.ndarray           # (n_features,)
    Phi: np.ndarray             # (n_features, k)
    singular_values: np.ndarray  # (n_modes,) spectre complet (pour énergie)


def stack_snapshots(h: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(T,H,W)*3 -> X (3*H*W, T), canaux empilés dans l'ordre [h,u,v]."""
    T, H, W = h.shape
    def flat(a: np.ndarray) -> np.ndarray:
        return a.reshape(T, H * W).T
    return np.concatenate([flat(h), flat(u), flat(v)], axis=0)


def unstack(X: np.ndarray, H: int, W: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X (3*H*W, n) -> (h,u,v) chacun (n,H,W)."""
    n = X.shape[1]
    hw = H * W
    parts = [X[i * hw:(i + 1) * hw, :].T.reshape(n, H, W) for i in range(3)]
    return parts[0], parts[1], parts[2]


def _channel_scale(X: np.ndarray, n_channels: int = 3) -> np.ndarray:
    """Écart-type par canal (n_channels blocs égaux), diffusé sur (n_features,).
    n_channels=3 : comportement POC ([h,u,v]). n_channels=1 : un seul écart-type global."""
    n_features = X.shape[0]
    assert n_features % n_channels == 0, f"{n_features} non divisible par {n_channels} canaux"
    block_size = n_features // n_channels
    scale = np.empty(n_features)
    for i in range(n_channels):
        block = X[i * block_size:(i + 1) * block_size, :]
        s = float(block.std())
        scale[i * block_size:(i + 1) * block_size] = s if s > _EPS else 1.0
    return scale


def fit_pod(X: np.ndarray, energy_threshold: float, max_modes: int,
            n_channels: int = 3) -> PODBasis:
    """Centre + met à l'échelle par canal, SVD économique, choisit k au seuil.
    n_channels par défaut = 3 (comportement POC, ordre [h,u,v]) ; n_channels=1 pour
    une base de hauteur seule (cf. V1)."""
    mean = X.mean(axis=1)
    scale = _channel_scale(X, n_channels=n_channels)
    Xn = (X - mean[:, None]) / scale[:, None]
    U, s, _ = np.linalg.svd(Xn, full_matrices=False)
    energy = cumulative_energy(s)
    k = int(np.searchsorted(energy, energy_threshold) + 1)
    k = max(1, min(k, max_modes, U.shape[1]))
    return PODBasis(mean=mean, scale=scale, Phi=U[:, :k], singular_values=s)


def encode(basis: PODBasis, X: np.ndarray) -> np.ndarray:
    """X (n_features, n) -> z (k, n)."""
    Xn = (X - basis.mean[:, None]) / basis.scale[:, None]
    return basis.Phi.T @ Xn


def decode(basis: PODBasis, z: np.ndarray) -> np.ndarray:
    """z (k, n) -> X (n_features, n)."""
    Xn = basis.Phi @ z
    return basis.scale[:, None] * Xn + basis.mean[:, None]


def cumulative_energy(singular_values: np.ndarray) -> np.ndarray:
    """Énergie cumulée normalisée (cumsum des sigma^2 / somme)."""
    e2 = singular_values ** 2
    total = float(e2.sum())
    if total < _EPS:
        return np.ones(len(singular_values))
    return np.cumsum(e2) / total


def stack_height(h_seq: np.ndarray) -> np.ndarray:
    """(T,H,W) -> X (H*W, T) : snapshots de hauteur seule (un canal)."""
    T, H, W = h_seq.shape
    return h_seq.reshape(T, H * W).T


def unstack_height(X: np.ndarray, H: int, W: int) -> np.ndarray:
    """X (H*W, n) -> (n,H,W) : inverse de stack_height."""
    n = X.shape[1]
    return X.T.reshape(n, H, W)
