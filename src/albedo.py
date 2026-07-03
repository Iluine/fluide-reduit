"""Arc A / Task 1 — Readouts (albedo, relief ombré) + instruments perceptuels.

Deux readouts du champ sédiment `s` (ou du relief `b`) :
  - `albedo`   : readout SÉDIMENT — assombrissement saturant en s (point d'opération).
  - `relief_shaded` : readout RELIEF — ombrage lambertien de b0+s, référence de
    re-validation UNIQUEMENT (cf. Global Constraints : jamais de comparaison L2 sur
    l'état, seul le readout perceptuel compte).

Deux instruments perceptuels (gravés au journal pocCascade2phys 2026-07-03,
répliqués ici EXACTEMENT — cf. contrat) opèrent sur des readouts déjà calculés
(pas sur les champs bruts s/b) :
  - `f_fraction(x, y, jnd)` : fraction du domaine où le readout candidat `x` dévie
    du readout de référence `y` de plus de `jnd` (JND = just noticeable difference),
    normalisé par la moyenne spatiale de `y`.
  - `chi_bands` / `delta_chi` : décomposition spectrale radiale (octaves) du readout
    + écart par bande, restreint aux bandes PORTEUSES (celles qui portent l'essentiel
    de l'énergie du signal AC de la référence).

Indexation array[y, x] comme le reste du dépôt."""
from __future__ import annotations

import numpy as np

# --- Readouts ---------------------------------------------------------------


def albedo(s: np.ndarray, s_half: float) -> np.ndarray:
    """A = 1 − exp(−s / s_half) : 0 pour s=0, sature vers 1 quand s ≫ s_half.
    Monotone croissante en s (pas de recouvrement d'ordre possible)."""
    return 1.0 - np.exp(-s / s_half)


def relief_shaded(b: np.ndarray, *, dx: float = 1.0, dy: float = 1.0,
                  elevation_deg: float = 15.0, azimuth_deg: float = 45.0) -> np.ndarray:
    """Ombrage lambertien rasant de `b` (H, W) : source directionnelle à
    `elevation_deg` au-dessus de l'horizon, azimut `azimuth_deg` (convention
    mathématique : 0° = +x, sens trigonométrique direct dans le plan (x, y) —
    axe 1 = x, axe 0 = y comme le reste du dépôt).

    Normale de surface via gradient centré (np.gradient) : N = (−∂b/∂x, −∂b/∂y, 1),
    normalisée. Éclairement = max(0, N·L) où L est le vecteur unitaire de la source.
    Le produit scalaire de deux vecteurs unitaires est dans [−1, 1] ; le tronquer à
    [0, 1] EST la normalisation demandée (lobe auto-ombré à 0, pas de rescale
    supplémentaire qui changerait la sémantique des écarts relatifs)."""
    dbdy, dbdx = np.gradient(b, dy, dx)
    normal = np.stack([-dbdx, -dbdy, np.ones_like(b)], axis=-1)
    normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    elev = np.deg2rad(elevation_deg)
    az = np.deg2rad(azimuth_deg)
    light = np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)])
    shade = normal @ light
    return np.clip(shade, 0.0, 1.0)


# --- Instrument f_fraction (JND ponctuel) -----------------------------------


def f_fraction(x: np.ndarray, y: np.ndarray, jnd: float) -> float:
    """Fraction du domaine où |x − y| / ⟨y⟩ > jnd (⟨y⟩ = moyenne spatiale du
    readout de RÉFÉRENCE y, sert de normalisation). f_fraction(x, x) = 0 par
    construction (numérateur nul partout, y compris si ⟨y⟩ = 0 — cas dégénéré géré
    explicitement pour éviter un 0/0)."""
    diff = np.abs(np.asarray(x) - np.asarray(y))
    denom = float(np.mean(y))
    if denom == 0.0:
        rel = np.where(diff == 0.0, 0.0, np.inf)
    else:
        rel = diff / denom
    return float(np.mean(rel > jnd))


# --- Instrument chi_bands / delta_chi (spectre radial par octave) ----------

# Bandes radiales en cycles/domaine (octaves) : la borne haute de la dernière bande
# est ouverte (couvre tout k_r >= 16, jusqu'au Nyquist ~45.25 pour une grille 64x64)
# de sorte que les 5 bandes PARTITIONNENT exactement tout le spectre AC (hors DC) —
# c'est ce qui garantit l'invariant "les puissances par bande somment à la puissance
# AC totale" (testé). Le nom "16-31" reste la borne nominale FIGÉE au journal
# d'origine ; en pratique, pour une grille 64x64, l'essentiel de l'énergie de cette
# bande est de toute façon à k_r <= 31 (bin le plus proche du Nyquist).
_BAND_LABELS: tuple[str, ...] = ("1", "2-3", "4-7", "8-15", "16-31")
_BAND_BOUNDS: tuple[tuple[int, int], ...] = ((1, 1), (2, 3), (4, 7), (8, 15), (16, 1 << 30))


def _radial_bins(shape: tuple[int, int]) -> np.ndarray:
    """Grille de fréquences radiales entières k_r = round(sqrt(kx²+ky²)), avec
    kx, ky ∈ fftfreq(N)·N (cycles/domaine), pour une grille carrée (H=W=N)."""
    H, W = shape
    ky = np.fft.fftfreq(H) * H
    kx = np.fft.fftfreq(W) * W
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_r = np.sqrt(kx_grid ** 2 + ky_grid ** 2)
    return np.round(k_r).astype(np.int64)


def chi_bands(a: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """FFT 2D de `a`, puissance |F|² regroupée par bande radiale (octaves, cf.
    `_BAND_BOUNDS`) ; la composante DC (k_r arrondi à 0) est exclue de tout.
    χ_b = RMS(bande)/⟨a⟩ où RMS(bande) = sqrt(Σ_{k∈bande} |F_k|² / N⁴) (Parseval,
    N=H=W) restitue la variance réelle représentée par cette bande. Retourne
    (chi: {bande: χ_b}, puissance: {bande: Σ|F_k|²})."""
    a = np.asarray(a, dtype=np.float64)
    H, W = a.shape
    F = np.fft.fft2(a)
    power2d = np.abs(F) ** 2
    k_int = _radial_bins((H, W))
    mean_a = float(np.mean(a))

    chi: dict[str, float] = {}
    power: dict[str, float] = {}
    for label, (lo, hi) in zip(_BAND_LABELS, _BAND_BOUNDS):
        mask = (k_int >= lo) & (k_int <= hi)
        p = float(power2d[mask].sum())
        power[label] = p
        rms = np.sqrt(p / (H * W) ** 2)
        chi[label] = rms / mean_a if mean_a != 0.0 else (0.0 if p == 0.0 else np.inf)
    return chi, power


def delta_chi(a_cand: np.ndarray, a_ref: np.ndarray) -> dict:
    """Δχ_b = |χ_b(candidat) − χ_b(vrai)| par bande ; bandes PORTEUSES = celles dont
    la puissance de la RÉFÉRENCE représente ≥10 % de la puissance AC totale de la
    référence ; critère = max(Δχ_b) sur les bandes porteuses (0.0 si aucune bande
    n'est porteuse — cas dégénéré, ex. référence constante).

    Retourne {"delta_chi": {bande: Δχ_b}, "carriers": {bande: bool},
    "max_carrier": float}. `delta_chi(a, a) = {"max_carrier": 0.0, ...}`."""
    chi_c, _power_c = chi_bands(a_cand)
    chi_r, power_r = chi_bands(a_ref)
    total_ac = sum(power_r.values())

    delta = {b: abs(chi_c[b] - chi_r[b]) for b in _BAND_LABELS}
    if total_ac > 0.0:
        carriers = {b: (power_r[b] / total_ac) >= 0.10 for b in _BAND_LABELS}
    else:
        carriers = {b: False for b in _BAND_LABELS}
    carrier_deltas = [delta[b] for b in _BAND_LABELS if carriers[b]]
    max_carrier = max(carrier_deltas) if carrier_deltas else 0.0
    return {"delta_chi": delta, "carriers": carriers, "max_carrier": max_carrier}
