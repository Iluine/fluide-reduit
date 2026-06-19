"""V0 — Famille de terrains paramétrée + condition initiale au repos submergé.

Indexation array[y, x] : axe 0 = y (H lignes), axe 1 = x (W colonnes).
Tout terrain est lisse (le schéma Rusanov du POC n'est pas well-balanced) et
SUBMERGÉ sous la surface de repos REST_SURFACE (eau partout mouillée — on ne quitte
jamais le régime humide validé du POC ; le lit sec serait une défaillance de
solveur, pas un plafond de représentation)."""
from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace

import numpy as np

from config import GridConfig, SolverConfig
from src.solver import simulate

REST_SURFACE: float = 1.5      # surface libre au repos (eta0)
MIN_REST_DEPTH: float = 0.2    # marge de submersion minimale imposée à la CI
SAMPLE_SEED: int = 20260619    # graine du tirage train/holdout (cf. sample_split)

_KIND_ID = {"bump": 0.0, "obstacle": 1.0, "channel": 2.0}


@dataclass(frozen=True)
class TerrainParams:
    """Paramètres d'un terrain ; vecteur theta canonique 6-D via to_vector().

    Sens des champs selon `kind` :
      bump / obstacle : amp, x0_frac, y0_frac, sigma, slope (gaussienne + pente).
      channel         : amp = hauteur des parois ; y0_frac = centre du corridor (y) ;
                        sigma = demi-largeur du corridor ; slope = douceur des parois
                        (cellules). x0_frac est ignoré (laisser 0.5)."""
    kind: str
    amp: float
    x0_frac: float
    y0_frac: float
    sigma: float
    slope: float

    def to_vector(self) -> np.ndarray:
        return np.array([_KIND_ID[self.kind], self.amp, self.x0_frac,
                         self.y0_frac, self.sigma, self.slope], dtype=np.float64)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "amp": self.amp, "x0_frac": self.x0_frac,
                "y0_frac": self.y0_frac, "sigma": self.sigma, "slope": self.slope}


def gaussian_terrain(grid: GridConfig, amp: float, x0_frac: float, y0_frac: float,
                     sigma: float, slope: float = 0.0) -> np.ndarray:
    """Bosse/obstacle gaussien sur plan incliné : b = amp·exp(−r²/2σ²) + slope·x/(W−1)."""
    H, W = grid.H, grid.W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = x0_frac * (W - 1), y0_frac * (H - 1)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return amp * np.exp(-r2 / (2.0 * sigma ** 2)) + slope * (xx / (W - 1))


def channel_terrain(grid: GridConfig, wall_height: float, y0_frac: float,
                    half_width: float, wall_softness: float) -> np.ndarray:
    """Canal : corridor profond le long de x, parois LISSÉES par tanh (pas de marche
    dure). b = wall_height·½(1 + tanh((|y − y_c| − half_width)/wall_softness))."""
    H, W = grid.H, grid.W
    yy, _xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cy = y0_frac * (H - 1)
    d = np.abs(yy - cy) - half_width
    return wall_height * 0.5 * (1.0 + np.tanh(d / wall_softness))


def make_terrain_from_params(grid: GridConfig, p: TerrainParams) -> np.ndarray:
    """Dérive la bathymétrie (H,W) à partir des paramètres, selon p.kind."""
    if p.kind in ("bump", "obstacle"):
        return gaussian_terrain(grid, p.amp, p.x0_frac, p.y0_frac, p.sigma, p.slope)
    if p.kind == "channel":
        return channel_terrain(grid, wall_height=p.amp, y0_frac=p.y0_frac,
                               half_width=p.sigma, wall_softness=p.slope)
    raise ValueError(f"kind de terrain inconnu : {p.kind!r}")


def rest_state_ic(grid: GridConfig, b: np.ndarray, drop_amp: float,
                  drop_x0_frac: float, drop_y0_frac: float, drop_width_frac: float,
                  rest_surface: float = REST_SURFACE
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CI au repos par rapport au relief : h = rest_surface − b, + goutte gaussienne.
    u = v = 0. Impose la submersion (b capé sous la surface avec marge)."""
    assert rest_surface - float(b.max()) >= MIN_REST_DEPTH, (
        f"terrain non submergé : b.max()={float(b.max()):.3f}, "
        f"surface={rest_surface}, marge requise={MIN_REST_DEPTH}")
    H, W = grid.H, grid.W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = drop_x0_frac * (W - 1), drop_y0_frac * (H - 1)
    sigma = drop_width_frac * min(H, W)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    drop = drop_amp * np.exp(-r2 / (2.0 * sigma ** 2))
    h = (rest_surface - b) + drop
    z = np.zeros((H, W), dtype=np.float64)
    return h, z.copy(), z.copy()


# --- Tirage du split train/holdout (append à src/terrains.py) ---

DROP_ICS: dict[str, dict] = {
    # CI vues en entraînement
    "drop_center": dict(drop_amp=0.4, drop_x0_frac=0.5, drop_y0_frac=0.5, drop_width_frac=0.1),
    "drop_offset": dict(drop_amp=0.4, drop_x0_frac=0.3, drop_y0_frac=0.6, drop_width_frac=0.1),
    # CI nouvelle, réservée aux terrains holdout
    "drop_new":    dict(drop_amp=0.4, drop_x0_frac=0.6, drop_y0_frac=0.4, drop_width_frac=0.1),
}


@dataclass(frozen=True)
class SplitEntry:
    terrain_id: str
    role: str       # "train" | "holdout_interp" | "holdout_extrap"
    regime: str     # "train" | "interp" | "extrap_obstacle" | "extrap_channel"
    params: TerrainParams
    ic_ids: tuple[str, ...]


def sample_split(grid: GridConfig, seed: int = SAMPLE_SEED) -> list[SplitEntry]:
    """Tirage déterministe : 9 terrains train (5 bosses + 4 obstacles submergés,
    répartis sur les deux topologies), 1 holdout interp (dans la plage, non tiré),
    2 holdout extrap (obstacle submergé très étroit hors plage de géométrie +
    canal, topologie absente du train). Holdout = CI nouvelle (drop_new)."""
    rng = np.random.default_rng(seed)
    train_ics = ("drop_center", "drop_offset")
    entries: list[SplitEntry] = []

    for i in range(5):  # bosses douces
        p = TerrainParams("bump",
                          amp=float(rng.uniform(0.2, 0.5)),
                          x0_frac=float(rng.uniform(0.4, 0.6)),
                          y0_frac=float(rng.uniform(0.4, 0.6)),
                          sigma=float(rng.uniform(8.0, 13.0)),
                          slope=float(rng.uniform(0.0, 0.01)))
        entries.append(SplitEntry(f"train_bump{i}", "train", "train", p, train_ics))

    for i in range(4):  # obstacles submergés (σ petit, amplitude haute mais < surface)
        p = TerrainParams("obstacle",
                          amp=float(rng.uniform(0.6, 1.0)),
                          x0_frac=float(rng.uniform(0.4, 0.6)),
                          y0_frac=float(rng.uniform(0.4, 0.6)),
                          sigma=float(rng.uniform(4.0, 7.0)),
                          slope=0.0)
        entries.append(SplitEntry(f"train_obst{i}", "train", "train", p, train_ics))

    # interp : dans les plages, distinct du tirage, CI nouvelle
    entries.append(SplitEntry(
        "holdout_interp", "holdout_interp", "interp",
        TerrainParams("obstacle", amp=0.8, x0_frac=0.55, y0_frac=0.45, sigma=5.5, slope=0.0),
        ("drop_new",)))

    # extrap (géométrie) : obstacle submergé TRÈS étroit, position hors plage
    entries.append(SplitEntry(
        "holdout_extrap_obstacle", "holdout_extrap", "extrap_obstacle",
        TerrainParams("obstacle", amp=0.6, x0_frac=0.3, y0_frac=0.65, sigma=3.0, slope=0.0),
        ("drop_new",)))

    # extrap (topologie nouvelle) : canal lissé, submergé
    entries.append(SplitEntry(
        "holdout_extrap_channel", "holdout_extrap", "extrap_channel",
        TerrainParams("channel", amp=1.0, x0_frac=0.5, y0_frac=0.5, sigma=8.0, slope=2.0),
        ("drop_new",)))

    return entries


# --- Garde-fou d'oracle : résidu au repos (well-balancedness) ---


def rest_residual(grid: GridConfig, b: np.ndarray, solver_cfg: SolverConfig,
                  rest_surface: float = REST_SURFACE, n_steps: int = 50
                  ) -> tuple[float, float]:
    """Simule l'état au repos (h = rest_surface − b, u=v=0, SANS goutte) sur n_steps
    pas et mesure les artefacts du schéma non-well-balanced : déviation de surface
    max |η − rest_surface| et vitesse parasite max √(u²+v²). Petit = oracle sain."""
    h0 = rest_surface - b
    z = np.zeros_like(b)
    cfg = _dc_replace(solver_cfg, n_steps=n_steps)
    hs, us, vs, _dt = simulate(h0, z.copy(), z.copy(), b, grid, cfg)
    eta = hs + b
    surf_dev = float(np.abs(eta - rest_surface).max())
    speed = float(np.sqrt(us ** 2 + vs ** 2).max())
    return surf_dev, speed
