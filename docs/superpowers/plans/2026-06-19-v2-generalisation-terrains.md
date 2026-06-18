# v2 — Généralisation inter-terrain : colonne diagnostique V0 + V1 — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construire la famille de terrains paramétrée (V0) puis mesurer le plafond de représentation d'une base POD de hauteur sur terrains/CI jamais vus (V1, le pas décisif).

**Architecture:** Réutilise intégralement le solveur, la POD, les métriques et le rendu du POC (tag `v1`). N'ajoute que `src/terrains.py` (famille paramétrée + CI au repos submergé + garde-fous d'oracle) et deux scripts `run_v0`/`run_v1`. La POD travaille sur la **hauteur seule** ; on étend `src/pod.py` de façon rétro-compatible (paramètre `n_channels`, défaut 3 = comportement POC inchangé).

**Tech Stack:** Python 3.12 (venv `.venv` via `uv`), numpy/scipy/matplotlib uniquement. Lancement : `.venv/bin/python …`, tests : `.venv/bin/python -m pytest`.

## Global Constraints

- **Frugal, CPU, numpy/scipy/matplotlib seulement.** Pas de `pytorch` (réservé V5). Demander avant toute nouvelle dépendance.
- **Réutiliser les modules du POC sans les réécrire** (`solver.py`, `metrics.py`, `render.py`, `io_utils.py`) ; `pod.py` est **étendu** (param `n_channels` rétro-compatible), pas réécrit. N'ajouter que `src/terrains.py` et les scripts `run_v*`.
- **Non-régression** : le cas mono-terrain du POC (tag `v1`) reste intact ; la suite des 35 tests existants doit continuer à passer (verdict : `.venv/bin/python -m pytest -q` → 35+ passed).
- **Indexation `array[y, x]`** (axe 0 = y = lignes = H, axe 1 = x = colonnes = W). `u` selon x, `v` selon y. `b` = bathymétrie.
- **CI = repos submergé** : `h = η₀ − b` (+ goutte), `η₀ = REST_SURFACE = 1.5`. Tout `b` capé sous η₀ avec marge → eau partout mouillée. **Jamais de lit sec** (ce serait une défaillance de solveur, pas un plafond de représentation).
- **Toute bathymétrie est lisse** (gaussiennes, canal à parois tanh) — le schéma Rusanov n'est pas well-balanced : pas de marche dure.
- **Extrapolation par la géométrie** (σ étroit / position hors plage / topologie canal), jamais par une amplitude qui assèche.
- **Contrat données** : `data/v2/<terrain_id>__<ic_id>.npz` contient `h,u,v,b,theta` + `meta` (dx, dt, schema, cfl, terrain_id, ic_id, regime, role, rest_surface). Split dans `data/v2/split.json`.
- **Toujours distinguer interp vs extrap** dans les métriques et figures.
- Constantes partagées (définies dans `src/terrains.py`) : `REST_SURFACE = 1.5`, `MIN_REST_DEPTH = 0.2` (marge de submersion imposée à la CI), `SAMPLE_SEED = 20260619`.

---

### Task 1 : `src/terrains.py` — paramètres, générateurs de terrain, CI au repos submergé

**Files:**
- Create: `src/terrains.py`
- Test: `tests/test_terrains.py`

**Interfaces:**
- Consumes : `config.GridConfig` ; `src.solver.make_terrain` (uniquement pour le test de non-régression du bump).
- Produces :
  - `REST_SURFACE: float = 1.5`, `MIN_REST_DEPTH: float = 0.2`
  - `@dataclass(frozen=True) class TerrainParams` champs `kind:str, amp:float, x0_frac:float, y0_frac:float, sigma:float, slope:float` ; méthodes `to_vector() -> np.ndarray` (6-D `[kind_id, amp, x0_frac, y0_frac, sigma, slope]`, kind_id ∈ {bump:0, obstacle:1, channel:2}) et `to_dict() -> dict`.
  - `gaussian_terrain(grid, amp, x0_frac, y0_frac, sigma, slope=0.0) -> np.ndarray` (H,W)
  - `channel_terrain(grid, wall_height, y0_frac, half_width, wall_softness) -> np.ndarray` (H,W)
  - `make_terrain_from_params(grid, p: TerrainParams) -> np.ndarray` (H,W)
  - `rest_state_ic(grid, b, drop_amp, drop_x0_frac, drop_y0_frac, drop_width_frac, rest_surface=REST_SURFACE) -> tuple[np.ndarray, np.ndarray, np.ndarray]` (h, u, v) chacun (H,W)

- [ ] **Step 1 : Écrire les tests qui échouent**

```python
# tests/test_terrains.py
import numpy as np
import pytest

from config import GridConfig
from src.solver import make_terrain
from src.terrains import (REST_SURFACE, MIN_REST_DEPTH, TerrainParams,
                          gaussian_terrain, channel_terrain,
                          make_terrain_from_params, rest_state_ic)

GRID = GridConfig(H=64, W=64, dx=1.0, dy=1.0)


def test_gaussian_terrain_matches_poc_bump():
    # Le bump du POC (make_terrain "bump") est un point de la famille : amp 0.4,
    # centre (0.5,0.5), sigma=min(H,W)/6, pas de pente. Pont de non-régression.
    b = gaussian_terrain(GRID, amp=0.4, x0_frac=0.5, y0_frac=0.5,
                         sigma=min(GRID.H, GRID.W) / 6.0, slope=0.0)
    assert b.shape == (64, 64)
    assert np.allclose(b, make_terrain(GRID, "bump"))


def test_gaussian_terrain_slope_adds_tilt():
    b = gaussian_terrain(GRID, amp=0.0, x0_frac=0.5, y0_frac=0.5, sigma=10.0, slope=0.02)
    # pente pure : croît de 0 (x=0) à 0.02 (x=W-1), constante en y
    assert b[0, 0] == pytest.approx(0.0)
    assert b[0, -1] == pytest.approx(0.02)
    assert np.allclose(b[0, :], b[-1, :])


def test_channel_terrain_is_smooth_and_bounded():
    b = channel_terrain(GRID, wall_height=1.0, y0_frac=0.5, half_width=8.0, wall_softness=2.0)
    assert b.shape == (64, 64)
    assert b.min() >= 0.0 and b.max() <= 1.0 + 1e-9
    # corridor central (autour de y0) peu élevé ; bords (parois) hauts
    assert b[32, 0] < 0.05          # centre du corridor
    assert b[0, 0] > 0.9            # paroi
    # lissage : gradient discret borné (pas de marche dure)
    grad = np.abs(np.diff(b, axis=0))
    assert grad.max() < 0.5


def test_make_terrain_from_params_dispatch():
    pb = TerrainParams("bump", amp=0.3, x0_frac=0.5, y0_frac=0.5, sigma=10.0, slope=0.0)
    pc = TerrainParams("channel", amp=1.0, x0_frac=0.5, y0_frac=0.5, sigma=8.0, slope=2.0)
    assert make_terrain_from_params(GRID, pb).shape == (64, 64)
    assert make_terrain_from_params(GRID, pc).shape == (64, 64)
    with pytest.raises(ValueError):
        make_terrain_from_params(GRID, TerrainParams("zzz", 0, 0.5, 0.5, 1, 0))


def test_terrain_params_vector_roundtrip():
    p = TerrainParams("obstacle", amp=0.8, x0_frac=0.55, y0_frac=0.45, sigma=5.0, slope=0.0)
    v = p.to_vector()
    assert v.shape == (6,)
    assert v[0] == 1.0  # obstacle
    assert v[1] == pytest.approx(0.8)
    assert p.to_dict()["kind"] == "obstacle"


def test_rest_state_ic_is_submerged_and_positive():
    b = gaussian_terrain(GRID, amp=1.0, x0_frac=0.5, y0_frac=0.5, sigma=5.0, slope=0.0)
    h, u, v = rest_state_ic(GRID, b, drop_amp=0.4, drop_x0_frac=0.6,
                            drop_y0_frac=0.4, drop_width_frac=0.1)
    assert h.shape == u.shape == v.shape == (64, 64)
    assert np.all(u == 0) and np.all(v == 0)
    # surface au repos = REST_SURFACE loin de la goutte ; h = REST_SURFACE - b là-bas
    # partout strictement positif avec marge
    assert h.min() > MIN_REST_DEPTH - 1e-9
    # la goutte ajoute de l'eau : pic > niveau de repos local
    assert h.max() > REST_SURFACE - b.min()


def test_rest_state_ic_rejects_unsubmerged_terrain():
    # un b qui perce la surface (b >= REST_SURFACE - marge) doit lever
    b = np.full((64, 64), REST_SURFACE, dtype=np.float64)
    with pytest.raises(AssertionError):
        rest_state_ic(GRID, b, drop_amp=0.4, drop_x0_frac=0.5,
                      drop_y0_frac=0.5, drop_width_frac=0.1)
```

- [ ] **Step 2 : Lancer les tests pour vérifier qu'ils échouent**

Run: `.venv/bin/python -m pytest tests/test_terrains.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.terrains'`).

- [ ] **Step 3 : Implémenter `src/terrains.py` (générateurs + CI)**

```python
"""V0 — Famille de terrains paramétrée + condition initiale au repos submergé.

Indexation array[y, x] : axe 0 = y (H lignes), axe 1 = x (W colonnes).
Tout terrain est lisse (le schéma Rusanov du POC n'est pas well-balanced) et
SUBMERGÉ sous la surface de repos REST_SURFACE (eau partout mouillée — on ne quitte
jamais le régime humide validé du POC ; le lit sec serait une défaillance de
solveur, pas un plafond de représentation)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import GridConfig

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
```

- [ ] **Step 4 : Lancer les tests pour vérifier qu'ils passent**

Run: `.venv/bin/python -m pytest tests/test_terrains.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5 : Commit**

```bash
git add src/terrains.py tests/test_terrains.py
git commit -m "feat(v2): famille de terrains paramétrée + CI au repos submergé"
```

---

### Task 2 : `src/terrains.py` — tirage déterministe du split train/holdout

**Files:**
- Modify: `src/terrains.py` (ajout en fin de module)
- Test: `tests/test_terrains.py` (ajout)

**Interfaces:**
- Consumes : `TerrainParams`, `make_terrain_from_params`, `REST_SURFACE`, `MIN_REST_DEPTH`, `SAMPLE_SEED` (Task 1) ; `config.GridConfig`.
- Produces :
  - `DROP_ICS: dict[str, dict]` — params de goutte par `ic_id` (`drop_center`, `drop_offset`, `drop_new`).
  - `@dataclass(frozen=True) class SplitEntry` champs `terrain_id:str, role:str, regime:str, params:TerrainParams, ic_ids:tuple[str, ...]`.
  - `sample_split(grid, seed=SAMPLE_SEED) -> list[SplitEntry]` — 9 train (5 bump + 4 obstacle, 2 CI), 1 holdout_interp, 2 holdout_extrap (obstacle étroit hors plage + canal), toutes CI holdout = `drop_new`.

- [ ] **Step 1 : Écrire les tests qui échouent**

```python
# tests/test_terrains.py  (append)
from src.terrains import DROP_ICS, SplitEntry, sample_split, SAMPLE_SEED


def test_sample_split_counts_and_topologies():
    entries = sample_split(GRID)
    roles = [e.role for e in entries]
    assert roles.count("train") == 9
    assert roles.count("holdout_interp") == 1
    assert roles.count("holdout_extrap") == 2
    kinds = {e.params.kind for e in entries if e.role == "train"}
    assert {"bump", "obstacle"} <= kinds  # les deux topologies présentes en train


def test_sample_split_train_in_range_and_submerged():
    for e in sample_split(GRID):
        b = make_terrain_from_params(GRID, e.params)
        assert REST_SURFACE - float(b.max()) >= MIN_REST_DEPTH  # submergé partout
        if e.role == "train" and e.params.kind == "bump":
            assert 0.2 <= e.params.amp <= 0.5
            assert 8.0 <= e.params.sigma <= 13.0
        if e.role == "train" and e.params.kind == "obstacle":
            assert 0.6 <= e.params.amp <= 1.0
            assert 4.0 <= e.params.sigma <= 7.0


def test_sample_split_extrapolation_is_geometric():
    entries = {e.regime: e for e in sample_split(GRID)}
    assert "extrap_obstacle" in entries and "extrap_channel" in entries
    obst = entries["extrap_obstacle"].params
    # extrapolation par la GÉOMÉTRIE : sigma sous la plage train [4,7], amp submergée
    assert obst.sigma < 4.0
    assert obst.amp <= 1.0
    # topologie nouvelle
    assert entries["extrap_channel"].params.kind == "channel"


def test_sample_split_is_deterministic():
    a = sample_split(GRID, seed=SAMPLE_SEED)
    b = sample_split(GRID, seed=SAMPLE_SEED)
    assert [e.params.to_vector().tolist() for e in a] == \
           [e.params.to_vector().tolist() for e in b]


def test_holdout_uses_new_ic():
    for e in sample_split(GRID):
        if e.role.startswith("holdout"):
            assert e.ic_ids == ("drop_new",)
    assert "drop_new" in DROP_ICS and "drop_center" in DROP_ICS
```

- [ ] **Step 2 : Lancer les tests pour vérifier qu'ils échouent**

Run: `.venv/bin/python -m pytest tests/test_terrains.py -q`
Expected: FAIL (`ImportError: cannot import name 'sample_split'`).

- [ ] **Step 3 : Implémenter le tirage (append à `src/terrains.py`)**

```python
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
        TerrainParams("obstacle", amp=1.0, x0_frac=0.3, y0_frac=0.65, sigma=3.0, slope=0.0),
        ("drop_new",)))

    # extrap (topologie nouvelle) : canal lissé, submergé
    entries.append(SplitEntry(
        "holdout_extrap_channel", "holdout_extrap", "extrap_channel",
        TerrainParams("channel", amp=1.0, x0_frac=0.5, y0_frac=0.5, sigma=8.0, slope=2.0),
        ("drop_new",)))

    return entries
```

- [ ] **Step 4 : Lancer les tests pour vérifier qu'ils passent**

Run: `.venv/bin/python -m pytest tests/test_terrains.py -q`
Expected: PASS (12 tests au total dans le fichier).

- [ ] **Step 5 : Commit**

```bash
git add src/terrains.py tests/test_terrains.py
git commit -m "feat(v2): tirage déterministe du split train/holdout (interp + extrap géométrique)"
```

---

### Task 3 : `src/terrains.py` — résidu au repos (garde-fou de well-balancedness)

**Files:**
- Modify: `src/terrains.py` (ajout)
- Test: `tests/test_terrains.py` (ajout)

**Interfaces:**
- Consumes : `REST_SURFACE` (Task 1) ; `config.GridConfig`, `config.SolverConfig` ; `src.solver.simulate`.
- Produces : `rest_residual(grid, b, solver_cfg, rest_surface=REST_SURFACE, n_steps=50) -> tuple[float, float]` — `(déviation_surface_max, vitesse_parasite_max)` de l'état au repos simulé SANS goutte. Petit = oracle sain ; grand = bathymétrie trop raide pour le schéma non-well-balanced.

- [ ] **Step 1 : Écrire les tests qui échouent**

```python
# tests/test_terrains.py  (append)
from config import SolverConfig
from src.terrains import rest_residual

_SOLVER = SolverConfig(cfl=0.45, n_steps=800, save_every=4, min_depth=1e-3)


def test_rest_residual_flat_is_near_zero():
    # bathymétrie plate + surface constante = lac au repos exact : aucun courant parasite
    b = np.zeros((64, 64), dtype=np.float64)
    surf_dev, speed = rest_residual(GRID, b, _SOLVER, n_steps=30)
    assert surf_dev < 1e-9
    assert speed < 1e-9


def test_rest_residual_detects_bathymetry_but_stays_bounded():
    b = gaussian_terrain(GRID, amp=0.4, x0_frac=0.5, y0_frac=0.5, sigma=10.0, slope=0.0)
    surf_dev, speed = rest_residual(GRID, b, _SOLVER, n_steps=30)
    # le schéma n'est pas well-balanced -> résidu non nul, mais doit rester petit
    # (terrain doux) : ni explosion, ni NaN
    assert np.isfinite(surf_dev) and np.isfinite(speed)
    assert surf_dev > 0.0          # détecte bien le gradient de bathymétrie
    assert surf_dev < 0.2          # reste borné (oracle sain pour un terrain doux)
```

- [ ] **Step 2 : Lancer les tests pour vérifier qu'ils échouent**

Run: `.venv/bin/python -m pytest tests/test_terrains.py -q`
Expected: FAIL (`ImportError: cannot import name 'rest_residual'`).

- [ ] **Step 3 : Implémenter `rest_residual` (append à `src/terrains.py`)**

```python
# --- Garde-fou d'oracle : résidu au repos (well-balancedness) ---
from dataclasses import replace as _dc_replace  # placé en tête de module à l'implémentation

from config import SolverConfig  # idem : regrouper avec les imports en tête
from src.solver import simulate


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
```

> Note d'implémentation : remonter `from dataclasses import replace`, `from config import SolverConfig` et `from src.solver import simulate` dans le bloc d'imports en tête de `src/terrains.py` (ne pas laisser des imports en milieu de fichier).

- [ ] **Step 4 : Lancer les tests pour vérifier qu'ils passent**

Run: `.venv/bin/python -m pytest tests/test_terrains.py -q`
Expected: PASS (14 tests au total).

- [ ] **Step 5 : Commit**

```bash
git add src/terrains.py tests/test_terrains.py
git commit -m "feat(v2): rest_residual — garde-fou de well-balancedness de l'oracle"
```

---

### Task 4 : `scripts/run_v0_generate.py` — génération du dataset + garde-fous d'oracle

**Files:**
- Create: `scripts/run_v0_generate.py`
- Test: `tests/test_run_v0.py`

**Interfaces:**
- Consumes : `src.terrains` (`sample_split`, `make_terrain_from_params`, `rest_state_ic`, `rest_residual`, `DROP_ICS`, `REST_SURFACE`) ; `src.solver.simulate` ; `src.io_utils` (`Dataset`, `save_dataset`, `save_animation`) ; `src.metrics.mass_series` ; `src.render.surface_height` ; `config`.
- Produces :
  - `generate_split(out_data_dir, out_fig_dir, grid, solver_cfg, entries, rest_surface=REST_SURFACE, save_extrap_anim=True) -> dict` — génère un `.npz` par (terrain, CI), applique les garde-fous (masse, positivité, résidu au repos), écrit `split.json`, sauvegarde les animations oracle η des terrains extrap, retourne un rapport `dict`.
  - constantes `POSITIVITY_MARGIN = 0.1`, `REST_SURF_DEV_TOL = 0.15`, `REST_SPEED_TOL = 0.5` (bornes de sécurité ; le jugement fin se fait sur les valeurs rapportées).
  - `main()` lançant la génération complète vers `data/v2/` et `outputs/v2/`.

- [ ] **Step 1 : Écrire le test qui échoue**

```python
# tests/test_run_v0.py
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import GridConfig, SolverConfig
from src.io_utils import load_dataset
from src.terrains import sample_split


def test_generate_split_writes_contract(tmp_path):
    from scripts.run_v0_generate import generate_split
    grid = GridConfig(H=64, W=64, dx=1.0, dy=1.0)
    # config réduite pour un test rapide : peu de pas, sous-ensemble de terrains
    solver = SolverConfig(cfl=0.45, n_steps=12, save_every=4, min_depth=1e-3)
    entries = [sample_split(grid)[0],                       # 1 train (bump)
               [e for e in sample_split(grid) if e.regime == "extrap_channel"][0]]
    data_dir = tmp_path / "data"
    fig_dir = tmp_path / "fig"
    report = generate_split(data_dir, fig_dir, grid, solver, entries,
                            save_extrap_anim=False)

    # contrat fichiers : un .npz par (terrain, CI)
    train0 = sample_split(grid)[0]
    npz = data_dir / f"{train0.terrain_id}__{train0.ic_ids[0]}.npz"
    assert npz.exists()
    ds = load_dataset(npz)
    assert ds.h.shape[1:] == (64, 64)
    assert ds.meta["regime"] == "train"
    assert "theta" in np.load(npz).files  # theta présent dans le .npz

    # split.json présent et structuré
    split = json.loads((data_dir / "split.json").read_text())
    assert "entries" in split and "rest_surface" in split
    roles = {e["regime"] for e in split["entries"]}
    assert {"train", "extrap_channel"} <= roles

    # garde-fous reportés et respectés
    assert report["max_mass_drift"] < 1e-7
    assert report["min_depth"] > 0.0
    assert all(np.isfinite(v) for v in report["rest_residual_surf"].values())
```

- [ ] **Step 2 : Lancer le test pour vérifier qu'il échoue**

Run: `.venv/bin/python -m pytest tests/test_run_v0.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'scripts.run_v0_generate'`).

- [ ] **Step 3 : Implémenter `scripts/run_v0_generate.py`**

```python
"""V0 — Génère le dataset v2 (famille de terrains × CI) et applique les garde-fous
d'oracle. La conservation de masse ne suffit pas : on vérifie aussi la positivité
avec marge (assèchement) et le résidu au repos (well-balancedness), et on rend les
animations oracle des terrains d'extrapolation pour le sanity visuel.

Usage : .venv/bin/python scripts/run_v0_generate.py
Sorties : data/v2/<terrain_id>__<ic_id>.npz, data/v2/split.json,
          outputs/v2/v0_oracle_<terrain>.gif (terrains extrap)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from config import GridConfig, SolverConfig
from src.terrains import (REST_SURFACE, DROP_ICS, sample_split,
                          make_terrain_from_params, rest_state_ic, rest_residual)
from src.solver import simulate
from src.io_utils import Dataset, save_dataset, save_animation
from src.metrics import mass_series
from src.render import surface_height

# ----------------------------- CONFIG ------------------------------------
GRID = GridConfig(H=64, W=64, dx=1.0, dy=1.0)
SOLVER = SolverConfig(cfl=0.45, n_steps=800, save_every=4, min_depth=1e-3)
OUT_DATA = ROOT / "data" / "v2"
OUT_FIG = ROOT / "outputs" / "v2"
POSITIVITY_MARGIN = 0.1   # min(h) doit rester > cette marge sur toute la trajectoire
# Bornes de SÉCURITÉ du résidu au repos (well-balancedness) : elles attrapent une
# vraie rupture (instabilité / courants parasites massifs qui invalident l'oracle),
# pas la non-well-balancedness légère. Le jugement fin « assez petit ? » se fait sur
# les VALEURS rapportées (report["rest_residual_*"]) + les animations oracle extrap.
REST_SURF_DEV_TOL = 0.15  # ~10% de REST_SURFACE
REST_SPEED_TOL = 0.5
# -------------------------------------------------------------------------


def generate_split(out_data_dir, out_fig_dir, grid: GridConfig, solver_cfg: SolverConfig,
                   entries, rest_surface: float = REST_SURFACE,
                   save_extrap_anim: bool = True) -> dict:
    """Génère un .npz par (terrain, CI), applique les garde-fous, écrit split.json,
    rend les animations oracle des terrains extrap. Retourne un rapport."""
    out_data_dir = Path(out_data_dir)
    out_fig_dir = Path(out_fig_dir)
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    report = {"max_mass_drift": 0.0, "min_depth": np.inf,
              "rest_residual_surf": {}, "rest_residual_speed": {}, "trajectories": []}
    split_entries = []

    for e in entries:
        b = make_terrain_from_params(grid, e.params)
        assert rest_surface - float(b.max()) >= 0.0, f"{e.terrain_id}: terrain non submergé"

        # garde-fou de well-balancedness (une fois par terrain)
        surf_dev, speed = rest_residual(grid, b, solver_cfg, rest_surface, n_steps=50)
        report["rest_residual_surf"][e.terrain_id] = surf_dev
        report["rest_residual_speed"][e.terrain_id] = speed
        assert surf_dev < REST_SURF_DEV_TOL, (
            f"{e.terrain_id}: résidu de surface au repos {surf_dev:.3e} "
            f">= {REST_SURF_DEV_TOL} (bathymétrie trop raide)")
        assert speed < REST_SPEED_TOL, (
            f"{e.terrain_id}: vitesse parasite au repos {speed:.3e} >= {REST_SPEED_TOL}")

        for ic_id in e.ic_ids:
            h0, u0, v0 = rest_state_ic(grid, b, **DROP_ICS[ic_id], rest_surface=rest_surface)
            hs, us, vs, dt = simulate(h0, u0, v0, b, grid, solver_cfg)

            masses = mass_series(hs, grid.dx, grid.dy)
            drift = float((np.abs(masses - masses[0]) / masses[0]).max())
            min_depth = float(hs.min())
            assert drift < 1e-7, f"{e.terrain_id}__{ic_id}: dérive masse {drift:.2e}"
            assert min_depth > POSITIVITY_MARGIN, (
                f"{e.terrain_id}__{ic_id}: min(h)={min_depth:.3f} <= "
                f"{POSITIVITY_MARGIN} (assèchement)")
            report["max_mass_drift"] = max(report["max_mass_drift"], drift)
            report["min_depth"] = min(report["min_depth"], min_depth)
            report["trajectories"].append(f"{e.terrain_id}__{ic_id}")

            meta = {"dx": grid.dx, "dy": grid.dy, "dt": dt, "schema": "lax-friedrichs",
                    "cfl": solver_cfg.cfl, "terrain_id": e.terrain_id, "ic_id": ic_id,
                    "regime": e.regime, "role": e.role, "rest_surface": rest_surface}
            ds = Dataset(hs, us, vs, b, meta)
            path = out_data_dir / f"{e.terrain_id}__{ic_id}.npz"
            # save_dataset n'écrit pas theta : on l'ajoute via un re-dump complet
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path, h=hs.astype(np.float64), u=us.astype(np.float64),
                v=vs.astype(np.float64), b=b.astype(np.float64),
                theta=e.params.to_vector(), meta_json=np.array(json.dumps(meta)))
            print(f"[V0] {e.terrain_id:24s} {ic_id:11s} regime={e.regime:16s} "
                  f"drift={drift:.2e} min_h={min_depth:.3f}")

            if save_extrap_anim and e.role == "holdout_extrap":
                eta = surface_height(hs, b)
                written = save_animation(
                    out_fig_dir / f"v0_oracle_{e.terrain_id}.gif", eta, fps=20,
                    cmap="viridis", title=f"V0 oracle η — {e.terrain_id} ({e.regime})")
                print(f"[V0] animation oracle extrap -> {written}")

        split_entries.append({"terrain_id": e.terrain_id, "role": e.role,
                              "regime": e.regime, "params": e.params.to_dict(),
                              "theta": e.params.to_vector().tolist(),
                              "ic_ids": list(e.ic_ids)})

    split = {"rest_surface": rest_surface, "grid": {"H": grid.H, "W": grid.W,
             "dx": grid.dx, "dy": grid.dy}, "entries": split_entries}
    (out_data_dir / "split.json").write_text(json.dumps(split, indent=2))
    report["min_depth"] = float(report["min_depth"])
    return report


def main() -> None:
    entries = sample_split(GRID)
    report = generate_split(OUT_DATA, OUT_FIG, GRID, SOLVER, entries)
    print(f"\n[V0] {len(report['trajectories'])} trajectoires générées.")
    print(f"[V0] dérive de masse max = {report['max_mass_drift']:.2e}")
    print(f"[V0] profondeur min sur tout le dataset = {report['min_depth']:.3f} "
          f"(marge {POSITIVITY_MARGIN})")
    print(f"[V0] résidu de surface au repos max = "
          f"{max(report['rest_residual_surf'].values()):.3e} (tol {REST_SURF_DEV_TOL})")


if __name__ == "__main__":
    main()
```

> Note : `save_dataset` du POC n'écrit pas `theta` ; ici on écrit le `.npz` directement (même format `meta_json`) en ajoutant la clé `theta`, conforme au contrat §5. `load_dataset` du POC relit `h,u,v,b,meta` sans toucher `theta` (compatible).

- [ ] **Step 4 : Lancer le test pour vérifier qu'il passe**

Run: `.venv/bin/python -m pytest tests/test_run_v0.py -q`
Expected: PASS (1 test).

- [ ] **Step 5 : Générer le dataset complet et vérifier les garde-fous**

Run: `.venv/bin/python scripts/run_v0_generate.py`
Expected: ~21 lignes `[V0] …`, dérive de masse < 1e-7, profondeur min > 0.1, résidu de surface au repos < `REST_SURF_DEV_TOL` (0.15), 2 animations oracle extrap écrites dans `outputs/v2/`. **Si un assert se déclenche** (assèchement, résidu trop grand, dérive), ne pas contourner : c'est un signal que le terrain sort du régime valide → rapporter le terrain fautif (ajuster ses bornes de géométrie dans `sample_split`, en restant submergé).

- [ ] **Step 6 : Commit**

```bash
git add scripts/run_v0_generate.py tests/test_run_v0.py
git commit -m "feat(v2): génération dataset V0 + garde-fous oracle (masse/positivité/repos)"
```

---

### Task 5 : `src/pod.py` — POD agnostique au nombre de canaux (extension rétro-compatible)

**Files:**
- Modify: `src/pod.py`
- Test: `tests/test_pod.py` (ajout)

**Interfaces:**
- Consumes : (interne au module).
- Produces :
  - `_channel_scale(X, n_channels=3)` — défaut 3 = comportement POC inchangé.
  - `fit_pod(X, energy_threshold, max_modes, n_channels=3)` — paramètre `n_channels` ajouté en fin de signature (rétro-compatible : tous les appels existants gardent 3 canaux).
  - `stack_height(h_seq) -> np.ndarray` — `(T,H,W) -> (H*W, T)`.
  - `unstack_height(X, H, W) -> np.ndarray` — `(H*W, n) -> (n,H,W)`.

- [ ] **Step 1 : Écrire les tests qui échouent**

```python
# tests/test_pod.py  (append)
import numpy as np

from src.pod import (fit_pod, encode, decode, _channel_scale,
                     stack_height, unstack_height)


def test_channel_scale_single_channel_is_global_std():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 3.0, size=(100, 20))
    scale = _channel_scale(X, n_channels=1)
    assert np.allclose(scale, scale[0])              # un seul bloc -> scale uniforme
    assert np.isclose(scale[0], X.std())             # = écart-type global


def test_channel_scale_default_is_three_blocks_unchanged():
    rng = np.random.default_rng(1)
    X = np.concatenate([rng.normal(0, 1, (30, 10)),
                        rng.normal(0, 5, (30, 10)),
                        rng.normal(0, 9, (30, 10))], axis=0)
    scale = _channel_scale(X)  # défaut n_channels=3, comportement POC
    assert not np.isclose(scale[0], scale[40])       # blocs distincts
    assert not np.isclose(scale[40], scale[70])


def test_height_pod_roundtrip():
    rng = np.random.default_rng(2)
    H = W = 8
    T = 40
    h_seq = rng.normal(1.0, 0.2, size=(T, H, W))
    X = stack_height(h_seq)
    assert X.shape == (H * W, T)
    basis = fit_pod(X, energy_threshold=0.999, max_modes=64, n_channels=1)
    z = encode(basis, X)
    Xr = decode(basis, z)
    seq_r = unstack_height(Xr, H, W)
    assert seq_r.shape == (T, H, W)
    # reconstruction fidèle au seuil d'énergie choisi
    assert np.linalg.norm(seq_r - h_seq) / np.linalg.norm(h_seq) < 0.05
```

- [ ] **Step 2 : Lancer les tests pour vérifier qu'ils échouent**

Run: `.venv/bin/python -m pytest tests/test_pod.py -q`
Expected: FAIL (`ImportError: cannot import name 'stack_height'` et signature `_channel_scale`).

- [ ] **Step 3 : Étendre `src/pod.py`**

Modifier `_channel_scale` (remplacer le `3` codé en dur par `n_channels`) :

```python
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
```

Modifier `fit_pod` pour propager `n_channels` :

```python
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
```

Ajouter en fin de module les helpers hauteur :

```python
def stack_height(h_seq: np.ndarray) -> np.ndarray:
    """(T,H,W) -> X (H*W, T) : snapshots de hauteur seule (un canal)."""
    T, H, W = h_seq.shape
    return h_seq.reshape(T, H * W).T


def unstack_height(X: np.ndarray, H: int, W: int) -> np.ndarray:
    """X (H*W, n) -> (n,H,W) : inverse de stack_height."""
    n = X.shape[1]
    return X.T.reshape(n, H, W)
```

- [ ] **Step 4 : Lancer les tests pour vérifier qu'ils passent**

Run: `.venv/bin/python -m pytest tests/test_pod.py -q`
Expected: PASS (tests existants + 3 nouveaux).

- [ ] **Step 5 : Vérifier la non-régression complète du POC**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (35 anciens + nouveaux V2, aucune régression). Si un ancien test échoue, c'est que l'extension a changé le comportement par défaut → corriger.

- [ ] **Step 6 : Commit**

```bash
git add src/pod.py tests/test_pod.py
git commit -m "feat(v2): POD agnostique au nombre de canaux (n_channels) + helpers hauteur"
```

---

### Task 6 : `scripts/run_v1_representation.py` — plafond de représentation inter-terrain (pas décisif)

**Files:**
- Create: `scripts/run_v1_representation.py`
- Create: `docs/v2_V1_representation_ceiling.md` (écrit par le script avec les chiffres mesurés)
- Test: `tests/test_run_v1.py`

**Interfaces:**
- Consumes : `src.pod` (`fit_pod`, `encode`, `decode`, `stack_height`, `unstack_height`, `cumulative_energy`) ; `src.metrics.relative_l2_error`, `error_growth` ; `src.io_utils.load_dataset` ; `config` ; `matplotlib`.
- Produces :
  - `representation_ceiling(train_h_seqs, holdout_h_seqs, H, W, energy_threshold, max_modes) -> dict` — construit la base POD hauteur sur les trajectoires train, encode-décode chaque holdout, retourne `{"k": int, "energy_at_k": float, "train_err": float, "regimes": {regime: {"err": float, "err_max": float}}}`.
  - `main()` — charge `data/v2/`, calcule le plafond, écrit la figure `outputs/v2/v1_representation_ceiling.png` et la note `docs/v2_V1_representation_ceiling.md`.

- [ ] **Step 1 : Écrire le test qui échoue**

```python
# tests/test_run_v1.py
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_representation_ceiling_separates_in_and_out_of_family():
    from scripts.run_v1_representation import representation_ceiling
    rng = np.random.default_rng(0)
    H = W = 8
    T = 30
    # famille de rang 2 : deux motifs spatiaux fixes, amplitudes variables dans le temps
    p1 = rng.normal(size=(H, W)); p2 = rng.normal(size=(H, W))

    def traj(seed):
        r = np.random.default_rng(seed)
        a = r.normal(size=(T, 1, 1)); b = r.normal(size=(T, 1, 1))
        return 1.0 + a * p1 + b * p2

    train = [traj(s) for s in (1, 2, 3)]
    in_family = traj(99)                       # même sous-espace -> reconstruction quasi exacte
    out_family = 1.0 + rng.normal(size=(T, H, W))  # bruit plein rang -> mal reconstruit
    res = representation_ceiling(train, {"interp": in_family, "extrap": out_family},
                                 H, W, energy_threshold=0.9999, max_modes=64)
    assert res["k"] >= 2
    assert res["regimes"]["interp"]["err"] < 0.05      # dans la famille : plafond bas
    assert res["regimes"]["extrap"]["err"] > res["regimes"]["interp"]["err"]  # hors famille : pire
```

- [ ] **Step 2 : Lancer le test pour vérifier qu'il échoue**

Run: `.venv/bin/python -m pytest tests/test_run_v1.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'scripts.run_v1_representation'`).

- [ ] **Step 3 : Implémenter `scripts/run_v1_representation.py`**

```python
"""V1 — Plafond de représentation inter-terrain (le pas décisif).

Base POD de HAUTEUR SEULE construite sur les terrains d'entraînement, puis
encode-décode du h VRAI des terrains holdout (interp + extrap). Aucune dynamique.
Tranche : la base statique span-t-elle un terrain nouveau ? Plafond bas (<~10%) ->
conditionner la dynamique (V2/V3b). Plafond haut (>~30%, surtout extrap) ->
conditionner la représentation (V3a, voire encodeur V5).

Usage : .venv/bin/python scripts/run_v1_representation.py
Sorties : outputs/v2/v1_representation_ceiling.png, docs/v2_V1_representation_ceiling.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.pod import (fit_pod, encode, decode, stack_height, unstack_height,
                     cumulative_energy)
from src.metrics import relative_l2_error, error_growth
from src.io_utils import load_dataset

DATA = ROOT / "data" / "v2"
OUT_FIG = ROOT / "outputs" / "v2"
OUT_DOC = ROOT / "docs" / "v2_V1_representation_ceiling.md"
ENERGY_THRESHOLD = 0.9999
MAX_MODES = 2000  # cap volontairement élevé : laisser le seuil d'énergie gouverner k


def representation_ceiling(train_h_seqs, holdout_h_seqs, H: int, W: int,
                           energy_threshold: float, max_modes: int) -> dict:
    """Construit la base POD hauteur sur les trajectoires train, encode-décode chaque
    holdout. Retourne k, énergie, erreur train (plancher) et erreur par régime."""
    X_train = np.concatenate([stack_height(s) for s in train_h_seqs], axis=1)
    basis = fit_pod(X_train, energy_threshold, max_modes, n_channels=1)
    k = basis.Phi.shape[1]
    energy = cumulative_energy(basis.singular_values)
    energy_at_k = float(energy[k - 1])

    # plancher in-sample : reconstruction des données d'entraînement
    train_err = relative_l2_error(decode(basis, encode(basis, X_train)), X_train)

    regimes = {}
    for regime, h_seq in holdout_h_seqs.items():
        X = stack_height(h_seq)
        Xr = decode(basis, encode(basis, X))
        seq_r = unstack_height(Xr, H, W)
        regimes[regime] = {
            "err": relative_l2_error(seq_r, h_seq),
            "err_max": float(error_growth(seq_r, h_seq).max()),
        }
    return {"k": k, "energy_at_k": energy_at_k, "train_err": float(train_err),
            "regimes": regimes}


def _load_split():
    split = json.loads((DATA / "split.json").read_text())
    H, W = split["grid"]["H"], split["grid"]["W"]
    train, holdout = [], {}
    for e in split["entries"]:
        for ic in e["ic_ids"]:
            ds = load_dataset(DATA / f"{e['terrain_id']}__{ic}.npz")
            if e["role"] == "train":
                train.append(ds.h)
            else:
                holdout[e["regime"]] = ds.h  # un holdout par régime (1 CI chacun)
    return H, W, train, holdout


def _render_figure(res: dict, H: int, W: int, holdout_h: dict, basis_seqs: dict):
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    regimes = list(res["regimes"].keys())
    fig = plt.figure(figsize=(4 + 3 * len(regimes), 6))
    # barres : plancher train + un bar par régime
    ax = fig.add_subplot(2, 1, 1)
    labels = ["train (réf)"] + regimes
    vals = [res["train_err"]] + [res["regimes"][r]["err"] for r in regimes]
    ax.bar(labels, vals, color=["#888"] + ["#3a7"] * len(regimes))
    ax.set_ylabel("erreur L2 relative (h)")
    ax.set_title(f"V1 — plafond de représentation hauteur (k={res['k']}, "
                 f"énergie={res['energy_at_k']:.5f})")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    # champs résiduels : dernière frame, un régime extrap représentatif
    show = [r for r in regimes if r.startswith("extrap")][:2] or regimes[:1]
    for j, r in enumerate(show):
        true = holdout_h[r][-1]
        rec = basis_seqs[r][-1]
        for col, (field, ttl) in enumerate(
                [(true, "vérité"), (rec, "reconstruit"), (np.abs(rec - true), "|résidu|")]):
            a = fig.add_subplot(2, len(show) * 3, len(show) * 3 + j * 3 + col + 1)
            a.imshow(field, origin="lower", cmap="viridis")
            a.set_title(f"{r}\n{ttl}", fontsize=8); a.axis("off")
    fig.tight_layout()
    out = OUT_FIG / "v1_representation_ceiling.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def main() -> None:
    H, W, train, holdout = _load_split()
    res = representation_ceiling(train, holdout, H, W, ENERGY_THRESHOLD, MAX_MODES)

    # reconstruire les séquences holdout pour la figure (résidus)
    X_train = np.concatenate([stack_height(s) for s in train], axis=1)
    basis = fit_pod(X_train, ENERGY_THRESHOLD, MAX_MODES, n_channels=1)
    basis_seqs = {r: unstack_height(decode(basis, encode(basis, stack_height(h))), H, W)
                  for r, h in holdout.items()}
    fig_path = _render_figure(res, H, W, holdout, basis_seqs)

    print(f"[V1] k={res['k']} (énergie {res['energy_at_k']:.6f}), "
          f"erreur train (plancher) = {res['train_err']:.4f}")
    for r, m in res["regimes"].items():
        print(f"[V1] {r:18s} err={m['err']:.4f}  err_max={m['err_max']:.4f}")

    # verdict automatique sur le seuil de représentation
    extrap_errs = [m["err"] for r, m in res["regimes"].items() if r.startswith("extrap")]
    worst_extrap = max(extrap_errs) if extrap_errs else 0.0
    ch = res["regimes"].get("extrap_channel", {}).get("err")  # signal porteur (topo neuve)
    if worst_extrap < 0.10:
        verdict = ("Plafond BAS (<10% même en extrapolation) : la base statique span "
                   "le terrain nouveau dans le régime submergé. Défaut probable = la "
                   "DYNAMIQUE -> conditionner l'opérateur (V2 baseline puis V3b).")
    elif worst_extrap < 0.30:
        verdict = ("Plafond INTERMÉDIAIRE (10–30% en extrapolation) : la base tient "
                   "l'interpolation mais peine en extrapolation -> évaluer V2, puis "
                   "conditionner la représentation (V3a : b en canal) si nécessaire.")
    else:
        verdict = ("Plafond HAUT (>30% en extrapolation). NE PAS conclure tout de suite "
                   "'base fondamentalement terrain-spécifique -> encodeur appris'. Avec "
                   "~9 terrains la famille est sous-échantillonnée (cf. plafond vitesse "
                   "du POC, en partie data-limité). CHECK BON MARCHÉ D'ABORD : le plafond "
                   "baisse-t-il en ajoutant des terrains d'entraînement ? S'il baisse -> "
                   "data-limité (ajouter des terrains, garder la base statique). S'il "
                   "tient -> structurel (alors V3a / encodeur V5 justifié).")

    # corollaire : extrap_channel (topologie jamais vue) donne ses dents au diagnostic ;
    # interp/extrap_obstacle restent du régime de réfraction douce.
    if ch is not None:
        verdict += (f" Signal porteur : extrap_channel (topologie jamais vue) = {ch:.4f}"
                    f" ; ne pas conclure 'ça généralise' sur les seuls cas de réfraction"
                    f" douce (interp/extrap_obstacle).")

    # portée du résultat (toujours écrite, pour calibrer le ✅)
    scope = ("Tous les terrains sont SUBMERGÉS : la dépendance au terrain passe par la "
             "réfraction (contraste de célérité ~1.7×, réel mais modéré). Un V1 qui "
             "passe certifie la généralisation DANS le régime submergé/réfraction — pas "
             "sur tout terrain de jeu. Le sec / les îles (sillages, séparation) sont un "
             "régime distinct plus dur, reporté en v2.5 (solveur mouillé/sec "
             "positivity-preserving & well-balanced).")

    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# V1 — Plafond de représentation inter-terrain (hauteur)", "",
             f"Base POD hauteur seule (n_channels=1), seuil d'énergie {ENERGY_THRESHOLD}.",
             "",
             f"- **k = {res['k']}** (énergie cumulée {res['energy_at_k']:.6f}). "
             f"Pour mémoire, le POC mono-terrain donnait k≈43 ; un k plus grand ici "
             f"est attendu et mesure la complexité accrue de la famille.",
             f"- Erreur de reconstruction train (plancher in-sample) : "
             f"{res['train_err']:.4f}.", "",
             "## Plafond par régime (erreur L2 relative de h)", "",
             "| régime | err | err_max (par frame) |", "|---|---|---|"]
    for r, m in res["regimes"].items():
        lines.append(f"| {r} | {m['err']:.4f} | {m['err_max']:.4f} |")
    lines += ["", "## Verdict", "", verdict, "", "## Portée du résultat (calibrage du ✅)",
              "", scope, "", f"Figure : `{fig_path.relative_to(ROOT)}`."]
    OUT_DOC.write_text("\n".join(lines))
    print(f"[V1] figure -> {fig_path}")
    print(f"[V1] note   -> {OUT_DOC}")
    print(f"[V1] VERDICT : {verdict}")
    print(f"[V1] PORTÉE  : {scope}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4 : Lancer le test pour vérifier qu'il passe**

Run: `.venv/bin/python -m pytest tests/test_run_v1.py -q`
Expected: PASS (1 test).

- [ ] **Step 5 : Lancer V1 sur le dataset réel et lire le chiffre décisif**

Run: `.venv/bin/python scripts/run_v1_representation.py`
Expected: affiche `k`, l'erreur train (plancher), puis err/err_max pour `interp`, `extrap_obstacle`, `extrap_channel`, écrit la figure et la note avec le verdict. **C'est le chiffre qui oriente V2–V4** : ne pas le maquiller. Reporter `k` tel quel (un `k` élevé est lui-même un résultat).

- [ ] **Step 6 : Commit**

```bash
git add scripts/run_v1_representation.py tests/test_run_v1.py docs/v2_V1_representation_ceiling.md outputs/v2/v1_representation_ceiling.png
git commit -m "feat(v2): V1 — plafond de représentation inter-terrain (interp/extrap), verdict chiffré"
```

---

## Self-Review (rempli par l'auteur du plan)

**1. Couverture de la spec (V0+V1) :**
- §6 V0 « famille paramétrée + split + masse conservée + reproductible » → Tasks 1–4 (générateurs, `sample_split` déterministe, garde-fou masse, `split.json`). ✓
- §6 V1 « POD train, encode-décode holdout, erreur par terrain interp/extrap, aucune dynamique » → Tasks 5–6. ✓
- §5 contrat données (`<terrain>__<ic>.npz` avec `h,u,v,b,theta` + meta, `split.json`) → Task 4. ✓
- Annexe A.2 (CI repos submergé), A.3 (extrap par géométrie), A.6 (garde-fous oracle) → Tasks 1, 2, 4. ✓
- Non-régression POC (35 tests, défaut `n_channels=3`) → Task 5 step 5. ✓
- Distinction interp/extrap dans métriques et figure → Task 6. ✓

**2. Placeholders :** aucun « TBD/TODO » ; code complet à chaque step. Seul point d'attention documenté : si un garde-fou V0 se déclenche (Task 4 step 5), ajuster les bornes de géométrie en restant submergé — c'est une boucle de diagnostic prévue, pas un placeholder.

**3. Cohérence des types :** `TerrainParams`/`SplitEntry`/`DROP_ICS` (Tasks 1–2) consommés tels quels par Task 4 ; `fit_pod(..., n_channels=1)`, `stack_height`/`unstack_height` (Task 5) consommés par Task 6. `simulate(h0,u0,v0,b,grid,cfg)->(hs,us,vs,dt)` et `load_dataset` conformes au POC.

---

## Execution Handoff

Plan sauvegardé. V2–V4 seront planifiés **après** le chiffre de V1 (le contenu de V3 en dépend — cf. spec A.5). Exécution recommandée : **subagent-driven-development** (un subagent implémenteur par tâche + revue spec/qualité entre tâches), comme pour le POC.
