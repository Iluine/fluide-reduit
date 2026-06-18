# POC Fluide Réduit Appris (shallow-water 2D) — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construire un POC frugal CPU qui remplace un solveur shallow-water 2D par un modèle d'ordre réduit appris (POD + DMD), et qui *tranche* trois hypothèses (H1 colonne vertébrale, H2 dérive temporelle, H3 multirésolution) avec métriques + figures.

**Architecture:** Pipeline `champ (h,u,v) → encode POD → latent z → opérateur DMD → z suivant → decode POD → champ → rendu`. Le cœur reste linéaire (POD/DMD) sur CPU ; PyTorch/GPU (M4/M5) est derrière une porte de décision, activé seulement si la baseline linéaire est insuffisante. Chaque jalon est un script exécutable seul qui produit un chiffre + une figure.

**Tech Stack:** Python 3.12, `numpy` / `scipy` / `matplotlib` (cœur CPU), `pytest` (tests, dev), `pillow` optionnel (export GIF, fallback montage PNG pur-matplotlib si absent), `torch` (M4/M5 uniquement, GPU CUDA RTX 3050 Ti ≤ 4 Go, déféré).

## Global Constraints

- **Python ≥ 3.10** (présent : 3.12.3). Environnement isolé dans `.venv` local du projet.
- **Dépendances cœur strictement limitées** à `numpy`, `scipy`, `matplotlib`. `pytest` autorisé (dev/test uniquement). `pillow` traité comme **optionnel** (le code doit fonctionner sans, via fallback montage PNG). **`torch` interdit avant M4** et derrière une porte de décision. Demander avant toute autre dépendance.
- **GPU** : RTX 3050 Ti Laptop, **4096 MiB de VRAM**. Cible **≤ 4 Go**, AMP si besoin. Réservé à M4/M5 ; le cœur reste CPU.
- **Grille par défaut 64×64**, configurable. Doit tourner en **minutes** sur laptop CPU.
- **Réduction d'abord** : POD + DMD avant tout réseau. N'introduire M4/M5 que si M2/M3 le justifient.
- **Code typé** (type hints) sur les interfaces publiques, docstrings courtes, **contrats de formes/dtypes documentés**, pas de cleverness. Le relecteur est un dev senior qui ne produit pas de Python.
- **Contrats de données (§5 de la spec), à respecter verbatim** :
  - `h, u, v` : `(T, H, W)` `float64` ; terrain `b` : `(H, W)` `float64`. État empilé `state` : `(T, C=3, H, W)`.
  - Snapshots POD : `X` de forme `(n_features, n_snapshots)`, `n_features = C*H*W` ; modes `Phi` `(n_features, k)` ; `mean` `(n_features,)`. **Extension documentée** : un vecteur `scale` `(n_features,)` (écart-type par canal, diffusé) pour équilibrer h/u/v ; `X ≈ scale * (Phi @ z) + mean[:, None]`.
  - Latent : `z` `(k, T)`. DMD : `A` `(k, k)`, `z[:, t+1] ≈ A @ z[:, t]`.
  - Datasets : `data/ground_truth/<nom>.npz` contenant `h,u,v,b` + métadonnées (`dx, dt, ci, schema, cfl`).
- **Convention d'indexation des tableaux** : `array[y, x]` — axe 0 = lignes = y (hauteur `H`), axe 1 = colonnes = x (largeur `W`). `u` = vitesse selon x (axe 1), `v` = vitesse selon y (axe 0). Cette convention est globale et documentée dans chaque module.
- **Sorties** : données `.npz` dans `data/`, figures `.png` et animations `.gif` (ou montage `.png`) dans `outputs/`. Toutes les figures clés listées dans le `README.md`.
- **Invariants instrumentés** : conservation de masse (assert + tracé), respect CFL (assert), erreur de reconstruction POD vs `k` (tracé), croissance d'erreur de rollout (tracé), saut de couture multirésolution (tracé).

---

## File Structure

```
pocPhysicator/                      # racine = répertoire de travail courant
  README.md                         # commandes par jalon + interprétation attendue
  requirements.txt
  .gitignore
  config.py                         # dataclasses de config partagée (constantes physiques + défauts)
  src/
    __init__.py
    io_utils.py                     # save/load dataset .npz, export animation (gif/montage)
    metrics.py                      # masse, erreur L2 relative, croissance d'erreur, saut de couture
    solver.py                       # M0 : oracle shallow-water (Lax-Friedrichs conservatif)
    pod.py                          # M1 : base réduite POD (SVD)
    dmd.py                          # M2 : dynamique latente linéaire (DMD)
    multiresolution.py              # M6 : grossier global + fenêtre fine mobile
    render.py                       # M7 : heatmap + champ de hauteur
    # dynamics_nn.py                # M4 (déféré, derrière porte de décision)
    # physics_loss.py               # M5 (déféré)
  scripts/
    run_m0_generate.py              # génère les rollouts oracle (vus + test)
    run_m1_pod.py                   # énergie cumulée, erreur recon, modes spatiaux
    run_m2_dmd.py                   # fit DMD, rollout court vs vérité
    run_m3_eval_rollout.py          # H2 : croissance d'erreur + dérive masse, CI vue + test
    run_m6_multiresolution.py       # H3 : saut de couture vs temps, fenêtre mobile
    run_m7_render.py                # rendu de l'état prédit
  tests/
    __init__.py
    test_io_utils.py
    test_metrics.py
    test_solver.py
    test_pod.py
    test_dmd.py
    test_multiresolution.py
  data/ground_truth/                # .npz oracle (gitignored)
  outputs/                          # figures + animations (gitignored)
  docs/superpowers/plans/           # ce plan
```

**Responsabilités (une par fichier) :** `config.py` = constantes + défauts ; `io_utils.py` = persistance & animations ; `metrics.py` = diagnostics chiffrés (sans dépendance aux autres modules src) ; `solver.py` = oracle physique ; `pod.py` = réduction spatiale ; `dmd.py` = dynamique temporelle linéaire ; `multiresolution.py` = représentation spatiale à deux niveaux ; `render.py` = relevé visuel de l'état. Les scripts `run_m*` ne contiennent que de l'orchestration + un **bloc de config en tête**.

**Ordre des dépendances entre modules :** `config` ← `io_utils`, `metrics` (feuilles) ← `solver` ← `pod` ← `dmd`. `multiresolution` et `render` dépendent de `config`/`metrics`/`io_utils`. `metrics.seam_jump` prend des entiers `(i0, j0, size)` (pas le type `Window`) pour rester découplé de `multiresolution`.

---

## Task 0 : Échafaudage du projet (env, structure, config, git)

**Files:**
- Create: `requirements.txt`, `.gitignore`, `config.py`, `src/__init__.py`, `tests/__init__.py`
- Create: dossiers `data/ground_truth/`, `outputs/`

**Interfaces:**
- Consumes: rien.
- Produces: `config.py` exporte `GRAVITY: float` et les dataclasses `GridConfig`, `SolverConfig`, `PODConfig` + instances par défaut `GRID`, `SOLVER`, `POD`. Champs :
  - `GridConfig(H=64, W=64, dx=1.0, dy=1.0)`
  - `SolverConfig(cfl=0.45, n_steps=400, save_every=1, min_depth=1e-3)`
  - `PODConfig(energy_threshold=0.99, max_modes=128)`

- [ ] **Step 1 : Créer le venv et installer le cœur (via `uv`)**

> Note environnement : `python3-venv`/`pip` système sont absents et il n'y a pas
> de sudo. On utilise **`uv`** (installé dans `~/.local/bin`) pour créer le venv
> et installer les paquets. `uv venv` ne met pas `pip` dans le venv (inutile :
> on lance tout via `.venv/bin/python -m ...`, les modules étant installés par uv).

```bash
export PATH="$HOME/.local/bin:$PATH"   # uv est dans ~/.local/bin
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python "numpy>=1.26" "scipy>=1.11" "matplotlib>=3.8" "pytest>=8.0"
```

- [ ] **Step 2 : Écrire `requirements.txt`**

```text
# Cœur (CPU) — POC fluide réduit
numpy>=1.26
scipy>=1.11
matplotlib>=3.8

# Tests (dev uniquement)
pytest>=8.0

# Optionnel : export GIF des animations. Sans pillow, fallback montage PNG (matplotlib seul).
# pillow>=10.0

# M4/M5 uniquement (déféré, GPU CUDA RTX 3050 Ti <=4 Go, AMP). NE PAS installer avant M4.
# torch>=2.2
```

- [ ] **Step 3 : Écrire `.gitignore`**

```text
.venv/
__pycache__/
*.pyc
data/
outputs/
.pytest_cache/
```

- [ ] **Step 4 : Écrire `config.py`**

```python
"""Configuration partagée du POC (constantes physiques + valeurs par défaut).

Les scripts run_m*.py définissent leur propre bloc de config en tête en
s'appuyant sur ces dataclasses. Indexation des tableaux : array[y, x]
(axe 0 = y = lignes = H, axe 1 = x = colonnes = W)."""
from __future__ import annotations

from dataclasses import dataclass

GRAVITY: float = 9.81  # m/s^2


@dataclass(frozen=True)
class GridConfig:
    """Grille régulière. dx, dy = pas d'espace (mêmes unités que les vitesses)."""
    H: int = 64
    W: int = 64
    dx: float = 1.0
    dy: float = 1.0


@dataclass(frozen=True)
class SolverConfig:
    """Paramètres du solveur oracle (M0)."""
    cfl: float = 0.45        # nombre de Courant cible (<1 pour Lax-Friedrichs)
    n_steps: int = 400       # nombre de pas de temps simulés
    save_every: int = 1      # sous-échantillonnage temporel des snapshots
    min_depth: float = 1e-3  # plancher de hauteur pour les divisions (vitesses/flux)


@dataclass(frozen=True)
class PODConfig:
    """Paramètres de la base réduite (M1)."""
    energy_threshold: float = 0.99  # fraction d'énergie cumulée pour choisir k
    max_modes: int = 128            # borne dure sur le nombre de modes


GRID = GridConfig()
SOLVER = SolverConfig()
POD = PODConfig()
```

- [ ] **Step 5 : Créer les fichiers `__init__.py` vides et les dossiers**

```bash
touch src/__init__.py tests/__init__.py
mkdir -p data/ground_truth outputs
```

- [ ] **Step 6 : Vérifier que la config s'importe et que pytest tourne**

Run :
```bash
.venv/bin/python -c "import config; print(config.GRID, config.SOLVER.cfl, config.GRAVITY)"
.venv/bin/python -m pytest -q
```
Expected : la 1re commande imprime `GridConfig(H=64, W=64, dx=1.0, dy=1.0) 0.45 9.81` ; pytest affiche `no tests ran` (sortie 5) sans erreur d'import.

- [ ] **Step 7 : Init git + branche de travail + commit**

```bash
git init
git checkout -b poc-impl
git add requirements.txt .gitignore config.py src/__init__.py tests/__init__.py
git commit -m "chore: scaffolding venv, config partagée, structure du dépôt"
```

---

## Task 1 : `io_utils` — persistance dataset + animations

**Files:**
- Create: `src/io_utils.py`
- Test: `tests/test_io_utils.py`

**Interfaces:**
- Consumes: rien (feuille).
- Produces:
  - `@dataclass class Dataset` avec champs `h, u, v: np.ndarray (T,H,W) float64`, `b: np.ndarray (H,W) float64`, `meta: dict`.
  - `save_dataset(path: str | Path, ds: Dataset) -> None` — écrit un `.npz` (clés `h,u,v,b` + `meta_json` = JSON du dict).
  - `load_dataset(path: str | Path) -> Dataset`.
  - `save_animation(path: str | Path, frames: np.ndarray, *, fps: int = 20, cmap: str = "viridis", title: str = "", vmin: float | None = None, vmax: float | None = None) -> str` — `frames` `(T,H,W)`. Tente un GIF via Pillow ; si Pillow absent, écrit un montage PNG (matplotlib seul) au même chemin avec extension `.png`. Retourne le chemin réellement écrit.

- [ ] **Step 1 : Écrire le test (roundtrip dataset + animation sans erreur)**

```python
# tests/test_io_utils.py
import numpy as np
from pathlib import Path

from src.io_utils import Dataset, save_dataset, load_dataset, save_animation


def test_dataset_roundtrip(tmp_path: Path):
    T, H, W = 5, 8, 8
    rng = np.random.default_rng(0)
    ds = Dataset(
        h=rng.random((T, H, W)),
        u=rng.random((T, H, W)),
        v=rng.random((T, H, W)),
        b=rng.random((H, W)),
        meta={"dx": 1.0, "dt": 0.1, "ci": "drop", "schema": "lax-friedrichs", "cfl": 0.45},
    )
    p = tmp_path / "d.npz"
    save_dataset(p, ds)
    out = load_dataset(p)
    assert np.allclose(out.h, ds.h)
    assert np.allclose(out.b, ds.b)
    assert out.h.dtype == np.float64
    assert out.meta["ci"] == "drop"
    assert out.meta["dt"] == 0.1


def test_save_animation_writes_a_file(tmp_path: Path):
    frames = np.random.default_rng(1).random((4, 8, 8))
    written = save_animation(tmp_path / "anim.gif", frames, fps=10, title="t")
    assert Path(written).exists()
    assert Path(written).stat().st_size > 0
```

- [ ] **Step 2 : Lancer le test et vérifier l'échec**

Run : `.venv/bin/python -m pytest tests/test_io_utils.py -q`
Expected : FAIL (`ModuleNotFoundError: No module named 'src.io_utils'`).

- [ ] **Step 3 : Implémenter `src/io_utils.py`**

```python
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
    except Exception:
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
```

- [ ] **Step 4 : Lancer les tests et vérifier le succès**

Run : `.venv/bin/python -m pytest tests/test_io_utils.py -q`
Expected : PASS (2 passed).

- [ ] **Step 5 : Commit**

```bash
git add src/io_utils.py tests/test_io_utils.py
git commit -m "feat(io): dataset .npz + export animation (gif/fallback montage)"
```

---

## Task 2 : `metrics` — masse, erreur L2, croissance d'erreur, saut de couture

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: rien (feuille ; ne dépend que de numpy).
- Produces:
  - `total_mass(h: np.ndarray, dx: float, dy: float) -> float` — `h` `(H,W)`, retourne `sum(h)*dx*dy`.
  - `mass_series(h_seq: np.ndarray, dx: float, dy: float) -> np.ndarray` — `h_seq` `(T,H,W)` → `(T,)`.
  - `relative_l2_error(pred: np.ndarray, true: np.ndarray) -> float` — `‖pred-true‖₂ / (‖true‖₂ + eps)`.
  - `error_growth(pred_seq: np.ndarray, true_seq: np.ndarray) -> np.ndarray` — `(T,...)` chacun → `(T,)` erreur L2 relative par frame.
  - `seam_jump(field: np.ndarray, i0: int, j0: int, size: int) -> float` — `field` `(H,W)` composé multirésolution ; fenêtre carrée `[i0:i0+size, j0:j0+size]`. Retourne le saut absolu moyen à travers les 4 bords (intérieur vs extérieur), bords hors-grille ignorés.

- [ ] **Step 1 : Écrire les tests**

```python
# tests/test_metrics.py
import numpy as np

from src.metrics import (total_mass, mass_series, relative_l2_error,
                         error_growth, seam_jump)


def test_total_mass_and_series():
    h = np.ones((4, 4))
    assert total_mass(h, 1.0, 1.0) == 16.0
    assert total_mass(h, 0.5, 2.0) == 16.0
    seq = np.stack([h, 2 * h, 3 * h])  # (3,4,4)
    assert np.allclose(mass_series(seq, 1.0, 1.0), [16.0, 32.0, 48.0])


def test_relative_l2_error():
    a = np.array([3.0, 4.0])
    assert relative_l2_error(a, a) == 0.0
    # pred=0 vs true=a -> ||a||/||a|| = 1
    assert abs(relative_l2_error(np.zeros_like(a), a) - 1.0) < 1e-9


def test_error_growth_shape_and_zero():
    seq = np.random.default_rng(0).random((6, 5, 5))
    g = error_growth(seq, seq)
    assert g.shape == (6,)
    assert np.allclose(g, 0.0)


def test_seam_jump_zero_on_constant_field():
    field = np.full((16, 16), 3.0)
    assert seam_jump(field, 4, 4, 6) == 0.0


def test_seam_jump_positive_on_discontinuity():
    field = np.zeros((16, 16))
    field[4:10, 4:10] = 1.0  # bloc à 1 entouré de 0 -> saut de 1 au bord
    j = seam_jump(field, 4, 4, 6)
    assert j > 0.5  # saut moyen proche de 1 sur les bords internes
```

- [ ] **Step 2 : Lancer et vérifier l'échec**

Run : `.venv/bin/python -m pytest tests/test_metrics.py -q`
Expected : FAIL (`ModuleNotFoundError: No module named 'src.metrics'`).

- [ ] **Step 3 : Implémenter `src/metrics.py`**

```python
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
```

- [ ] **Step 4 : Lancer et vérifier le succès**

Run : `.venv/bin/python -m pytest tests/test_metrics.py -q`
Expected : PASS (5 passed).

- [ ] **Step 5 : Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): masse, erreur L2 relative, croissance d'erreur, saut de couture"
```

---

## Task 3 : `solver` — noyaux M0 (terrain, CI, CFL, pas Lax-Friedrichs)

**Files:**
- Create: `src/solver.py`
- Test: `tests/test_solver.py`

**Interfaces:**
- Consumes: `config.GRAVITY`, `config.GridConfig`, `config.SolverConfig`.
- Produces:
  - `make_terrain(grid: GridConfig, kind: str = "bump") -> np.ndarray` — `(H,W)` ; `"flat"`=0, `"bump"`=bosse gaussienne centrale (amplitude 0.4).
  - `initial_condition_dam_break(grid, depth_left=2.0, depth_right=1.0, split_frac=0.5) -> tuple[h,u,v]` — discontinuité verticale en x ; `u=v=0`.
  - `initial_condition_gaussian_drop(grid, base=1.0, amp=0.5, cx_frac=0.5, cy_frac=0.5, width_frac=0.1) -> tuple[h,u,v]` — bosse gaussienne sur `h`, `u=v=0`.
  - `cfl_dt(h, u, v, grid, cfl) -> float` — pas de temps limité par la CFL pour l'état courant.
  - `lax_friedrichs_step(h, u, v, b, dt, grid, min_depth) -> tuple[h,u,v]` — un pas, paroi réfléchissante, forme conservative (flux d'interface global LF), exactement conservatif en masse.

  Tous les champs sont `(H,W)` `float64`.

- [ ] **Step 1 : Écrire les tests (conservation, lac au repos, symétrie, CFL)**

```python
# tests/test_solver.py
import numpy as np

from config import GridConfig, GRAVITY
from src.solver import (make_terrain, initial_condition_dam_break,
                        initial_condition_gaussian_drop, cfl_dt,
                        lax_friedrichs_step)

GRID = GridConfig(H=32, W=32, dx=1.0, dy=1.0)


def _run(h, u, v, b, n, cfl=0.45, min_depth=1e-3):
    dt = cfl_dt(h, u, v, GRID, cfl)
    for _ in range(n):
        h, u, v = lax_friedrichs_step(h, u, v, b, dt, GRID, min_depth)
    return h, u, v


def test_cfl_dt_positive_and_scales_with_cfl():
    h, u, v = initial_condition_gaussian_drop(GRID)
    b = make_terrain(GRID, "flat")
    dt1 = cfl_dt(h, u, v, GRID, 0.45)
    dt2 = cfl_dt(h, u, v, GRID, 0.9)
    assert dt1 > 0
    assert abs(dt2 / dt1 - 2.0) < 1e-9


def test_mass_conserved_flat_terrain():
    h, u, v = initial_condition_gaussian_drop(GRID)
    b = make_terrain(GRID, "flat")
    m0 = h.sum()
    h2, _, _ = _run(h, u, v, b, n=50)
    assert abs(h2.sum() - m0) / m0 < 1e-9  # paroi réfléchissante -> masse conservée


def test_lake_at_rest_flat_terrain_stays_still():
    h = np.full((GRID.H, GRID.W), 1.5)
    u = np.zeros((GRID.H, GRID.W))
    v = np.zeros((GRID.H, GRID.W))
    b = make_terrain(GRID, "flat")
    h2, u2, v2 = _run(h, u, v, b, n=20)
    assert np.allclose(h2, 1.5, atol=1e-9)
    assert np.allclose(u2, 0.0, atol=1e-9)
    assert np.allclose(v2, 0.0, atol=1e-9)


def test_centered_drop_stays_symmetric():
    h, u, v = initial_condition_gaussian_drop(GRID, cx_frac=0.5, cy_frac=0.5)
    b = make_terrain(GRID, "flat")
    h2, _, _ = _run(h, u, v, b, n=15)
    # symétrie gauche/droite (axe x) sur un domaine et une CI symétriques
    assert np.allclose(h2, h2[:, ::-1], atol=1e-8)
    assert np.allclose(h2, h2[::-1, :], atol=1e-8)


def test_step_preserves_shape_and_dtype():
    h, u, v = initial_condition_dam_break(GRID)
    b = make_terrain(GRID, "bump")
    dt = cfl_dt(h, u, v, GRID, 0.45)
    h2, u2, v2 = lax_friedrichs_step(h, u, v, b, dt, GRID, 1e-3)
    assert h2.shape == (GRID.H, GRID.W)
    assert h2.dtype == np.float64
```

- [ ] **Step 2 : Lancer et vérifier l'échec**

Run : `.venv/bin/python -m pytest tests/test_solver.py -q`
Expected : FAIL (`ModuleNotFoundError: No module named 'src.solver'`).

- [ ] **Step 3 : Implémenter `src/solver.py`**

```python
"""M0 — Oracle shallow-water 2D (Saint-Venant) sur terrain.

Schéma : volumes finis avec flux d'interface de Rusanov (Lax-Friedrichs LOCAL),
exactement conservatif en masse. La viscosité numérique de chaque interface est
la vitesse d'onde LOCALE (et non dx/2dt) : elle reste correctement dissipative
quand les vitesses augmentent, ce qui évite l'instabilité du LF global. Paroi
réfléchissante (cellules fantômes : hauteur en Neumann, composante normale de la
quantité de mouvement négée).
Variables conservées internes : q1=h, q2=h*u, q3=h*v.

Indexation array[y, x] : axe 0 = y (H lignes), axe 1 = x (W colonnes).
u = vitesse selon x, v = vitesse selon y. b = bathymétrie (terrain) fixe."""
from __future__ import annotations

import numpy as np

from config import GRAVITY, GridConfig


def make_terrain(grid: GridConfig, kind: str = "bump") -> np.ndarray:
    """Bathymétrie fixe (H,W). 'flat'=plat ; 'bump'=bosse gaussienne centrale."""
    H, W = grid.H, grid.W
    if kind == "flat":
        return np.zeros((H, W), dtype=np.float64)
    if kind == "bump":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        sigma = min(H, W) / 6.0
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        return 0.4 * np.exp(-r2 / (2.0 * sigma ** 2))
    raise ValueError(f"terrain inconnu : {kind!r}")


def initial_condition_dam_break(grid: GridConfig, depth_left: float = 2.0,
                                depth_right: float = 1.0,
                                split_frac: float = 0.5):
    """Rupture de barrage : marche de hauteur en x. u=v=0."""
    H, W = grid.H, grid.W
    h = np.full((H, W), depth_right, dtype=np.float64)
    split = int(split_frac * W)
    h[:, :split] = depth_left
    return h, np.zeros((H, W)), np.zeros((H, W))


def initial_condition_gaussian_drop(grid: GridConfig, base: float = 1.0,
                                    amp: float = 0.5, cx_frac: float = 0.5,
                                    cy_frac: float = 0.5, width_frac: float = 0.1):
    """Goutte/bosse gaussienne sur la hauteur. u=v=0."""
    H, W = grid.H, grid.W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = cx_frac * (W - 1), cy_frac * (H - 1)
    sigma = width_frac * min(H, W)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    h = base + amp * np.exp(-r2 / (2.0 * sigma ** 2))
    return h, np.zeros((H, W)), np.zeros((H, W))


def cfl_dt(h: np.ndarray, u: np.ndarray, v: np.ndarray,
           grid: GridConfig, cfl: float) -> float:
    """Pas de temps CFL 2D : cfl / (max(|u|+c)/dx + max(|v|+c)/dy).

    Forme combinée 2D (somme des contributions x et y), qui est la condition de
    stabilité de l'intégration explicite (Euler avant + flux de Rusanov). Avec
    cfl < 1 on conserve une marge même si les vitesses montent au fil du temps."""
    c = np.sqrt(GRAVITY * np.maximum(h, 0.0))
    inv_dt = (float((np.abs(u) + c).max()) / grid.dx
              + float((np.abs(v) + c).max()) / grid.dy)
    return cfl / max(inv_dt, 1e-12)


def _pad_reflective(q1: np.ndarray, q2: np.ndarray, q3: np.ndarray):
    """Cellules fantômes pour paroi réfléchissante.

    q1=h : Neumann (edge). q2=h*u : composante normale négée aux bords x
    (gauche/droite). q3=h*v : composante normale négée aux bords y (haut/bas).
    Retourne des tableaux (H+2, W+2)."""
    p1 = np.pad(q1, 1, mode="edge")
    p2 = np.pad(q2, 1, mode="edge")
    p3 = np.pad(q3, 1, mode="edge")
    p2[:, 0] = -p2[:, 1]      # bord gauche (x) : négation de h*u
    p2[:, -1] = -p2[:, -2]    # bord droit (x)
    p3[0, :] = -p3[1, :]      # bord haut (y) : négation de h*v
    p3[-1, :] = -p3[-2, :]    # bord bas (y)
    return p1, p2, p3


def lax_friedrichs_step(h: np.ndarray, u: np.ndarray, v: np.ndarray,
                        b: np.ndarray, dt: float, grid: GridConfig,
                        min_depth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Un pas de temps en volumes finis (flux de Rusanov / Lax-Friedrichs local).

    Conservatif en masse : le flux numérique de h aux parois est nul."""
    g = GRAVITY
    dx, dy = grid.dx, grid.dy

    q1, q2, q3 = h, h * u, h * v
    p1, p2, p3 = _pad_reflective(q1, q2, q3)
    hp = np.maximum(p1, min_depth)            # plancher pour les divisions
    up, vp = p2 / hp, p3 / hp

    # Flux physiques en chaque cellule (fantômes inclus)
    F1, F2, F3 = p2, p2 * up + 0.5 * g * p1 ** 2, p2 * vp     # flux selon x
    G1, G2, G3 = p3, p3 * up, p3 * vp + 0.5 * g * p1 ** 2     # flux selon y

    # Flux de Rusanov : viscosité numérique = vitesse d'onde LOCALE par interface
    # (max des deux cellules adjacentes), découplée de dt. Reste correctement
    # dissipative quand |u|,|v| augmentent (contrairement au LF global dont la
    # viscosité dx/2dt, figée par le dt initial, devient insuffisante).
    cwave = np.sqrt(g * hp)
    sx = np.abs(up) + cwave   # vitesse d'onde selon x, par cellule (H+2, W+2)
    sy = np.abs(vp) + cwave   # vitesse d'onde selon y, par cellule (H+2, W+2)

    def fx(F, U):  # flux numérique aux interfaces x -> (H+2, W+1)
        alpha = np.maximum(sx[:, :-1], sx[:, 1:])
        return 0.5 * (F[:, :-1] + F[:, 1:]) - 0.5 * alpha * (U[:, 1:] - U[:, :-1])

    def fy(G, U):  # flux numérique aux interfaces y -> (H+1, W+2)
        alpha = np.maximum(sy[:-1, :], sy[1:, :])
        return 0.5 * (G[:-1, :] + G[1:, :]) - 0.5 * alpha * (U[1:, :] - U[:-1, :])

    Fx1, Fx2, Fx3 = fx(F1, p1), fx(F2, p2), fx(F3, p3)
    Gy1, Gy2, Gy3 = fy(G1, p1), fy(G2, p2), fy(G3, p3)

    def divx(Fx):  # divergence en x sur les cellules réelles -> (H, W)
        return (Fx[1:-1, 1:] - Fx[1:-1, :-1]) / dx

    def divy(Gy):  # divergence en y sur les cellules réelles -> (H, W)
        return (Gy[1:, 1:-1] - Gy[:-1, 1:-1]) / dy

    new1 = q1 - dt * (divx(Fx1) + divy(Gy1))
    new2 = q2 - dt * (divx(Fx2) + divy(Gy2))
    new3 = q3 - dt * (divx(Fx3) + divy(Gy3))

    # Terme source de bathymétrie : -g h db/dx (sur h*u), -g h db/dy (sur h*v)
    dbdx = np.zeros_like(b)
    dbdy = np.zeros_like(b)
    dbdx[:, 1:-1] = (b[:, 2:] - b[:, :-2]) / (2.0 * dx)
    dbdy[1:-1, :] = (b[2:, :] - b[:-2, :]) / (2.0 * dy)
    new2 = new2 - dt * g * q1 * dbdx
    new3 = new3 - dt * g * q1 * dbdy

    h_safe = np.maximum(new1, min_depth)
    return new1, new2 / h_safe, new3 / h_safe
```

- [ ] **Step 4 : Lancer et vérifier le succès**

Run : `.venv/bin/python -m pytest tests/test_solver.py -q`
Expected : PASS (5 passed). Si `test_lake_at_rest` échoue avec terrain "bump", c'est attendu (LF n'est pas well-balanced) — le test utilise `"flat"`, où il doit passer.

- [ ] **Step 5 : Commit**

```bash
git add src/solver.py tests/test_solver.py
git commit -m "feat(solver): noyaux M0 shallow-water Lax-Friedrichs conservatif + CI + CFL"
```

---

## Task 4 : M0 — `simulate` + script de génération des datasets

**Files:**
- Modify: `src/solver.py` (ajouter `simulate`)
- Create: `scripts/run_m0_generate.py`
- Test: `tests/test_solver.py` (ajouter `test_simulate_*`)

**Interfaces:**
- Consumes: `lax_friedrichs_step`, `cfl_dt`, CI, terrain (Task 3) ; `io_utils.Dataset`, `save_dataset`, `save_animation` ; `metrics.mass_series`.
- Produces:
  - `simulate(h0, u0, v0, b, grid, solver_cfg) -> tuple[h_seq, u_seq, v_seq, dt]` — `h_seq,u_seq,v_seq` `(T,H,W)` avec `T = n_steps // save_every + 1` (état initial inclus), `dt` fixe utilisé. **Assert CFL à chaque pas** (lève `RuntimeError` si violée).
  - Datasets écrits dans `data/ground_truth/<nom>.npz` : `drop_center`, `drop_offset`, `dam_break` (vus) + `drop_test` (CI de test mise de côté). Terrain `"bump"` commun.

- [ ] **Step 1 : Écrire le test de `simulate`**

```python
# Ajouter à tests/test_solver.py
from config import SolverConfig
from src.solver import simulate


def test_simulate_shapes_and_mass_conservation():
    grid = GridConfig(H=24, W=24)
    cfg = SolverConfig(cfl=0.45, n_steps=40, save_every=4, min_depth=1e-3)
    h0, u0, v0 = initial_condition_gaussian_drop(grid)
    b = make_terrain(grid, "flat")
    hs, us, vs, dt = simulate(h0, u0, v0, b, grid, cfg)
    assert hs.shape == (40 // 4 + 1, 24, 24)
    assert dt > 0
    masses = hs.sum(axis=(1, 2))
    assert (abs(masses - masses[0]) / masses[0]).max() < 1e-9
```

- [ ] **Step 2 : Lancer et vérifier l'échec**

Run : `.venv/bin/python -m pytest tests/test_solver.py::test_simulate_shapes_and_mass_conservation -q`
Expected : FAIL (`ImportError: cannot import name 'simulate'`).

- [ ] **Step 3 : Ajouter `simulate` à `src/solver.py`**

```python
from config import SolverConfig  # ajouter en tête avec les autres imports


def simulate(h0: np.ndarray, u0: np.ndarray, v0: np.ndarray, b: np.ndarray,
             grid: GridConfig, cfg: SolverConfig):
    """Intègre la dynamique sur cfg.n_steps avec un dt FIXE (échantillonnage
    temporel uniforme requis par POD/DMD). Assert CFL à chaque pas.

    Retourne (h_seq, u_seq, v_seq, dt) avec les séquences (T,H,W)."""
    # dt FIXE (échantillonnage temporel uniforme requis par POD/DMD), choisi
    # conservativement sur la CI avec une marge 0.5 : laisse ~2x de headroom pour
    # la montée de vitesse (ex. rupture de barrage où |u|+sqrt(gh) augmente) avant
    # que la CFL ne soit violée. La CFL est revérifiée à chaque pas ci-dessous.
    dt = 0.5 * cfl_dt(h0, u0, v0, grid, cfg.cfl)

    h, u, v = h0.copy(), u0.copy(), v0.copy()
    hs, us, vs = [h.copy()], [u.copy()], [v.copy()]
    for step in range(1, cfg.n_steps + 1):
        # Vérifie la CFL pour l'état courant avec le dt fixe
        dt_limit = cfl_dt(h, u, v, grid, cfg.cfl)
        if dt > dt_limit + 1e-12:
            raise RuntimeError(
                f"CFL violée au pas {step}: dt={dt:.4e} > limite={dt_limit:.4e}")
        h, u, v = lax_friedrichs_step(h, u, v, b, dt, grid, cfg.min_depth)
        if step % cfg.save_every == 0:
            hs.append(h.copy())
            us.append(u.copy())
            vs.append(v.copy())
    return (np.stack(hs), np.stack(us), np.stack(vs), dt)
```

- [ ] **Step 4 : Lancer et vérifier le succès**

Run : `.venv/bin/python -m pytest tests/test_solver.py -q`
Expected : PASS (6 passed).

- [ ] **Step 5 : Écrire `scripts/run_m0_generate.py`**

```python
"""M0 — Génère les rollouts oracle (vérité terrain) et une animation de contrôle.

Usage : .venv/bin/python scripts/run_m0_generate.py
Sorties : data/ground_truth/*.npz, outputs/m0_control_drop_center.(gif|png)
Interprétation attendue : masse conservée (~1e-9), ondes qui se propagent et
se réfléchissent sur les parois de façon physique."""
from __future__ import annotations

import sys
from pathlib import Path

# Rendre src/ et config importables quel que soit le CWD
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from config import GridConfig, SolverConfig
from src.solver import (make_terrain, simulate, initial_condition_dam_break,
                        initial_condition_gaussian_drop)
from src.io_utils import Dataset, save_dataset, save_animation
from src.metrics import mass_series

# ----------------------------- CONFIG ------------------------------------
GRID = GridConfig(H=64, W=64, dx=1.0, dy=1.0)
# n_steps relevé (et save_every=4) car la CFL 2D donne un dt plus petit : il faut
# ~800 pas pour que les ondes traversent le domaine (64 unités) et se réfléchissent.
# -> 201 frames par rollout, suffisant pour POD/DMD et l'évaluation long-horizon H2.
SOLVER = SolverConfig(cfl=0.45, n_steps=800, save_every=4, min_depth=1e-3)
TERRAIN_KIND = "bump"
OUT_DATA = ROOT / "data" / "ground_truth"
OUT_FIG = ROOT / "outputs"
# CI : (nom, fonction, est_test)
CASES = [
    ("drop_center", lambda g: initial_condition_gaussian_drop(g, cx_frac=0.5, cy_frac=0.5), False),
    ("drop_offset", lambda g: initial_condition_gaussian_drop(g, cx_frac=0.3, cy_frac=0.6), False),
    ("dam_break", lambda g: initial_condition_dam_break(g, 2.0, 1.0, 0.4), False),
    ("drop_test", lambda g: initial_condition_gaussian_drop(g, cx_frac=0.65, cy_frac=0.35, amp=0.6), True),
]
# -------------------------------------------------------------------------


def main() -> None:
    b = make_terrain(GRID, TERRAIN_KIND)
    for name, ci_fn, is_test in CASES:
        h0, u0, v0 = ci_fn(GRID)
        hs, us, vs, dt = simulate(h0, u0, v0, b, GRID, SOLVER)
        masses = mass_series(hs, GRID.dx, GRID.dy)
        drift = float((np.abs(masses - masses[0]) / masses[0]).max())
        assert drift < 1e-8, f"{name}: dérive de masse {drift:.2e} trop élevée"
        meta = {"dx": GRID.dx, "dy": GRID.dy, "dt": dt, "ci": name,
                "schema": "lax-friedrichs", "cfl": SOLVER.cfl,
                "terrain": TERRAIN_KIND, "is_test": is_test}
        save_dataset(OUT_DATA / f"{name}.npz", Dataset(hs, us, vs, b, meta))
        print(f"[M0] {name:12s} T={hs.shape[0]:3d} dt={dt:.4e} dérive_masse={drift:.2e}")
        if name == "drop_center":
            written = save_animation(OUT_FIG / "m0_control_drop_center.gif", hs,
                                     fps=20, cmap="viridis", title="M0 — h(t) drop_center")
            print(f"[M0] animation de contrôle -> {written}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6 : Exécuter le script et inspecter le résultat**

Run :
```bash
.venv/bin/python scripts/run_m0_generate.py
ls -la data/ground_truth outputs
```
Expected : 4 fichiers `.npz` créés ; lignes `dérive_masse` toutes `< 1e-8` ; un fichier `m0_control_drop_center.gif` (ou `.png` en fallback). **Ouvrir l'animation et confirmer visuellement** que les ondes se propagent et se réfléchissent (validation H1 partielle, oracle physique).

- [ ] **Step 7 : Commit**

```bash
git add src/solver.py scripts/run_m0_generate.py tests/test_solver.py
git commit -m "feat(M0): simulate + génération datasets oracle + animation de contrôle"
```

---

## Task 5 : M1 — `pod` (base réduite SVD)

**Files:**
- Create: `src/pod.py`
- Test: `tests/test_pod.py`

**Interfaces:**
- Consumes: rien d'autre que numpy/scipy.
- Produces:
  - `@dataclass class PODBasis` : `mean (n_features,)`, `scale (n_features,)`, `Phi (n_features,k)`, `singular_values (n_modes,)`.
  - `stack_snapshots(h, u, v) -> np.ndarray` — entrées `(T,H,W)` → `X (3*H*W, T)`, ordre des canaux `[h, u, v]`, aplatissement C-order par canal.
  - `unstack(X, H, W) -> tuple[h,u,v]` — `X (3*H*W, n)` → trois `(n,H,W)`.
  - `fit_pod(X, energy_threshold, max_modes) -> PODBasis` — centre + met à l'échelle par canal, SVD économique, choisit `k` au seuil d'énergie.
  - `encode(basis, X) -> z (k, n)` ; `decode(basis, z) -> X (n_features, n)`.
  - `cumulative_energy(singular_values) -> np.ndarray` — énergie cumulée normalisée `(n_modes,)`.

  **Contrat** : `X ≈ scale[:,None] * (Phi @ z) + mean[:,None]`.

- [ ] **Step 1 : Écrire les tests (roundtrip, énergie, décroissance erreur)**

```python
# tests/test_pod.py
import numpy as np

from src.pod import (stack_snapshots, unstack, fit_pod, encode, decode,
                     cumulative_energy)


def _toy_field(T=20, H=8, W=8, seed=0):
    rng = np.random.default_rng(seed)
    # signal de faible rang : quelques modes spatiaux x temporels
    H_, W_ = H, W
    modes = rng.random((3, H_, W_))
    coeffs = rng.random((T, 3))
    h = 1.0 + coeffs[:, 0, None, None] * modes[0]
    u = 0.1 * coeffs[:, 1, None, None] * modes[1]
    v = 0.1 * coeffs[:, 2, None, None] * modes[2]
    return h, u, v


def test_stack_unstack_roundtrip():
    h, u, v = _toy_field()
    X = stack_snapshots(h, u, v)
    assert X.shape == (3 * 8 * 8, 20)
    h2, u2, v2 = unstack(X, 8, 8)
    assert np.allclose(h2, h) and np.allclose(u2, u) and np.allclose(v2, v)


def test_cumulative_energy_monotone_and_reaches_one():
    s = np.array([3.0, 2.0, 1.0, 0.0])
    e = cumulative_energy(s)
    assert e.shape == (4,)
    assert np.all(np.diff(e) >= -1e-12)
    assert abs(e[-1] - 1.0) < 1e-12


def test_full_rank_reconstruction_is_near_exact():
    h, u, v = _toy_field()
    X = stack_snapshots(h, u, v)
    basis = fit_pod(X, energy_threshold=1.0, max_modes=64)
    z = encode(basis, X)
    X_rec = decode(basis, z)
    assert np.linalg.norm(X_rec - X) / np.linalg.norm(X) < 1e-8


def test_reconstruction_error_decreases_with_k():
    h, u, v = _toy_field(T=40, seed=3)
    X = stack_snapshots(h, u, v)
    errs = []
    for thr in (0.5, 0.9, 0.999):
        basis = fit_pod(X, energy_threshold=thr, max_modes=64)
        z = encode(basis, X)
        errs.append(np.linalg.norm(decode(basis, z) - X) / np.linalg.norm(X))
    assert errs[0] >= errs[1] >= errs[2] - 1e-12
```

- [ ] **Step 2 : Lancer et vérifier l'échec**

Run : `.venv/bin/python -m pytest tests/test_pod.py -q`
Expected : FAIL (`ModuleNotFoundError: No module named 'src.pod'`).

- [ ] **Step 3 : Implémenter `src/pod.py`**

```python
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
    flat = lambda a: a.reshape(T, H * W).T          # (H*W, T)
    return np.concatenate([flat(h), flat(u), flat(v)], axis=0)


def unstack(X: np.ndarray, H: int, W: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X (3*H*W, n) -> (h,u,v) chacun (n,H,W)."""
    n = X.shape[1]
    hw = H * W
    parts = [X[i * hw:(i + 1) * hw, :].T.reshape(n, H, W) for i in range(3)]
    return parts[0], parts[1], parts[2]


def _channel_scale(X: np.ndarray) -> np.ndarray:
    """Écart-type par canal (3 blocs égaux), diffusé sur (n_features,)."""
    n_features = X.shape[0]
    hw = n_features // 3
    scale = np.empty(n_features)
    for i in range(3):
        block = X[i * hw:(i + 1) * hw, :]
        s = float(block.std())
        scale[i * hw:(i + 1) * hw] = s if s > _EPS else 1.0
    return scale


def fit_pod(X: np.ndarray, energy_threshold: float, max_modes: int) -> PODBasis:
    """Centre + met à l'échelle par canal, SVD économique, choisit k au seuil."""
    mean = X.mean(axis=1)
    scale = _channel_scale(X)
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
    total = float(e2.sum()) + _EPS
    return np.cumsum(e2) / total
```

- [ ] **Step 4 : Lancer et vérifier le succès**

Run : `.venv/bin/python -m pytest tests/test_pod.py -q`
Expected : PASS (4 passed).

- [ ] **Step 5 : Commit**

```bash
git add src/pod.py tests/test_pod.py
git commit -m "feat(M1): base réduite POD (SVD, scale par canal, encode/decode)"
```

---

## Task 6 : M1 — script POD (énergie, erreur recon, modes spatiaux)

**Files:**
- Create: `scripts/run_m1_pod.py`

**Interfaces:**
- Consumes: `io_utils.load_dataset` ; `pod.*` ; datasets de Task 4.
- Produces: `outputs/m1_energy.png`, `outputs/m1_recon_error_vs_k.png`, `outputs/m1_modes.png` ; `data/pod_basis.npz` (mean, scale, Phi, singular_values + H,W) pour réutilisation M2/M3.

- [ ] **Step 1 : Écrire `scripts/run_m1_pod.py`**

```python
"""M1 — POD : énergie cumulée, erreur de reconstruction vs k, modes spatiaux.

Usage : .venv/bin/python scripts/run_m1_pod.py
Interprétation attendue : l'énergie atteint 99 % avec peu de modes (quelques
dizaines) ; l'erreur de reconstruction est faible à k petit -> H1 (volet POD) OK."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from config import PODConfig
from src.io_utils import load_dataset
from src.pod import stack_snapshots, unstack, fit_pod, encode, decode, cumulative_energy

# ----------------------------- CONFIG ------------------------------------
POD = PODConfig(energy_threshold=0.99, max_modes=128)
TRAIN_CASES = ["drop_center", "drop_offset", "dam_break"]  # CI vues
DATA = ROOT / "data" / "ground_truth"
OUT = ROOT / "outputs"
# -------------------------------------------------------------------------


def main() -> None:
    cols = []
    H = W = None
    for name in TRAIN_CASES:
        ds = load_dataset(DATA / f"{name}.npz")
        H, W = ds.h.shape[1], ds.h.shape[2]
        cols.append(stack_snapshots(ds.h, ds.u, ds.v))
    X = np.concatenate(cols, axis=1)
    print(f"[M1] X shape = {X.shape} (n_features, n_snapshots)")

    basis = fit_pod(X, POD.energy_threshold, POD.max_modes)
    k = basis.Phi.shape[1]
    energy = cumulative_energy(basis.singular_values)
    print(f"[M1] k retenu pour {POD.energy_threshold:.0%} d'énergie = {k}")

    # Sauvegarde de la base pour M2/M3
    np.savez_compressed(ROOT / "data" / "pod_basis.npz",
                        mean=basis.mean, scale=basis.scale, Phi=basis.Phi,
                        singular_values=basis.singular_values, H=H, W=W)

    # Figure 1 : énergie cumulée
    plt.figure(figsize=(5, 4))
    plt.plot(np.arange(1, len(energy) + 1), energy, marker=".")
    plt.axhline(POD.energy_threshold, color="r", ls="--", label=f"{POD.energy_threshold:.0%}")
    plt.axvline(k, color="g", ls="--", label=f"k={k}")
    plt.xlabel("nombre de modes"); plt.ylabel("énergie cumulée"); plt.legend()
    plt.title("M1 — énergie cumulée POD"); plt.tight_layout()
    plt.savefig(OUT / "m1_energy.png", dpi=120); plt.close()

    # Figure 2 : erreur de reconstruction vs k
    ks = [k_ for k_ in (1, 2, 4, 8, 16, 32, 64, 128) if k_ <= basis.Phi.shape[1]]
    errs = []
    for k_ in ks:
        b_k = type(basis)(basis.mean, basis.scale, basis.Phi[:, :k_], basis.singular_values)
        errs.append(np.linalg.norm(decode(b_k, encode(b_k, X)) - X) / np.linalg.norm(X))
    plt.figure(figsize=(5, 4))
    plt.semilogy(ks, errs, marker="o")
    plt.xlabel("k (nombre de modes)"); plt.ylabel("erreur L2 relative")
    plt.title("M1 — erreur de reconstruction vs k"); plt.tight_layout()
    plt.savefig(OUT / "m1_recon_error_vs_k.png", dpi=120); plt.close()
    print("[M1] erreur recon par k :", {k_: round(e, 4) for k_, e in zip(ks, errs)})

    # Figure 3 : 4 premiers modes spatiaux (canal h)
    hw = H * W
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for m in range(min(4, basis.Phi.shape[1])):
        mode_h = basis.Phi[:hw, m].reshape(H, W)
        axes[m].imshow(mode_h, cmap="RdBu_r", origin="lower")
        axes[m].set_title(f"mode {m} (h)"); axes[m].axis("off")
    fig.suptitle("M1 — premiers modes spatiaux POD (canal h)")
    fig.tight_layout(); fig.savefig(OUT / "m1_modes.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2 : Exécuter et inspecter**

Run :
```bash
.venv/bin/python scripts/run_m1_pod.py
```
Expected : impression de `k` retenu (idéalement quelques dizaines), trois PNG dans `outputs/`, `data/pod_basis.npz` créé. **Vérifier visuellement** que l'énergie atteint 99 % tôt et que l'erreur recon décroît vite. **Décision H1 (POD)** : si `k` reste petit avec erreur faible → volet POD de H1 validé.

- [ ] **Step 3 : Commit**

```bash
git add scripts/run_m1_pod.py
git commit -m "feat(M1): script POD (énergie, erreur recon vs k, modes spatiaux)"
```

---

## Task 7 : M2 — `dmd` (dynamique latente linéaire)

**Files:**
- Create: `src/dmd.py`
- Test: `tests/test_dmd.py`

**Interfaces:**
- Consumes: numpy.
- Produces:
  - `fit_dmd(z_list: list[np.ndarray]) -> np.ndarray` — chaque `z` `(k, T_i)` ; identifie `A (k,k)` par moindres carrés sur les paires consécutives **internes à chaque trajectoire** : `A = Z2 @ pinv(Z1)`.
  - `rollout(A: np.ndarray, z0: np.ndarray, n_steps: int) -> np.ndarray` — `z0 (k,)` → `z_pred (k, n_steps+1)`.
  - `spectral_radius(A) -> float` — `max |eig(A)|` (diagnostic de stabilité ; >1 ⇒ croissance attendue).

- [ ] **Step 1 : Écrire les tests (récupération d'un système linéaire connu)**

```python
# tests/test_dmd.py
import numpy as np

from src.dmd import fit_dmd, rollout, spectral_radius


def _make_trajectory(A0, z0, T):
    z = np.zeros((A0.shape[0], T))
    z[:, 0] = z0
    for t in range(T - 1):
        z[:, t + 1] = A0 @ z[:, t]
    return z


def test_fit_dmd_recovers_linear_operator():
    rng = np.random.default_rng(0)
    k = 4
    A0 = 0.95 * np.array([[np.cos(0.2), -np.sin(0.2), 0, 0],
                          [np.sin(0.2), np.cos(0.2), 0, 0],
                          [0, 0, 0.9, 0],
                          [0, 0, 0, 0.8]])
    trajs = [_make_trajectory(A0, rng.random(k), 30) for _ in range(3)]
    A = fit_dmd(trajs)
    assert A.shape == (k, k)
    assert np.linalg.norm(A - A0) / np.linalg.norm(A0) < 1e-8


def test_rollout_matches_truth_for_known_system():
    k = 3
    A0 = np.diag([0.99, 0.95, 0.9])
    z0 = np.array([1.0, 1.0, 1.0])
    true = _make_trajectory(A0, z0, 20)
    pred = rollout(A0, z0, 19)
    assert pred.shape == (k, 20)
    assert np.allclose(pred, true)


def test_spectral_radius():
    A0 = np.diag([0.5, 1.2, 0.3])
    assert abs(spectral_radius(A0) - 1.2) < 1e-9
```

- [ ] **Step 2 : Lancer et vérifier l'échec**

Run : `.venv/bin/python -m pytest tests/test_dmd.py -q`
Expected : FAIL (`ModuleNotFoundError: No module named 'src.dmd'`).

- [ ] **Step 3 : Implémenter `src/dmd.py`**

```python
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
```

- [ ] **Step 4 : Lancer et vérifier le succès**

Run : `.venv/bin/python -m pytest tests/test_dmd.py -q`
Expected : PASS (3 passed).

- [ ] **Step 5 : Commit**

```bash
git add src/dmd.py tests/test_dmd.py
git commit -m "feat(M2): DMD (fit moindres carrés, rollout autorégressif, rayon spectral)"
```

---

## Task 8 : M2 — script DMD (rollout court vs vérité, côte à côte)

**Files:**
- Create: `scripts/run_m2_dmd.py`

**Interfaces:**
- Consumes: `io_utils`, `pod` (recharge `data/pod_basis.npz`), `dmd`, `metrics`.
- Produces: `data/dmd_A.npz` (A) ; `outputs/m2_dmd_vs_truth.gif` (ou montage) côte à côte sur horizon court ; impression du rayon spectral.

- [ ] **Step 1 : Écrire `scripts/run_m2_dmd.py`**

```python
"""M2 — DMD : fit sur les trajectoires latentes, rollout court vs vérité terrain.

Usage : .venv/bin/python scripts/run_m2_dmd.py
Interprétation attendue : à court horizon, le champ reconstruit par DMD suit la
vérité (baseline de référence). Le rayon spectral indique la tendance long-terme."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.io_utils import load_dataset, save_animation
from src.pod import PODBasis, encode, decode, stack_snapshots, unstack
from src.dmd import fit_dmd, rollout, spectral_radius

# ----------------------------- CONFIG ------------------------------------
TRAIN_CASES = ["drop_center", "drop_offset", "dam_break"]
DEMO_CASE = "drop_center"   # rollout démontré
SHORT_HORIZON = 40          # nombre de pas du rollout court
DATA = ROOT / "data"; GT = DATA / "ground_truth"; OUT = ROOT / "outputs"
# -------------------------------------------------------------------------


def load_basis() -> tuple[PODBasis, int, int]:
    d = np.load(DATA / "pod_basis.npz")
    basis = PODBasis(d["mean"], d["scale"], d["Phi"], d["singular_values"])
    return basis, int(d["H"]), int(d["W"])


def main() -> None:
    basis, H, W = load_basis()
    z_list = []
    for name in TRAIN_CASES:
        ds = load_dataset(GT / f"{name}.npz")
        z_list.append(encode(basis, stack_snapshots(ds.h, ds.u, ds.v)))
    A = fit_dmd(z_list)
    np.savez_compressed(DATA / "dmd_A.npz", A=A)
    rho = spectral_radius(A)
    print(f"[M2] rayon spectral de A = {rho:.4f} ({'stable' if rho <= 1.0 else 'CROISSANT'})")

    ds = load_dataset(GT / f"{DEMO_CASE}.npz")
    z_true = encode(basis, stack_snapshots(ds.h, ds.u, ds.v))
    n = min(SHORT_HORIZON, z_true.shape[1] - 1)
    z_pred = rollout(A, z_true[:, 0], n)
    h_pred, _, _ = unstack(decode(basis, z_pred), H, W)
    h_true = ds.h[:n + 1]

    # Animation côte à côte (concat horizontale vérité | prédiction)
    side = np.concatenate([h_true, h_pred], axis=2)  # (n+1, H, 2W)
    written = save_animation(OUT / "m2_dmd_vs_truth.gif", side, fps=15,
                             title=f"M2 — {DEMO_CASE} : vérité | DMD")
    print(f"[M2] animation -> {written}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2 : Exécuter et inspecter**

Run :
```bash
.venv/bin/python scripts/run_m2_dmd.py
```
Expected : rayon spectral imprimé ; `data/dmd_A.npz` créé ; animation côte à côte. **Vérifier visuellement** que la prédiction DMD suit la vérité sur l'horizon court (validation H1, volet dynamique).

- [ ] **Step 3 : Commit**

```bash
git add scripts/run_m2_dmd.py
git commit -m "feat(M2): script DMD rollout court vs vérité (côte à côte)"
```

---

## Task 9 : M3 — Évaluation H2 (dérive en rollout) — **PRIORITAIRE**

**Files:**
- Create: `scripts/run_m3_eval_rollout.py`

**Interfaces:**
- Consumes: `io_utils`, `pod` (`data/pod_basis.npz`), `dmd` (`data/dmd_A.npz`), `metrics` (`error_growth`, `mass_series`).
- Produces: `outputs/m3_error_growth.png` (CI vue + CI test), `outputs/m3_mass_drift.png`, `outputs/m3_longhorizon_<case>.gif` ; `data/m3_eval.npz` (courbes). Impression d'un verdict H2 chiffré.

- [ ] **Step 1 : Écrire `scripts/run_m3_eval_rollout.py`**

```python
"""M3 — H2 : dérive du rollout DMD sur long horizon (CI vue ET CI de test).

Usage : .venv/bin/python scripts/run_m3_eval_rollout.py
Interprétation attendue : caractériser si/quand le rollout dérive ou explose, et
si la masse totale prédite dérive. Résultat valable même s'il est négatif."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.io_utils import load_dataset, save_animation
from src.pod import PODBasis, encode, decode, stack_snapshots, unstack
from src.dmd import rollout
from src.metrics import error_growth, mass_series

# ----------------------------- CONFIG ------------------------------------
SEEN_CASE = "drop_center"   # CI vue à l'entraînement
TEST_CASE = "drop_test"     # CI mise de côté (généralisation)
DATA = ROOT / "data"; GT = DATA / "ground_truth"; OUT = ROOT / "outputs"
# -------------------------------------------------------------------------


def load_basis():
    d = np.load(DATA / "pod_basis.npz")
    return PODBasis(d["mean"], d["scale"], d["Phi"], d["singular_values"]), int(d["H"]), int(d["W"])


def evaluate(name, basis, A, H, W):
    """Rollout long-horizon vs vérité ; retourne (err_curve, mass_pred, mass_true, h_true, h_pred, dx, dy)."""
    ds = load_dataset(GT / f"{name}.npz")
    dx, dy = ds.meta["dx"], ds.meta["dy"]
    X_true = stack_snapshots(ds.h, ds.u, ds.v)
    z_true = encode(basis, X_true)
    T = z_true.shape[1]
    z_pred = rollout(A, z_true[:, 0], T - 1)
    X_pred = decode(basis, z_pred)
    h_pred, _, _ = unstack(X_pred, H, W)
    err = error_growth(X_pred.T.reshape(T, -1), X_true.T.reshape(T, -1))
    return (err, mass_series(h_pred, dx, dy), mass_series(ds.h, dx, dy),
            ds.h, h_pred, dx, dy)


def main() -> None:
    basis, H, W = load_basis()
    A = np.load(DATA / "dmd_A.npz")["A"]

    results = {name: evaluate(name, basis, A, H, W) for name in (SEEN_CASE, TEST_CASE)}

    # Figure 1 : croissance d'erreur (vue + test)
    plt.figure(figsize=(6, 4))
    for name in (SEEN_CASE, TEST_CASE):
        err = results[name][0]
        plt.plot(err, label=f"{name} (final={err[-1]:.3f})")
    plt.xlabel("pas de temps"); plt.ylabel("erreur L2 relative")
    plt.title("M3 — H2 : croissance d'erreur du rollout"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "m3_error_growth.png", dpi=120); plt.close()

    # Figure 2 : dérive de masse (prédite vs vérité)
    plt.figure(figsize=(6, 4))
    for name in (SEEN_CASE, TEST_CASE):
        _, m_pred, m_true, *_ = results[name]
        plt.plot((m_pred - m_true[0]) / m_true[0], label=f"{name} prédit")
        plt.plot((m_true - m_true[0]) / m_true[0], ls="--", label=f"{name} vérité")
    plt.xlabel("pas de temps"); plt.ylabel("dérive relative de masse")
    plt.title("M3 — H2 : dérive de la masse totale"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "m3_mass_drift.png", dpi=120); plt.close()

    # Animations long-horizon côte à côte + verdict chiffré
    verdicts = {}
    for name in (SEEN_CASE, TEST_CASE):
        err, m_pred, m_true, h_true, h_pred, *_ = results[name]
        side = np.concatenate([h_true, h_pred], axis=2)
        save_animation(OUT / f"m3_longhorizon_{name}.gif", side, fps=15,
                       title=f"M3 — {name} : vérité | DMD (long horizon)")
        verdicts[name] = {
            "err_final": float(err[-1]),
            "err_max": float(err.max()),
            "exploded": bool(err.max() > 5.0 or not np.isfinite(err).all()),
            "mass_drift_final": float((m_pred[-1] - m_true[0]) / m_true[0]),
        }
    np.savez_compressed(DATA / "m3_eval.npz",
                        **{f"{n}_err": results[n][0] for n in results})
    print("[M3] verdict H2 :")
    for name, vd in verdicts.items():
        print(f"   {name:12s} err_final={vd['err_final']:.3f} err_max={vd['err_max']:.3f} "
              f"explosé={vd['exploded']} dérive_masse_finale={vd['mass_drift_final']:.2e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2 : Exécuter et inspecter (verdict H2)**

Run :
```bash
.venv/bin/python scripts/run_m3_eval_rollout.py
```
Expected : impression du verdict H2 (err_final, err_max, explosé, dérive masse) pour CI vue + CI test ; deux PNG + deux animations. **Interprétation** : caractériser le moment où le rollout dérive/explose et l'ampleur de la dérive de masse. C'est le livrable central de H2 — un verdict, positif ou négatif. **Noter ici si la baseline DMD est jugée insuffisante** → conditionne la porte M4/M5 (Task 13).

- [ ] **Step 3 : Commit**

```bash
git add scripts/run_m3_eval_rollout.py
git commit -m "feat(M3): évaluation H2 (croissance d'erreur + dérive masse, CI vue+test)"
```

---

## Task 10 : M6 — `multiresolution` (grossier global + fenêtre fine mobile)

**Files:**
- Create: `src/multiresolution.py`
- Test: `tests/test_multiresolution.py`

**Interfaces:**
- Consumes: numpy.
- Produces:
  - `@dataclass(frozen=True) class Window` : `i0: int`, `j0: int`, `size: int`.
  - `downsample(field, factor) -> np.ndarray` — `(H,W)` → `(H//factor, W//factor)` par moyenne de blocs (assert divisibilité).
  - `upsample(coarse, factor) -> np.ndarray` — répétition voisin-le-plus-proche (kron) → `(H,W)`.
  - `compose_multiresolution(field, window, coarse_factor, blend_width=0) -> np.ndarray` — fond grossier (down+up), fenêtre fine collée ; `blend_width>0` = anneau de fondu linéaire fine→grossier pour atténuer le popping.
  - `window_trajectory(grid, size, n_frames, axis="x", margin=4) -> list[Window]` — fenêtre qui translate linéairement.

- [ ] **Step 1 : Écrire les tests**

```python
# tests/test_multiresolution.py
import numpy as np

from config import GridConfig
from src.multiresolution import (Window, downsample, upsample,
                                 compose_multiresolution, window_trajectory)


def test_downsample_upsample_shapes():
    field = np.arange(64.0).reshape(8, 8)
    c = downsample(field, 2)
    assert c.shape == (4, 4)
    up = upsample(c, 2)
    assert up.shape == (8, 8)


def test_compose_window_is_exact_fine_inside():
    rng = np.random.default_rng(0)
    field = rng.random((16, 16))
    w = Window(i0=4, j0=4, size=6)
    comp = compose_multiresolution(field, w, coarse_factor=4, blend_width=0)
    assert comp.shape == field.shape
    # à l'intérieur de la fenêtre, le composé == champ fin exact
    assert np.allclose(comp[4:10, 4:10], field[4:10, 4:10])


def test_blend_reduces_seam_jump_on_smooth_field():
    from src.metrics import seam_jump
    yy, xx = np.mgrid[0:32, 0:32].astype(float)
    field = np.sin(xx / 5.0) + np.cos(yy / 7.0)  # champ lisse
    w = Window(i0=10, j0=10, size=10)
    hard = compose_multiresolution(field, w, coarse_factor=4, blend_width=0)
    soft = compose_multiresolution(field, w, coarse_factor=4, blend_width=3)
    j_hard = seam_jump(hard, w.i0, w.j0, w.size)
    j_soft = seam_jump(soft, w.i0, w.j0, w.size)
    assert j_soft <= j_hard + 1e-9


def test_window_trajectory_in_bounds():
    grid = GridConfig(H=64, W=64)
    wins = window_trajectory(grid, size=16, n_frames=10, axis="x", margin=4)
    assert len(wins) == 10
    for w in wins:
        assert 0 <= w.j0 and w.j0 + w.size <= grid.W
        assert 0 <= w.i0 and w.i0 + w.size <= grid.H
```

- [ ] **Step 2 : Lancer et vérifier l'échec**

Run : `.venv/bin/python -m pytest tests/test_multiresolution.py -q`
Expected : FAIL (`ModuleNotFoundError: No module named 'src.multiresolution'`).

- [ ] **Step 3 : Implémenter `src/multiresolution.py`**

```python
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

    # Poids de fondu : 1 au cœur de la fenêtre, descend vers 0 sur l'anneau extérieur
    yy, xx = np.mgrid[i0:i1, j0:j1]
    dist = np.minimum.reduce([yy - i0, i1 - 1 - yy, xx - j0, j1 - 1 - xx]).astype(float)
    w = np.clip(dist / max(blend_width, 1), 0.0, 1.0)
    out[i0:i1, j0:j1] = w * field[i0:i1, j0:j1] + (1.0 - w) * background[i0:i1, j0:j1]
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
```

- [ ] **Step 4 : Lancer et vérifier le succès**

Run : `.venv/bin/python -m pytest tests/test_multiresolution.py -q`
Expected : PASS (4 passed).

- [ ] **Step 5 : Commit**

```bash
git add src/multiresolution.py tests/test_multiresolution.py
git commit -m "feat(M6): multirésolution (down/up, compose fenêtre + fondu, trajectoire)"
```

---

## Task 11 : M6 — Évaluation H3 (couture pendant le déplacement) — **PRIORITAIRE**

**Files:**
- Create: `scripts/run_m6_multiresolution.py`

**Interfaces:**
- Consumes: `io_utils`, `multiresolution`, `metrics.seam_jump`. Champ source = vérité terrain `drop_center` (option : champ reconstruit POD si `pod_basis.npz` présent — ici on prend la vérité pour isoler l'effet multirésolution).
- Produces: `outputs/m6_seam_jump.png` (saut vs temps, collage dur vs fondu), `outputs/m6_window_moving.gif` (composé avec fenêtre qui bouge) ; impression d'un verdict H3 chiffré.

- [ ] **Step 1 : Écrire `scripts/run_m6_multiresolution.py`**

```python
"""M6 — H3 : cohérence de la couture quand la fenêtre fine se déplace.

Usage : .venv/bin/python scripts/run_m6_multiresolution.py
Interprétation attendue : quantifier le saut résiduel au bord de la fenêtre au
fil du temps (collage dur vs fondu) -> caractériser le 'popping'."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from config import GridConfig
from src.io_utils import load_dataset, save_animation
from src.multiresolution import compose_multiresolution, window_trajectory
from src.metrics import seam_jump

# ----------------------------- CONFIG ------------------------------------
SOURCE_CASE = "drop_center"
COARSE_FACTOR = 4    # 64 -> 16 grossier
WINDOW_SIZE = 16
BLEND_WIDTH = 3      # largeur d'anneau de fondu (variante atténuée)
AXIS = "x"
DATA = ROOT / "data" / "ground_truth"; OUT = ROOT / "outputs"
# -------------------------------------------------------------------------


def main() -> None:
    ds = load_dataset(DATA / f"{SOURCE_CASE}.npz")
    h = ds.h                      # (T,H,W) — on observe le canal hauteur
    T, H, W = h.shape
    grid = GridConfig(H=H, W=W, dx=ds.meta["dx"], dy=ds.meta["dy"])
    wins = window_trajectory(grid, WINDOW_SIZE, T, axis=AXIS)

    jumps_hard = np.zeros(T)
    jumps_soft = np.zeros(T)
    composed_frames = np.zeros((T, H, W))
    for t in range(T):
        w = wins[t]
        hard = compose_multiresolution(h[t], w, COARSE_FACTOR, blend_width=0)
        soft = compose_multiresolution(h[t], w, COARSE_FACTOR, blend_width=BLEND_WIDTH)
        jumps_hard[t] = seam_jump(hard, w.i0, w.j0, w.size)
        jumps_soft[t] = seam_jump(soft, w.i0, w.j0, w.size)
        composed_frames[t] = soft

    # Figure : saut de couture vs temps
    plt.figure(figsize=(6, 4))
    plt.plot(jumps_hard, label=f"collage dur (moy={jumps_hard.mean():.3f})")
    plt.plot(jumps_soft, label=f"fondu w={BLEND_WIDTH} (moy={jumps_soft.mean():.3f})")
    plt.xlabel("pas de temps (fenêtre en déplacement)")
    plt.ylabel("saut de couture moyen")
    plt.title("M6 — H3 : saut à la couture vs temps"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "m6_seam_jump.png", dpi=120); plt.close()

    written = save_animation(OUT / "m6_window_moving.gif", composed_frames, fps=15,
                             title="M6 — fenêtre fine mobile (fond grossier)")
    print(f"[M6] verdict H3 : saut moyen collage_dur={jumps_hard.mean():.4f} "
          f"max={jumps_hard.max():.4f} | fondu={jumps_soft.mean():.4f} max={jumps_soft.max():.4f}")
    print(f"[M6] animation -> {written}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2 : Exécuter et inspecter (verdict H3)**

Run :
```bash
.venv/bin/python scripts/run_m6_multiresolution.py
```
Expected : impression du saut moyen/max (collage dur vs fondu) ; un PNG + une animation de la fenêtre mobile. **Interprétation H3** : la couture reste-t-elle cohérente (saut faible/stable) quand la fenêtre bouge, et le fondu réduit-il le popping ? Verdict chiffré.

- [ ] **Step 3 : Commit**

```bash
git add scripts/run_m6_multiresolution.py
git commit -m "feat(M6): évaluation H3 (saut de couture vs temps, fenêtre mobile)"
```

---

## Task 12 : M7 — `render` + script de rendu de l'état prédit

**Files:**
- Create: `src/render.py`
- Create: `scripts/run_m7_render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `io_utils.save_animation`, `pod`, `dmd`.
- Produces:
  - `surface_height(h, b) -> np.ndarray` — surface libre `eta = h + b`, formes `(T,H,W)`+`(H,W)` (b diffusé) ou `(H,W)`+`(H,W)`.
  - `render_rollout(path, field_seq, *, cmap="viridis", fps=20, title="") -> str` — délègue à `save_animation`.
  - Script `run_m7_render.py` : reconstruit le rollout prédit (POD+DMD) de `drop_center` et exporte heatmap de `h` + surface `eta`.

- [ ] **Step 1 : Écrire le test**

```python
# tests/test_render.py
import numpy as np
from pathlib import Path

from src.render import surface_height, render_rollout


def test_surface_height_adds_bathymetry():
    h = np.ones((3, 4, 4))
    b = np.full((4, 4), 0.5)
    eta = surface_height(h, b)
    assert eta.shape == (3, 4, 4)
    assert np.allclose(eta, 1.5)


def test_render_rollout_writes_file(tmp_path: Path):
    frames = np.random.default_rng(0).random((4, 8, 8))
    out = render_rollout(tmp_path / "r.gif", frames, title="t")
    assert Path(out).exists()
```

- [ ] **Step 2 : Lancer et vérifier l'échec**

Run : `.venv/bin/python -m pytest tests/test_render.py -q`
Expected : FAIL (`ModuleNotFoundError: No module named 'src.render'`).

- [ ] **Step 3 : Implémenter `src/render.py`**

```python
"""M7 — Rendu comme relevé de l'état : heatmap et surface de hauteur.

L'image est un relevé direct du champ simulé/prédit (« ce que tu vois est ce
que tu simules »)."""
from __future__ import annotations

import numpy as np

from src.io_utils import save_animation


def surface_height(h: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Surface libre eta = h + b (b diffusé sur l'axe temporel si besoin)."""
    return h + b


def render_rollout(path, field_seq: np.ndarray, *, cmap: str = "viridis",
                   fps: int = 20, title: str = "") -> str:
    """Exporte une séquence (T,H,W) en animation (délègue à io_utils)."""
    return save_animation(path, field_seq, fps=fps, cmap=cmap, title=title)
```

- [ ] **Step 4 : Lancer et vérifier le succès**

Run : `.venv/bin/python -m pytest tests/test_render.py -q`
Expected : PASS (2 passed).

- [ ] **Step 5 : Écrire `scripts/run_m7_render.py`**

```python
"""M7 — Rendu de l'état prédit (POD+DMD) : heatmap de h et surface eta=h+b.

Usage : .venv/bin/python scripts/run_m7_render.py
Interprétation attendue : l'animation prédite est visuellement plausible et
temporellement cohérente."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.io_utils import load_dataset
from src.pod import PODBasis, encode, decode, stack_snapshots, unstack
from src.dmd import rollout
from src.render import surface_height, render_rollout

# ----------------------------- CONFIG ------------------------------------
CASE = "drop_center"
DATA = ROOT / "data"; GT = DATA / "ground_truth"; OUT = ROOT / "outputs"
# -------------------------------------------------------------------------


def main() -> None:
    d = np.load(DATA / "pod_basis.npz")
    basis = PODBasis(d["mean"], d["scale"], d["Phi"], d["singular_values"])
    H, W = int(d["H"]), int(d["W"])
    A = np.load(DATA / "dmd_A.npz")["A"]

    ds = load_dataset(GT / f"{CASE}.npz")
    z_true = encode(basis, stack_snapshots(ds.h, ds.u, ds.v))
    z_pred = rollout(A, z_true[:, 0], z_true.shape[1] - 1)
    h_pred, _, _ = unstack(decode(basis, z_pred), H, W)

    out1 = render_rollout(OUT / "m7_height_heatmap.gif", h_pred, cmap="viridis",
                          title=f"M7 — h prédit ({CASE})")
    out2 = render_rollout(OUT / "m7_surface_eta.gif", surface_height(h_pred, ds.b),
                          cmap="terrain", title=f"M7 — surface eta=h+b ({CASE})")
    print(f"[M7] rendus -> {out1} ; {out2}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6 : Exécuter et inspecter**

Run :
```bash
.venv/bin/python scripts/run_m7_render.py
```
Expected : deux animations dans `outputs/`. **Vérifier visuellement** la plausibilité et la cohérence temporelle du rendu prédit (validation H1/M7).

- [ ] **Step 7 : Commit**

```bash
git add src/render.py scripts/run_m7_render.py tests/test_render.py
git commit -m "feat(M7): rendu de l'état (heatmap h + surface eta)"
```

---

## Task 13 : README, index des sorties, porte de décision M4/M5

**Files:**
- Create: `README.md`
- Create: `docs/M4_M5_decision_gate.md`

**Interfaces:**
- Consumes: tous les scripts précédents.
- Produces: documentation exécutable + critère explicite d'activation de M4/M5.

- [ ] **Step 1 : Écrire `README.md`**

````markdown
# POC — Simulateur réduit appris de fluide 2D (shallow-water)

Modèle d'ordre réduit (POD + DMD) approximant un solveur shallow-water 2D, pour
trancher trois hypothèses : H1 (colonne vertébrale POD+DMD), H2 (dérive du
rollout long-horizon), H3 (couture multirésolution mobile).

## Installation

L'environnement utilise [`uv`](https://github.com/astral-sh/uv) (le `pip`/`venv`
système peuvent être absents) :

```bash
export PATH="$HOME/.local/bin:$PATH"
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

(`pillow` est optionnel : sans lui, les animations sont exportées en montage PNG.
`torch` n'est nécessaire que pour M4/M5 — voir `docs/M4_M5_decision_gate.md`.)

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Exécution des jalons

| Jalon | Commande | Résultat attendu |
|-------|----------|------------------|
| M0 (oracle)        | `.venv/bin/python scripts/run_m0_generate.py`     | Datasets `data/ground_truth/*.npz`, masse conservée (~1e-9), `outputs/m0_control_drop_center.*` |
| M1 (POD)           | `.venv/bin/python scripts/run_m1_pod.py`          | `outputs/m1_energy.png`, `m1_recon_error_vs_k.png`, `m1_modes.png` ; k petit pour 99 % |
| M2 (DMD)           | `.venv/bin/python scripts/run_m2_dmd.py`          | `outputs/m2_dmd_vs_truth.*` ; rayon spectral imprimé |
| M3 (H2)            | `.venv/bin/python scripts/run_m3_eval_rollout.py` | `outputs/m3_error_growth.png`, `m3_mass_drift.png`, `m3_longhorizon_*` ; verdict H2 |
| M6 (H3)            | `.venv/bin/python scripts/run_m6_multiresolution.py` | `outputs/m6_seam_jump.png`, `m6_window_moving.*` ; verdict H3 |
| M7 (rendu)         | `.venv/bin/python scripts/run_m7_render.py`       | `outputs/m7_height_heatmap.*`, `m7_surface_eta.*` |

Exécuter dans l'ordre (M1→M7 dépendent des sorties de M0/M1/M2).

## Interprétation des hypothèses

- **H1** : `m1_energy.png` (k petit pour 99 %) + `m2_dmd_vs_truth` (suivi court horizon).
- **H2** : `m3_error_growth.png` + `m3_mass_drift.png` (CI vue *et* CI test) — caractérise quand/si le rollout dérive ou explose.
- **H3** : `m6_seam_jump.png` — saut à la couture pendant le déplacement de la fenêtre (collage dur vs fondu).
````

- [ ] **Step 2 : Écrire `docs/M4_M5_decision_gate.md` (porte de décision)**

```markdown
# Porte de décision M4 / M5 (déférée)

Conformément à la spec (« réduction d'abord », « n'attaque M4/M5 que si M2/M3 le
justifient »), M4 (dynamique latente non-linéaire, MLP/GRU PyTorch) et M5
(régularisation physique) ne sont implémentés QUE si la baseline DMD est jugée
insuffisante après M3.

## Critère d'activation (à renseigner après M3)

Activer M4 si AU MOINS un des points est vrai sur la CI vue ou la CI test :
- le rollout DMD **explose** (err_max > 5.0 ou non fini) avant la fin de l'horizon ;
- l'erreur L2 relative finale dépasse un seuil jugé inacceptable (p. ex. > 0.5) ;
- la dérive de masse prédite est qualitativement non physique (croissance non bornée).

Si la baseline DMD est « suffisante » (erreur bornée, pas d'explosion), **ne pas
implémenter M4/M5** : documenter le verdict H2 et s'arrêter.

## Contraintes M4/M5 (si activé)

- `torch>=2.2`, GPU CUDA RTX 3050 Ti **≤ 4 Go de VRAM**, AMP si besoin.
- Modèle minuscule (MLP/GRU sur le latent z de dimension k) ; entraînement sur les
  trajectoires latentes de M1 ; comparaison directe aux métriques de M3.
- M5 : pénalité de conservation de masse (ou résidu d'EDP grossier) ajoutée à la
  perte ; mesurer l'effet sur la dérive et la généralisation à la CI test.

> Un plan détaillé de M4/M5 sera rédigé séparément seulement après que le verdict
> de M3 ait confirmé le besoin.
```

- [ ] **Step 3 : Lancer la suite de tests complète**

Run : `.venv/bin/python -m pytest -q`
Expected : PASS (tous les tests des Tasks 1–12).

- [ ] **Step 4 : Commit**

```bash
git add README.md docs/M4_M5_decision_gate.md
git commit -m "docs: README exécutable + porte de décision M4/M5"
```

---

## Self-Review (vérification du plan contre la spec)

**1. Couverture de la spec :**
- M0 solveur oracle (LF conservatif, CFL, source bathymétrie, paroi réfléchissante, multiples CI + 1 CI test, animation, masse conservée) → Tasks 3–4. ✅
- M1 POD (SVD, k au seuil d'énergie 99 %, encode/decode, énergie + erreur recon + modes) → Tasks 5–6. ✅
- M2 DMD (A par moindres carrés, rollout autorégressif, côte à côte) → Tasks 7–8. ✅
- M3 H2 prioritaire (croissance d'erreur, dérive masse, CI vue + test, long horizon) → Task 9. ✅
- M4/M5 optionnels derrière porte de décision → Task 13. ✅ (déférés conformément à « réduction d'abord »)
- M6 H3 prioritaire (grossier global + fenêtre fine mobile, saut de couture vs temps, animation) → Tasks 10–11. ✅
- M7 rendu comme relevé de l'état → Task 12. ✅
- Contrats de données §5 (formes/dtypes, X, Phi, z, A, npz+meta) → respectés ; extension `scale` documentée dans Global Constraints + `pod.py`. ✅
- Invariants §9 (masse assert+tracé, CFL assert, erreur recon vs k, croissance erreur, saut couture, figures listées README) → couverts. ✅
- Structure §8 → Task 0 + arborescence (racine = CWD au lieu de sous-dossier `poc-fluide-reduit/`, choix documenté). ✅
- Contraintes §3 (frugal CPU minutes, deps minimales, incrémental, GPU ≤4 Go déféré) → respectées. ✅

**2. Placeholders :** aucun « TODO / à compléter / similaire à Task N ». Tout step de code montre le code complet. ✅

**3. Cohérence des types :** `PODBasis(mean, scale, Phi, singular_values)` utilisé identiquement dans Tasks 5/6/8/9/12 ; `stack_snapshots/unstack/encode/decode` signatures stables ; `Window(i0,j0,size)` et `seam_jump(field,i0,j0,size)` cohérents entre Tasks 2/10/11 ; `simulate(...) -> (h,u,v,dt)` cohérent Task 4↔9. ✅

**Écart assumé documenté :** racine du projet = répertoire de travail courant (`pocPhysicator/`) plutôt que le sous-dossier `poc-fluide-reduit/` suggéré — le dépôt EST déjà le dossier projet. Aucune autre déviation.
```
