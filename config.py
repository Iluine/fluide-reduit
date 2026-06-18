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
