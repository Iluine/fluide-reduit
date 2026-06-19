import numpy as np
import pytest
from config import GridConfig, GRAVITY
from src.analytic_thacker import thacker_radial, thacker_radial_period

GRID = GridConfig(H=64, W=64, dx=4.0 / 64, dy=4.0 / 64)


def test_radial_period():
    a, h0 = 1.0, 0.1
    omega = np.sqrt(8 * GRAVITY * h0) / a
    assert thacker_radial_period(a, h0) == pytest.approx(2 * np.pi / omega)


def test_radial_shapes_and_positivity():
    h, b = thacker_radial(GRID, t=0.0)
    assert h.shape == b.shape == (64, 64)
    assert (h >= 0).all()                         # hauteur d'eau toujours positive
    assert b.min() < 0 < b.max() or b.min() < b.max()  # paraboloïde (bol)


def test_radial_has_moving_wet_dry_front():
    # à deux phases différentes, l'aire mouillée (h>seuil) diffère -> shoreline mobile
    T = thacker_radial_period()
    h0_, _ = thacker_radial(GRID, t=0.0)
    h1_, _ = thacker_radial(GRID, t=T / 4)
    wet0 = (h0_ > 1e-3).sum()
    wet1 = (h1_ > 1e-3).sum()
    assert wet0 != wet1                           # le trait d'eau s'est déplacé
    assert wet0 > 0 and wet1 > 0                  # il reste de l'eau


def test_radial_mass_conserved_across_phases():
    # solution exacte sans friction -> masse (intégrale de h) constante dans le temps
    T = thacker_radial_period()
    m = [float(thacker_radial(GRID, t=ph * T)[0].sum()) for ph in (0.0, 0.13, 0.37, 0.5)]
    assert max(m) - min(m) < 0.02 * max(m)        # ~constante (erreur de discrétisation)


def test_radial_shoreline_radius_landmark_at_t0():
    # LANDMARK DISCRIMINANT : à t=0 le trait d'eau est au rayon √(a·r0)≈0.894, PAS r0=0.8
    # (r0 est un paramètre de la solution, pas le rayon de la shoreline). On mesure le rayon
    # équivalent du disque mouillé (aire/π, robuste au bruit de grille) et on EXIGE qu'il
    # soit plus proche de √(a·r0) que de r0 — sinon une formule fausse plaçant la shoreline
    # en r0 passerait (l'écart 0.094 est comparable à la résolution dx).
    a, r0 = 1.0, 0.8
    h, _ = thacker_radial(GRID, t=0.0, a=a, r0=r0)
    wet_area = float((h > 1e-6).sum()) * GRID.dx * GRID.dy
    r_shore = np.sqrt(wet_area / np.pi)                # rayon équivalent du disque mouillé
    target = np.sqrt(a * r0)                           # √0.8 ≈ 0.894
    assert abs(r_shore - target) < abs(r_shore - r0)   # plus proche de √(a·r0) que de r0
    assert r_shore == pytest.approx(target, abs=0.05)  # et proche en absolu de √(a·r0)
