import numpy as np
import pytest
from config import GridConfig, GRAVITY
from src.analytic_thacker import (thacker_radial, thacker_radial_period,
                                  thacker_planar_diag, thacker_planar_period,
                                  front_band_mask)

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


def test_planar_diag_is_translating_rotated_1d_front():
    # HONNÊTE : c'est un profil 1D PIVOTÉ, extrudé le long de l'anti-diagonale (h ne dépend
    # que de s=((x-L/2)+(y-L/2))/√2, donc h[y,x]==h[x,y]). Son rôle n'est PAS d'être
    # structurellement 2D (rang limité), mais de TRANSLATER — la translation d'un profil
    # est le cas n-width canonique (stresseur de transport). On teste ça + un landmark de
    # physique (largeur de bande mouillée = 2a), pas une fausse 2D-ité.
    a = 1.0
    T = thacker_planar_period()
    h0f, _ = thacker_planar_diag(GRID, t=0.0, a=a)
    hqf, _ = thacker_planar_diag(GRID, t=T / 4, a=a)
    assert h0f.shape == (64, 64) and (h0f >= 0).all()
    assert h0f[20, 30] == pytest.approx(h0f[30, 20])      # extrudé le long de l'anti-diagonale
    # LANDMARK : la bande mouillée a une largeur ≈ 2a le long de s (pin la formule)
    xs = (np.arange(64) + 0.5) * GRID.dx
    ys = (np.arange(64) + 0.5) * GRID.dy
    xx, yy = np.meshgrid(xs, ys)
    s = ((xx - 2.0) + (yy - 2.0)) / np.sqrt(2.0)
    wet = h0f > 1e-3
    assert (s[wet].max() - s[wet].min()) == pytest.approx(2.0 * a, abs=3 * GRID.dx)
    # le centroïde mouillé TRANSLATE entre deux phases (le front traverse le domaine)
    def centroid(h):
        m = h > 1e-3
        yy2, xx2 = np.mgrid[0:64, 0:64]
        return (xx2[m].mean(), yy2[m].mean()) if m.any() else (np.nan, np.nan)
    c0, c1 = centroid(h0f), centroid(hqf)
    assert abs(c0[0] - c1[0]) + abs(c0[1] - c1[1]) > 1.0   # translation nette


def test_front_band_mask_hugs_shoreline():
    h, _ = thacker_radial(GRID, t=0.0)
    band = front_band_mask(h, eps=1e-3, width=2)
    assert band.dtype == bool and band.shape == (64, 64)
    assert band.sum() > 0
    assert (h[band] > 0).all()                    # la bande est dans la zone mouillée
    # la bande exclut le coeur profond (on garde le voisinage du trait d'eau)
    assert band.sum() < (h > 1e-3).sum()
