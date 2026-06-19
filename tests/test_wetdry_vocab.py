import re
import numpy as np
from config import GridConfig, GRAVITY
from src.solver_wetdry import simulate_wetdry_o2
from src.wetdry_vocab import (moving_wave_ic, dam_break_ic, slope_bed, island_bed,
                              vocabulary, Scenario)

GRID = GridConfig(H=48, W=48, dx=4.0 / 48, dy=4.0 / 48)


def test_moving_wave_propagates_one_way_not_split():
    # Correction 1 : avec u co-directionnel, l'onde va vers +x ; sans, elle se scinderait.
    b = slope_bed(GRID, slope=0.6, axis="x")
    h0, hu0, hv0 = moving_wave_ic(GRID, b, still_surface=0.15, amp=0.10,
                                  x0_frac=0.25, y0_frac=0.5, width_frac=0.08, direction="+x")
    assert (h0 >= 0).all()
    assert float(hu0.sum()) > 0.0                       # momentum net vers +x (co-directionnel)
    # vitesse bornée (pas de pic supersonique au trait d'eau) : Froude raisonnable
    wet = h0 > 1e-3
    u = hu0[wet] / h0[wet]
    froude = np.abs(u) / np.sqrt(GRAVITY * h0[wet])
    assert float(froude.max()) < 3.0                    # bornée (pas 40 m/s en eau peu profonde)
    # après une courte évolution, le paquet a AVANCÉ en +x (pas scindé/immobile).
    # Centroïde de l'ANOMALIE de surface |（h+b)−repos| en zone mouillée (profil en x) :
    # une onde +x déplace le centroïde ; une onde scindée l'étale symétriquement.
    _, hs, _, _ = simulate_wetdry_o2(h0, hu0, hv0, b, GRID, t_end=0.3)
    xs = (np.arange(48) + 0.5) * GRID.dx

    def anom_xprofile(frame):
        eta = frame + b
        anom = np.where(frame > 1e-3, np.abs(eta - 0.15), 0.0)
        return anom.sum(axis=0)                          # somme sur y -> profil en x

    p0, pT = anom_xprofile(hs[0]), anom_xprofile(hs[-1])
    c0 = float((xs * p0).sum() / (p0.sum() + 1e-12))
    cT = float((xs * pT).sum() / (pT.sum() + 1e-12))
    assert cT > c0 + GRID.dx                             # anomalie avancée en +x


def test_vocabulary_weighted_runup_spread_and_nondegenerate():
    voc = vocabulary(GRID)
    assert 12 <= len(voc) <= 18
    runups = [s for s in voc if "runup" in s.name]
    assert len(runups) >= 8                              # PESÉ run-up (assèchement)
    xs_used = {re.search(r"x(\d+)", s.name).group(1) for s in voc if re.search(r"x(\d+)", s.name)}
    assert len(xs_used) >= 3                             # positions étalées

    for s in voc:                                        # anti-dégénérescence (tous)
        assert (s.h0 >= 0).all() and np.isfinite(s.h0).all()
        wet_frac = float((s.h0 > 1e-6).mean())
        assert wet_frac > 0.15, f"{s.name}: zone mouillée dégénérée ({wet_frac:.2f})"

    for s in runups:                                     # anti-dégénérescence (run-up)
        mom = float(np.sqrt(s.hu0 ** 2 + s.hv0 ** 2).sum())
        assert mom > 0.0, f"{s.name}: pas de momentum (onde scindée/absente)"
        wet = s.h0 > 1e-6
        pert = float(s.h0[wet].max() - np.median(s.h0[wet]))
        assert pert > 0.02, f"{s.name}: perturbation dégénérée (bosse en zone sèche ?)"
        wetv = s.h0 > 1e-3                               # vitesse bornée à la CI
        u = np.sqrt(s.hu0[wetv] ** 2 + s.hv0[wetv] ** 2) / s.h0[wetv]
        fr = u / np.sqrt(GRAVITY * s.h0[wetv])
        assert float(fr.max()) < 3.0, f"{s.name}: vitesse divergente (Froude {fr.max():.1f})"


def test_island_is_emergent_and_surrounded_by_water():
    # Correction 3 : still_surface>0 -> base submergée + sommet émergent (pas tout sec).
    isl = island_bed(GRID, amp=0.7, x0_frac=0.5, y0_frac=0.5, sigma=8.0)
    ss = 0.45
    h_still = np.maximum(ss - isl, 0.0)
    assert float((h_still > 1e-6).mean()) > 0.3          # entouré d'eau
    assert float((h_still <= 1e-6).mean()) > 0.0         # sommet émergent (sec)
