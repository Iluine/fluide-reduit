import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_band_error_exceeds_global_on_moving_front():
    """Cœur du raffinement #1 : sur un front mobile axisymétrique, l'erreur sur la
    bande-front doit DÉPASSER l'erreur globale (le bulk lisse dilue la difficulté)."""
    from scripts.run_w0_representation import nwidth_ceiling
    from config import GridConfig
    from src.analytic_thacker import thacker_radial, thacker_radial_period
    g = GridConfig(H=64, W=64, dx=4.0 / 64, dy=4.0 / 64)
    T = thacker_radial_period()
    seq = np.stack([thacker_radial(g, t=ph * T)[0] for ph in np.linspace(0, 1, 60)])
    res = nwidth_ceiling(seq, 64, 64, energy_threshold=0.999, max_modes=64)
    assert res["k"] >= 1
    assert res["global_err"] < 0.05
    assert res["band_err"] > res["global_err"]    # le front est plus dur que le bulk
