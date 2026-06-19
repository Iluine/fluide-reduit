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
