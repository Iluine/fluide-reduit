import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_transfer_works_when_dynamics_is_shared():
    """Si les trajectoires train ET éval suivent la MÊME dynamique latente linéaire,
    l'opérateur DMD global doit transférer : rollout fidèle sur la trajectoire éval
    (le rollout ne peut pas battre le plancher de représentation, mais doit s'en
    approcher quand la dynamique est exactement partagée)."""
    from scripts.run_v2_transfer import evaluate_transfer
    rng = np.random.default_rng(0)
    H = W = 8
    T = 30
    p1 = rng.normal(size=(H, W))
    p2 = rng.normal(size=(H, W))
    A0 = np.array([[0.99, -0.05], [0.05, 0.99]])  # stable (|λ|≈0.992 < 1)

    # base=0 : champ purement modal (z·p). DMD ajuste une carte linéaire HOMOGÈNE ;
    # pour que la dynamique CENTRÉE par la POD reste linéaire (pas affine), la moyenne
    # des snapshots doit être nulle -> CI train symétriques (z0 et -z0 s'annulent).
    def field(z0):
        z = np.array(z0, dtype=float)
        frames = []
        for _ in range(T):
            frames.append(z[0] * p1 + z[1] * p2)
            z = A0 @ z
        return np.stack(frames)

    train = [field([1.0, 0.0]), field([-1.0, 0.0]),
             field([0.0, 1.0]), field([0.0, -1.0])]   # moyenne ≈ 0 par symétrie
    eval_seq = field([0.5, -0.7])                      # nouvelle CI, MÊME dynamique
    out = evaluate_transfer(train, {"eval": eval_seq}, H, W,
                            energy_threshold=0.9999, max_modes=64)

    assert out["k"] >= 2
    assert out["rho"] <= 1.0 + 1e-9                       # écrêté dans le disque unité
    m = out["results"]["eval"]
    assert m["floor"] < 0.01                              # base rang-2 span la famille
    assert m["rollout"] < 0.05                            # transfère (dynamique partagée)
    assert m["rollout"] >= m["floor"] - 1e-9              # rollout ne bat pas le plancher
