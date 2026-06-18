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
