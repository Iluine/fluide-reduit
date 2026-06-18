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
