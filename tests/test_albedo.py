import numpy as np
import pytest

from src.albedo import albedo, relief_shaded, f_fraction, chi_bands, delta_chi


def test_albedo_monotone_and_saturating():
    s = np.linspace(0.0, 5.0, 200)
    a = albedo(s, s_half=0.5)
    assert np.all(np.diff(a) >= 0.0)          # monotone croissante
    assert a[0] == pytest.approx(0.0)          # A(0) = 0
    assert a[-1] > 0.999                       # sature vers 1 pour s >> s_half


def test_albedo_shape_and_bounds():
    s = np.abs(np.random.default_rng(0).normal(size=(16, 16)))
    a = albedo(s, s_half=0.3)
    assert a.shape == (16, 16)
    assert np.all(a >= 0.0) and np.all(a < 1.0)


def test_relief_shaded_bounded_in_unit_interval():
    rng = np.random.default_rng(1)
    b = np.cumsum(np.cumsum(rng.normal(size=(32, 32)), axis=0), axis=1) * 0.01
    shade = relief_shaded(b)
    assert shade.shape == (32, 32)
    assert shade.min() >= 0.0
    assert shade.max() <= 1.0


def test_relief_shaded_flat_terrain_is_uniform():
    # terrain plat : normale = (0,0,1) partout -> eclairement constant = sin(elevation)
    b = np.full((16, 16), 0.42)
    shade = relief_shaded(b, elevation_deg=15.0)
    expected = np.sin(np.deg2rad(15.0))
    assert np.allclose(shade, expected)


def test_f_fraction_identity_is_zero():
    rng = np.random.default_rng(2)
    x = rng.uniform(0.1, 1.0, size=(20, 20))
    assert f_fraction(x, x, jnd=0.02) == 0.0


def test_f_fraction_identity_is_zero_even_when_reference_mean_is_zero():
    z = np.zeros((8, 8))
    assert f_fraction(z, z, jnd=0.02) == 0.0


def test_f_fraction_detects_large_deviation():
    y = np.full((10, 10), 1.0)
    x = y.copy()
    x[0, 0] = 10.0  # 900% d'ecart local, tres au-dessus du JND
    frac = f_fraction(x, y, jnd=0.05)
    assert frac == pytest.approx(1.0 / 100.0)


def test_chi_bands_powers_sum_to_total_ac_power():
    rng = np.random.default_rng(3)
    a = np.abs(rng.normal(size=(64, 64))) + 0.1
    chi, power = chi_bands(a)
    F = np.fft.fft2(a)
    total_ac = float(np.sum(np.abs(F) ** 2)) - float(np.abs(F[0, 0]) ** 2)
    assert set(power.keys()) == {"1", "2-3", "4-7", "8-15", "16-31"}
    assert sum(power.values()) == pytest.approx(total_ac, rel=1e-9)


def test_chi_bands_constant_field_has_zero_ac_power():
    a = np.full((64, 64), 0.7)
    chi, power = chi_bands(a)
    assert all(p == pytest.approx(0.0, abs=1e-10) for p in power.values())
    assert all(c == pytest.approx(0.0, abs=1e-10) for c in chi.values())


def test_delta_chi_identity_is_zero():
    rng = np.random.default_rng(4)
    a = np.abs(rng.normal(size=(64, 64))) + 0.1
    d = delta_chi(a, a)
    assert d["max_carrier"] == 0.0
    assert all(v == pytest.approx(0.0) for v in d["delta_chi"].values())


def test_delta_chi_reports_carriers_and_nonzero_deviation():
    rng = np.random.default_rng(5)
    a_ref = np.abs(rng.normal(size=(64, 64))) + 0.1
    a_cand = a_ref + rng.normal(scale=0.2, size=(64, 64))
    d = delta_chi(a_cand, a_ref)
    assert any(d["carriers"].values())          # au moins une bande porteuse
    assert d["max_carrier"] >= 0.0
    assert d["max_carrier"] == max(
        v for b, v in d["delta_chi"].items() if d["carriers"][b])
