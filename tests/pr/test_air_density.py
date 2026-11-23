# Copyright 2026 Power Lab

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Air-density adjustment of the wind conversion (Plan 4.1, 4.2). Offline."""

import glob

import numpy as np
import pytest
import xarray as xr

from geodata.model.wind import hub_air_density, surface_pressure_temperature
from geodata.model.wind.interpolate import WindInterpolationModel

FIXTURE_SL = "tests/fixtures/era5/wind_solar_hourly_test/2016/*.nc"


def test_standard_atmosphere():
    # ISA sea level: 101325 Pa, 288.15 K -> 1.225 kg/m3; at 100 m about 1.213
    assert hub_air_density(101325.0, 288.15, 2.0) == pytest.approx(1.225, abs=2e-3)
    assert hub_air_density(101325.0, 288.15, 100.0) == pytest.approx(1.213, abs=2e-3)


def test_plateau_is_thinner_and_cold_is_denser():
    high = hub_air_density(65000.0, 275.0, 100.0)  # about 3.6 km, Tibetan Plateau
    assert 0.78 < high < 0.86
    assert hub_air_density(101325.0, 253.15, 100.0) > hub_air_density(101325.0, 303.15, 100.0)


def _model():
    from geodata.datasets import load_dataset

    ds = load_dataset("wind_3d_hourly_test")(years=slice(2016, 2016), months=slice(1, 1))
    m = WindInterpolationModel(ds)
    m.prepare()
    return m


def _surface_like(speed, sp, t2m):
    shape = (speed.sizes["time"], speed.sizes["y"], speed.sizes["x"])
    coords = {"time": speed["time"].values, "y": speed["y"].values, "x": speed["x"].values}
    return xr.Dataset(
        {"sp": (("time", "y", "x"), np.full(shape, sp)),
         "t2m": (("time", "y", "x"), np.full(shape, t2m))},
        coords=coords,
    )


def test_density_scales_speed_before_power_curve():
    m = _model()
    turbine = {"specific_power": 250, "hub_height": 90}
    xs, ys = slice(100, 104), slice(30, 34)
    raw = m.estimate(turbine=turbine, xs=xs, ys=ys, hour_average=False)
    speed = m.estimate(height=90, xs=xs, ys=ys)
    thin = _surface_like(speed, 70000.0, 280.0)
    adj = m.estimate(turbine=turbine, xs=xs, ys=ys, hour_average=False, air_density=thin)
    assert adj.dims == raw.dims and adj.shape == raw.shape
    assert "IEC" in adj.attrs["air_density"] and raw.attrs["air_density"] == "none"
    # thinner air: never more output, and less wherever the curve is rising
    assert np.all(adj.values <= raw.values + 1e-9)
    assert float(adj.mean()) < float(raw.mean())
    # a rho of exactly 1.225 changes nothing
    rho = xr.full_like(speed, 1.225).transpose("time", "y", "x")
    same = m.estimate(turbine=turbine, xs=xs, ys=ys, hour_average=False, air_density=rho)
    np.testing.assert_allclose(same.values, raw.values, atol=1e-6)


def test_density_matches_manual_conversion():
    from geodata.resource import resolve_windturbine

    m = _model()
    turbine = {"specific_power": 250, "hub_height": 90}
    xs, ys = slice(100, 104), slice(30, 34)
    speed = m.estimate(height=90, xs=xs, ys=ys)
    sl = _surface_like(speed, 90000.0, 270.0)
    adj = m.estimate(turbine=turbine, xs=xs, ys=ys, hour_average=False, air_density=sl)
    rho = hub_air_density(90000.0, 270.0, 90.0)
    c = resolve_windturbine(turbine)
    v = speed.values * (rho / 1.225) ** (1 / 3)
    expected = np.interp(v, c["V"], c["POW"], left=0.0, right=0.0) / c["P"]
    np.testing.assert_allclose(adj.transpose(*speed.dims).values, expected, atol=1e-6)


def test_missing_time_stamps_raise():
    m = _model()
    speed = m.estimate(height=90, xs=slice(100, 104), ys=slice(30, 34))
    sl = _surface_like(speed, 1e5, 288.0).isel(time=slice(1, None))
    with pytest.raises(ValueError, match="time stamps"):
        m.estimate(turbine={"specific_power": 250, "hub_height": 90},
                   xs=slice(100, 104), ys=slice(30, 34), air_density=sl)


@pytest.mark.skipif(not glob.glob(FIXTURE_SL), reason="no single-level fixture")
def test_fixture_single_level_files():
    sl = surface_pressure_temperature(FIXTURE_SL, xs=slice(100, 104), ys=slice(30, 34))
    assert set(sl.data_vars) == {"sp", "t2m"} and sl["sp"].dims == ("time", "y", "x")
    rho = hub_air_density(sl["sp"], sl["t2m"], 100.0)
    assert float(rho.min()) > 0.6 and float(rho.max()) < 1.5
    m = _model()
    turbine = {"specific_power": 250, "hub_height": 100}
    adj = m.estimate(turbine=turbine, xs=slice(100, 104), ys=slice(30, 34), air_density=sl)
    raw = m.estimate(turbine=turbine, xs=slice(100, 104), ys=slice(30, 34))
    assert adj.shape == raw.shape and np.isfinite(adj.values).all()
