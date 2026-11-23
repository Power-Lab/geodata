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

"""Unit tests for the raw-CF round-1 changes (Plan v6 sections 4.1-4.3).

Offline: synthetic data and the bundled ERA5 fixtures.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geodata.datasets.era5._base import _subset_x_y_era5
from geodata.datasets.era5.wind_solar._base import ERA5WindSolarBaseDataset
from geodata.model.pvlib import PlantSpec, Pvlib, model_level_wind
from geodata.model.pvlib.calculations import sun_position_offsets
from geodata.model.wind.interpolate import WindInterpolationModel
from geodata.resource import get_generic_windturbineconfig, resolve_windturbine
from geodata.utils import to_lon180

MODULE = "SunPower_SPR_P19_400_COM"
INVERTER = "Fronius_USA__CL_33_3_Delta__208V_"
FIXTURE_3D = "tests/fixtures/era5/wind_3d_hourly_test/2016/01/*.nc"


# ---------------------------------------------------------------- 4.1 met data
def _lon360_ds():
    x = np.arange(0, 360, 0.25)
    y = np.array([40.0, 39.75])
    v = np.broadcast_to(x[None, :], (2, x.size)).copy()
    return xr.Dataset({"v": (("y", "x"), v)}, coords={"x": x, "y": y})


def test_to_lon180_wraps_and_sorts():
    ds = to_lon180(_lon360_ds(), "x")
    assert float(ds.x.min()) == -180.0 and float(ds.x.max()) == 179.75
    assert np.all(np.diff(ds.x.values) > 0)
    # values travel with their coordinates: original 260 deg is now -100 deg
    assert float(ds.v.sel(x=-100.0).isel(y=0)) == 260.0
    # already in -180..180: unchanged
    again = to_lon180(ds, "x")
    assert again.x.equals(ds.x)


def test_western_hemisphere_subset_is_not_empty():
    sub = _subset_x_y_era5(_lon360_ds(), slice(-105, -95), slice(40, 39))
    assert sub.sizes["x"] == 41
    assert float(sub.x.min()) == -105.0


def test_u10_v10_become_wnd10m():
    t = pd.date_range("2019-07-01", periods=2, freq="h")
    shape = (2, 2, 2)
    one = np.ones(shape)
    ds = xr.Dataset(
        {name: (("time", "y", "x"), one * val) for name, val in {
            "fdir": 100.0, "tisr": 1000.0, "ssrd": 500.0, "ssr": 400.0, "t2m": 290.0,
            "d2m": 280.0, "u100": 6.0, "v100": 8.0, "u10": 3.0, "v10": 4.0,
            "ro": 0.0, "stl4": 290.0, "sp": 1e5, "fsr": 0.1,
        }.items()},
        coords={"time": t, "y": [1.0, 0.0], "x": [0.0, 1.0]},
    )
    out = ERA5WindSolarBaseDataset.transform_wind_solar_dataset(ds)
    assert "wnd10m" in out and "u10" not in out
    np.testing.assert_allclose(out["wnd10m"].values, 5.0)
    np.testing.assert_allclose(out["wnd100m"].values, 10.0)


# ---------------------------------------------------------------- 4.2 wind
def test_generic_curve_shape_and_cut_speeds():
    c = get_generic_windturbineconfig(300, 100, cut_in=3.0, cut_out=25.0)
    V, P = np.asarray(c["V"]), np.asarray(c["POW"])
    assert c["P"] == 1.0 and c["hub_height"] == 100.0
    assert P.max() == pytest.approx(1.0, abs=1e-6)
    assert np.all(P[V <= 3.0] == 0) and np.all(P[V >= 25.0] == 0)
    assert np.all(np.diff(P[(V > 3.0) & (V < 25.0)]) >= -1e-9)


def test_lower_specific_power_gives_more_power_at_moderate_speed():
    at8 = [np.interp(8.0, *(np.asarray(get_generic_windturbineconfig(sp, 100)[k]) for k in ("V", "POW")))
           for sp in (200, 250.5, 300, 400)]
    assert all(a > b for a, b in zip(at8, at8[1:]))


def test_resolve_turbine_forms():
    g = resolve_windturbine({"specific_power": 250, "hub_height": 90})
    assert g["hub_height"] == 90.0
    with pytest.raises(ValueError):
        resolve_windturbine({"specific_power": 250})
    with pytest.raises(TypeError):
        resolve_windturbine(250)


def test_height_warning_near_top_level():
    levels = np.array([10.0, 30.0, 54.0, 107.0, 169.5])
    assert WindInterpolationModel.check_height(100.0, levels) is None
    assert "within" in WindInterpolationModel.check_height(155.0, levels)
    assert "above" in WindInterpolationModel.check_height(200.0, levels)


def test_hour_average_on_fixture():
    from geodata.datasets import load_dataset

    ds = load_dataset("wind_3d_hourly_test")(years=slice(2016, 2016), months=slice(1, 1))
    model = WindInterpolationModel(ds)
    model.prepare()
    turbine = {"specific_power": 250, "hub_height": 90}
    xs, ys = slice(100, 104), slice(30, 34)
    avg = model.estimate(turbine=turbine, xs=xs, ys=ys)
    inst = model.estimate(turbine=turbine, xs=xs, ys=ys, hour_average=False)
    np.testing.assert_allclose(avg.isel(time=0), inst.isel(time=0))
    expected = (inst.isel(time=slice(1, None)).values + inst.isel(time=slice(None, -1)).values) / 2
    np.testing.assert_allclose(avg.isel(time=slice(1, None)).values, expected)
    assert avg.attrs["hour_average"] == 1 and inst.attrs["hour_average"] == 0
    assert float(avg.max()) <= 1.0 and float(avg.min()) >= 0.0


# ---------------------------------------------------------------- 4.3 solar
def test_sun_offsets_mid_hour_and_sunrise():
    # Equator, longitude 0, 2019-03-21: sunrise about 06:04 UTC.
    t = pd.date_range("2019-03-21 01:00", "2019-03-21 23:00", freq="h").values
    off, app, _ = sun_position_offsets(t, [0.0], [0.0], mode="mid_hour")
    off = off[:, 0, 0]
    hours = pd.DatetimeIndex(t).hour
    assert np.all(off[(hours >= 9) & (hours <= 15)] == 30.0)
    rise = off[hours == 7][0]          # hour 06:00-07:00, sun up for most of it
    assert 0.0 < rise < 30.0           # evaluated after the hour's midpoint
    stamp, _, _ = sun_position_offsets(t, [0.0], [0.0], mode="stamp")
    assert np.all(stamp == 0.0)
    with pytest.raises(ValueError):
        sun_position_offsets(t, [0.0], [0.0], mode="noon")


@pytest.fixture(scope="module")
def sam():
    m = Pvlib.__new__(Pvlib)
    return Pvlib.retrieve_sam(m, "CECMod")[MODULE], Pvlib.retrieve_sam(m, "CECInverter")[INVERTER]


@pytest.mark.parametrize("ratio", [1.1, 1.25, 1.3])
def test_plant_reaches_dc_ac_ratio(sam, ratio):
    mod, inv = sam
    p = PlantSpec(mod, inv, dc_ac_ratio=ratio)
    assert abs(p.achieved_dc_ac / ratio - 1) <= 0.01
    assert p.dc_nameplate_w / p.ac_nameplate_w == pytest.approx(p.achieved_dc_ac)


def test_plant_orientation_both_hemispheres(sam):
    mod, inv = sam
    p = PlantSpec(mod, inv)
    tilt_n, az_n = p.orientation(30.0)
    tilt_s, az_s = p.orientation(-30.0)
    assert az_n == 180 and az_s == 0 and tilt_n == pytest.approx(tilt_s)
    fixed = PlantSpec(mod, inv, tilt=20, azimuth=170)
    assert fixed.orientation(-30.0) == (20.0, 170.0)
    tracker = PlantSpec(mod, inv, mount="single_axis")
    assert type(tracker.mount_for(10.0)).__name__ == "SingleAxisTrackerMount"
    with pytest.raises(ValueError):
        PlantSpec(mod, inv, mount="dual_axis")


def _synthetic_solar_ds(lat, lon):
    time = pd.date_range("2020-06-21 00:00", periods=24, freq="h")
    rng = np.random.default_rng(1)
    shape = (24, 1, 1)
    ds = xr.Dataset(
        {
            "influx_diffuse": (("time", "y", "x"), rng.uniform(50, 200, shape)),
            "influx_direct": (("time", "y", "x"), rng.uniform(100, 700, shape)),
            "dewpoint_temperature": (("time", "y", "x"), np.full(shape, 280.0)),
            "temperature": (("time", "y", "x"), np.full(shape, 295.0)),
            "wnd100m": (("time", "y", "x"), np.full(shape, 6.0)),
            "wnd10m": (("time", "y", "x"), np.full(shape, 3.0)),
            "albedo": (("time", "y", "x"), np.full(shape, 0.2)),
            "height": (("time", "y", "x"), np.full(shape, 500.0)),
        },
        coords={"time": time, "y": [lat], "x": [lon]},
    )
    return ds


@pytest.mark.parametrize("lat,lon", [(34.3, 108.9), (-23.5, -46.6), (40.0, -105.0)])
def test_plant_cf_ac_and_cf_dc(sam, lat, lon):
    mod, inv = sam
    model = Pvlib.__new__(Pvlib)
    model.init_model_config(
        clearsky_model="haurwitz", transposition_model="perez", solar_position_method="nrel_numpy",
        airmass_model="kastenyoung1989", dc_model="cec", ac_model="sandia", aoi_model="physical",
        spectral_model="first_solar", dc_ohmic_model="no_loss",
    )
    for mount in ("fixed", "single_axis"):
        plant = model.init_plant(mod, inv, dc_ac_ratio=1.3, mount=mount)
        out = model._pvlib_model(_synthetic_solar_ds(lat, lon), plant, model.config,
                                 n_jobs=1, sun_position="mid_hour")
        assert set(out.data_vars) == {"ac", "cf_ac", "cf_dc"}
        cf_ac, cf_dc = out.cf_ac.values.ravel(), out.cf_dc.values.ravel()
        assert 0.0 <= cf_ac.min() and cf_ac.max() <= 1.0 + 1e-9
        assert cf_ac.max() > 0.1
        day = cf_dc > 0
        np.testing.assert_allclose(cf_ac[day] / cf_dc[day], plant.achieved_dc_ac)
        assert model._cooling_wind_source == "ERA5 10 m wind"


def test_model_level_wind_from_fixture():
    ws = model_level_wind(FIXTURE_3D, slice(100, 104), slice(34, 30))
    assert ws.dims == ("time", "y", "x") and ws.shape == (3, 5, 5)
    assert ws.attrs["source"] == "ERA5 model level 137 wind"
    assert float(ws.min()) >= 0.0
    with pytest.raises(ValueError):
        model_level_wind(FIXTURE_3D, level=120)
