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

"""Unit tests for the pvlib model's capacity normalization.

``_pvlib_model`` normalizes the inverter's AC output to a per-unit ``pv``
value by the array's PTC nameplate. The nameplate must count EVERY module
feeding the inverter -- ``modules_per_string * strings_per_inverter`` -- not
just one string. Historically only ``modules_per_string`` was used, so any
system with more than one string per inverter had ``pv`` overstated by
exactly the string count (a 4-string system reported a 4x capacity factor).

Offline: synthetic single-cell weather, real pvlib ModelChain.
"""

import numpy as np
import pandas as pd
import xarray as xr

from geodata.datasets import load_dataset
from geodata.model.pvlib import Pvlib


MODULE_NAME = "Jinko_Solar__Co___Ltd_JKM400M_72L"
INVERTER_NAME = "Sungrow_Power_Supply_Co___Ltd___SG100KU__480V_"
MODULES_PER_STRING = 50
LAT, LON = 34.3, 108.9  # Xi'an


def _synthetic_solar_ds(lat: float, lon: float) -> xr.Dataset:
    """Single-cell dataset over a few daytime-heavy hours."""
    time = pd.date_range("2020-06-21 00:00", periods=24, freq="h")
    nt = len(time)
    rng = np.random.default_rng(20260822)
    return xr.Dataset(
        {
            "influx_diffuse": (["time", "y", "x"], rng.uniform(50, 200, (nt, 1, 1))),
            "influx_direct": (["time", "y", "x"], rng.uniform(100, 700, (nt, 1, 1))),
            "dewpoint_temperature": (["time", "y", "x"], rng.uniform(270, 285, (nt, 1, 1))),
            "temperature": (["time", "y", "x"], rng.uniform(285, 300, (nt, 1, 1))),
            "wnd100m": (["time", "y", "x"], rng.uniform(1, 10, (nt, 1, 1))),
        },
        coords={"time": time, "y": [lat], "x": [lon]},
    )


def _run(strings_per_inverter: int):
    """Run the real _pvlib_model path with the given string count and return
    (result dataset, module PTC)."""
    fixture_cls = load_dataset("wind_solar_hourly_test")
    fixture = fixture_cls(years=slice(2016, 2016), months=slice(1, 1))
    model = Pvlib(fixture)

    module = model.retrieve_sam("CECMod")[MODULE_NAME]
    inverter = model.retrieve_sam("CECInverter")[INVERTER_NAME]

    model.init_model_config(
        clearsky_model="haurwitz",
        transposition_model="perez",
        solar_position_method="nrel_numpy",
        airmass_model="kastenyoung1989",
        dc_model="cec",
        ac_model="sandia",
        aoi_model="physical",
        spectral_model="first_solar",
        dc_ohmic_model="no_loss",
    )
    model.init_pv_system(
        arrays=None,
        surface_tilt=29.0,
        surface_azimuth=180.0,
        racking_model="open_rack",
        module_parameters=module,
        modules_per_string=MODULES_PER_STRING,
        module_type="glass_polymer",
        module=MODULE_NAME,
        strings_per_inverter=strings_per_inverter,
        inverter_parameters=inverter,
    )

    ds = _synthetic_solar_ds(LAT, LON)
    result = model._pvlib_model(
        ds, model.pv_system, model.config, n_jobs=1, compact_output=True
    )
    ptc = model.pv_system.arrays[0].module_parameters["PTC"]
    return result, ptc


def test_pv_divides_by_total_modules_across_strings():
    """With strings_per_inverter=4, pv must equal ac / (PTC * 50 * 4).

    The historical normalization divided by modules_per_string alone
    (PTC * 50), overstating pv by exactly 4x.
    """
    result, ptc = _run(strings_per_inverter=4)
    ac = result["ac"].values.ravel()
    pv = result["pv"].values.ravel()

    assert np.isfinite(ac).all() and ac.max() > 0, "expected some AC output"
    np.testing.assert_allclose(
        pv, ac / (ptc * MODULES_PER_STRING * 4), rtol=1e-12
    )
    # The buggy arithmetic yields exactly 4x these values.
    buggy = ac / (ptc * MODULES_PER_STRING)
    assert not np.allclose(pv[ac > 0], buggy[ac > 0]), (
        "pv still normalized by one string's modules only"
    )


def test_pv_per_unit_output_independent_of_string_count():
    """Per-unit pv must be comparable between 1-string and 4-string systems
    running identical weather: quadrupling the array size roughly quadruples
    AC output but also quadruples the nameplate. With the historical
    normalization the 4-string system reported ~4x the capacity factor."""
    result_1, _ = _run(strings_per_inverter=1)
    result_4, _ = _run(strings_per_inverter=4)

    pv_1 = result_1["pv"].values.ravel()
    pv_4 = result_4["pv"].values.ravel()

    sum_1, sum_4 = pv_1.sum(), pv_4.sum()
    assert sum_1 > 0, "expected some production in the 1-string run"
    ratio = sum_4 / sum_1
    # Inverter efficiency varies with loading, so allow a generous band --
    # the buggy code produced a ratio of ~4.
    assert 0.5 < ratio < 2.0, (
        f"pv should be per-unit (ratio ~1), got 4-string/1-string = {ratio:.3f}"
    )
