# Copyright 2026 Power Lab (UCSD)

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Air density at hub height for the wind power conversion (Plan 4.1, 4.2).

Power curves are defined at standard air density (1.225 kg/m3). Following the
turbine power-performance standard IEC 61400-12-1 (and PLUSWIND), wind speed is
scaled by (rho / 1.225) ** (1/3) before the power curve is applied, with rho the
air density at hub height.

rho is estimated from ERA5 surface pressure (``sp``) and 2 m temperature (``t2m``),
which are in the single-level files geodata already downloads for solar:

- temperature falls with height at the standard lapse rate (6.5 K/km);
- pressure falls hydrostatically over the 2 m-to-hub layer at its mean temperature;
- dry-air ideal gas law (humidity is ignored, under about 1% in rho).
"""

from __future__ import annotations

import glob
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr

from ...utils import to_lon180

RHO_STD = 1.225  # kg/m3, density of the standard power curves
R_DRY = 287.05  # J/(kg K)
G0 = 9.80665  # m/s2
LAPSE = 0.0065  # K/m


def hub_air_density(sp, t2m, hub_height: float):
    """Air density (kg/m3) at ``hub_height`` metres above ground.

    Args:
        sp: Surface pressure (Pa).
        t2m: Air temperature at 2 m (K).
        hub_height: Height above ground (m).

    Works on numbers, numpy arrays and DataArrays.
    """
    dz = float(hub_height) - 2.0
    t_hub = t2m - LAPSE * dz
    t_mean = 0.5 * (t2m + t_hub)
    p_hub = sp * np.exp(-G0 * (float(hub_height)) / (R_DRY * t_mean))
    rho = p_hub / (R_DRY * t_hub)
    if isinstance(rho, xr.DataArray):
        rho = rho.rename("air_density")
        rho.attrs.update(units="kg m**-3", hub_height_m=float(hub_height),
                         source="ERA5 sp and t2m, standard lapse rate, dry air")
    return rho


def surface_pressure_temperature(
    files: str | Path | Iterable[str | Path],
    xs: slice | None = None,
    ys: slice | None = None,
) -> xr.Dataset:
    """ERA5 ``sp`` and ``t2m`` from geodata's single-level (wind_solar) files.

    Args:
        files: A file, a glob pattern, or a list of files (e.g. one month).
        xs, ys: Optional longitude (-180..180) and latitude slices.

    Returns:
        Dataset with ``sp`` (Pa) and ``t2m`` (K), dims (time, y, x), loaded in memory;
        usable as the ``air_density`` argument of the wind models' ``estimate``.
    """
    if isinstance(files, (str, Path)):
        files = sorted(glob.glob(str(files))) or [str(files)]
    files = [str(f) for f in files]
    if not files:
        raise FileNotFoundError("No single-level files given.")

    with xr.open_mfdataset(files, engine="h5netcdf", combine="by_coords") as ds:
        rename = {k: v for k, v in (("valid_time", "time"), ("longitude", "x"),
                                    ("latitude", "y")) if k in ds.dims}
        ds = ds.rename(rename)
        missing = [v for v in ("sp", "t2m") if v not in ds]
        if missing:
            raise ValueError(f"Single-level files lack {missing}; air density needs sp and t2m.")
        ds = to_lon180(ds[["sp", "t2m"]], "x")
        if xs is not None:
            ds = ds.sel(x=_ordered(ds["x"], xs))
        if ys is not None:
            ds = ds.sel(y=_ordered(ds["y"], ys))
        out = ds.load()

    out = out.drop_vars([c for c in out.coords if c not in ("time", "y", "x")])
    out = out.transpose("time", "y", "x")
    out.attrs.update(source="ERA5 single levels (sp, t2m)")
    return out


def align_to(field: xr.DataArray, like: xr.DataArray) -> xr.DataArray:
    """Select ``field`` on the time stamps and grid cells of ``like``.

    Time must match exactly; cells are matched to the nearest point within 1e-3 degrees,
    so the 0.25 degree grids of ERA5's model-level and single-level files line up.
    """
    if "valid_time" in field.dims:
        field = field.rename({"valid_time": "time"})
    missing = np.setdiff1d(like["time"].values, field["time"].values)
    if missing.size:
        raise ValueError(
            f"Air-density input lacks {missing.size} of the wind time stamps "
            f"(first: {missing[0]})."
        )
    out = field.sel(time=like["time"])
    out = out.sel(y=like["y"], x=like["x"], method="nearest", tolerance=1e-3)
    return out.assign_coords(time=like["time"], y=like["y"], x=like["x"])


def _ordered(coord: xr.DataArray, s: slice) -> slice:
    """Order a slice to match an ascending or descending coordinate."""
    lo, hi = s.start, s.stop
    if lo is None or hi is None:
        return s
    descending = coord.size > 1 and float(coord[0]) > float(coord[-1])
    a, b = sorted((lo, hi))
    return slice(b, a) if descending else slice(a, b)
