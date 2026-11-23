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

"""Wind speed near the panels for the PV cell-temperature model.

The SAPM cell-temperature model is calibrated with wind measured near 10 m. ERA5's
lowest model level (137, about 10 m above ground) is in the wind_3d files geodata
already downloads for wind power, so it can be used when the single-level files
lack u10/v10 (Plan 4.3).
"""

from __future__ import annotations

import glob
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr

from ...utils import to_lon180


def model_level_wind(
    files: str | Path | Iterable[str | Path],
    xs: slice | None = None,
    ys: slice | None = None,
    level: int = 137,
) -> xr.DataArray:
    """Wind speed (m/s) at one ERA5 model level from geodata's wind_3d files.

    Args:
        files: A wind_3d file, a glob pattern, or a list of files (e.g. one month
            of daily files).
        xs, ys: Optional longitude (-180..180) and latitude slices.
        level: ERA5 model level; 137 is the lowest, about 10 m above ground.

    Returns:
        DataArray named ``wind_speed`` with dims (time, y, x), usable as the
        ``cooling_wind`` argument of :meth:`geodata.model.pvlib.Pvlib.estimate`.
    """
    if isinstance(files, (str, Path)):
        files = sorted(glob.glob(str(files))) or [str(files)]
    files = [str(f) for f in files]
    if not files:
        raise FileNotFoundError("No wind_3d files given.")

    with xr.open_mfdataset(files, engine="h5netcdf", combine="by_coords") as ds:
        rename = {k: v for k, v in (("valid_time", "time"), ("longitude", "x"),
                                    ("latitude", "y")) if k in ds.dims}
        ds = ds.rename(rename)
        if "model_level" not in ds.coords or level not in ds["model_level"].values:
            raise ValueError(f"Model level {level} is not in the wind_3d files.")
        ds = to_lon180(ds.sel(model_level=level), "x")
        if xs is not None:
            ds = ds.sel(x=_ordered(ds["x"], xs))
        if ys is not None:
            ds = ds.sel(y=_ordered(ds["y"], ys))
        ws = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2).load()

    ws = ws.drop_vars([c for c in ws.coords if c not in ("time", "y", "x")])
    ws = ws.transpose("time", "y", "x").rename("wind_speed")
    ws.attrs.update(units="m s**-1", source=f"ERA5 model level {level} wind")
    return ws


def _ordered(coord: xr.DataArray, s: slice) -> slice:
    """Order a slice to match an ascending or descending coordinate."""
    lo, hi = s.start, s.stop
    if lo is None or hi is None:
        return s
    descending = coord.size > 1 and float(coord[0]) > float(coord[-1])
    a, b = sorted((lo, hi))
    return slice(b, a) if descending else slice(a, b)
