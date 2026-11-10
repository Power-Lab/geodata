# Copyright 2023, 2025 Michael Davidson (UCSD), Xiqiang Liu (UCSD)

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

"""Wind speed estimation based on the Model interface.

This module provides a wind speed estimation model based on the Model interface.

Extrapolation:
    The wind speed is estimated using a linear regression of the logarithm of the wind speed
    against the logarithm of the height. The coefficients of the regression are stored in the
    dataset as a 2D array of shape (2,). The first coefficient is the slope of the regression
    line, and the second coefficient is the intercept. The wind speed is then estimated as
    :math:`\\hat{v} = \\alpha \\cdot \\log(z) + \\beta`, where :math:`\\alpha` and :math:`\\beta`
    are the coefficients, and :math:`z` is the height.

Residuals:
    The residuals of the regression are stored in the dataset as a 2D array of shape (n,).
    The residuals are the difference between the estimated wind speed and the actual wind speed.
    The residuals are stored in the same order as the heights used in the regression.

Example:
    >>> from geodata import Cutout
    >>> from geodata.model import WindModel
    >>> cutout = Cutout("merra2", 2010, 2010, 0, 0, 0, 0)
    >>> model = WindModel(cutout)
    >>> model.prepare()
    >>> model.estimate(xs=slice(1, 2), ys=slice(1, 2), years=slice(2010, 2010), months=slice(1, 2))
"""

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from ...logging import logger
from ...resource import resolve_windturbine
from .._base import BaseModel

HEIGHTS = {"u50m": 50, "u10m": 10, "u2m": 2}


class WindBaseModel(BaseModel):
    """Base class for wind speed estimation (interpolation/extrapolation).
    Not meant to be instantiated directly.

    Args:
        source (Union[Dataset, Cutout]): Source dataset or cutout.
        **kwargs: Keyword arguments.
    """

    type: str = "wind"

    def estimate(
        self,
        years: slice | None = None,
        months: slice | None = None,
        xs: slice | None = None,
        ys: slice | None = None,
        **kwargs,
    ) -> xr.DataArray:
        """Estimate wind speed or CF at the given locations and times. If a turbine is
        specified, the CF is calculated based on the wind speed and the turbine's power curve.
        Otherwise, pass in the `height` keyword argument to estimate wind speed at a specific height.

        Args:
            years (slice, optional): Years.
            months (slice, optional): Months. If None, all months are estimated.
            xs (slice, optional): X coordinates. If None, all x coordinates in source are estimated.
            ys (slice, optional): Y coordinates. If None, all y coordinates in source are estimated.
            **kwargs: Additional keyword arguments to pass to the model.
                - `turbine` (str): Name of the wind turbine to estimate power output.
                - `height` (int): Height at which to estimate wind speed. If not specified, the model will use the default height.

        Raises:
            ValueError: If neither 'turbine' nor 'height' is specified in kwargs.

        Returns:
            xr.DataArray: Estimated wind speed.
        """

        if "turbine" in kwargs:
            return self._estimate_power(
                years=years, months=months, xs=xs, ys=ys, **kwargs
            )

        if "height" not in kwargs:
            raise ValueError(
                "Either 'turbine' or 'height' must be specified to estimate wind speed."
            )

        return super().estimate(years, months, xs, ys, **kwargs)

    def _estimate_power(
        self,
        turbine: str | dict,
        xs: slice | None = None,
        ys: slice | None = None,
        years: slice | None = None,
        months: slice | None = None,
        hour_average: bool = True,
    ) -> xr.DataArray:
        """Estimate capacity factor (raw: no air-density or loss adjustment).

        Args:
            turbine (str | dict): A turbine YAML name in geodata's library, or a dict
                defining a generic turbine by specific power or a specific model's
                curve; see :func:`geodata.resource.resolve_windturbine`.
            xs (slice, optional): X slice. Defaults to None.
            ys (slice, optional): Y slice. Defaults to None.
            years (slice, optional): Year slice. Defaults to None.
            months (slice, optional): Month slice. Defaults to None.
            hour_average (bool, optional): Report each hour as the average of the
                instantaneous CF at the start and end of the hour (PLUSWIND), so the
                value at a time stamp is the average over the hour ending at it.
                ERA5 wind is instantaneous at each stamp; reported generation is an
                hourly average. The first stamp has no preceding value and keeps its
                instantaneous CF. Defaults to True.

        Returns:
            xr.DataArray: Capacity factor named ``cf``.
        """

        # Get the wind turbine configuration
        try:
            turbineconf = resolve_windturbine(turbine)
        except FileNotFoundError:
            raise ValueError(f"Wind turbine configuration '{turbine}' not found.")

        speed = self.estimate(
            years=years, months=months, xs=xs, ys=ys, height=turbineconf["hub_height"]
        )

        interp_fn = interp1d(
            turbineconf["V"],
            turbineconf["POW"],
            bounds_error=False,
            fill_value=0.0,
        )

        # Calculate the power output. interp1d is already vectorised over arrays,
        # so no per-element Python loop is needed.
        power = xr.apply_ufunc(
            interp_fn,
            speed,
            dask="parallelized",
            output_dtypes=[float],
        )
        cf = power / turbineconf["P"]

        if hour_average and "time" in cf.dims and cf.sizes["time"] > 1:
            step = np.diff(cf["time"].values[:2])[0]
            if step == np.timedelta64(1, "h"):
                cf = cf.rolling(time=2, min_periods=1).mean()
            else:
                logger.warning(
                    "hour_average skipped: time step is %s, not 1 hour.", step
                )
                hour_average = False

        cf = cf.rename("cf")
        cf.attrs.update(
            {
                "long_name": "wind capacity factor (raw)",
                "units": "1",
                "hub_height_m": float(turbineconf["hub_height"]),
                "turbine": turbine if isinstance(turbine, str) else str(turbine),
                "hour_average": int(bool(hour_average)),
                "time_convention": (
                    "average over the hour ending at the stamp"
                    if hour_average
                    else "instantaneous at the stamp"
                ),
            }
        )
        return cf
