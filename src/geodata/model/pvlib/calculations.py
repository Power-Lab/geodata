import numpy as np
import pandas as pd
import xarray as xr

from pvlib.atmosphere import gueymard94_pw
from pvlib.solarposition import get_solarposition



def calculate_pvlib_solarposition(ds: xr.Dataset) -> pd.DataFrame:
    """
    Wrapper for :code:`pvlib.solarposition.get_solarposition()`.  
    Allows for vectorized calculation of solar position across an xarray dataset.
    The solar zenith angle is a required input for :code:`_calculate_ghi()`.

    For full documentation on how :code:`pvlib.solarposition.get_solarposition()` calculates precipitable water,
    see: `the pvlib API reference for pvlib.solarposition.get_solarposition() <https://pvlib-python.readthedocs.io/en/v0.4.2/generated/pvlib.solarposition.get_solarposition.html>`.

    Parameters
    ----------
    ds : xarray dataset
        An xarray dataset containing series for both influx diffuse (dhi) and influx direct (dni).
    zenith : numeric
        Zenith angle of the sun in degrees, as calculated by :code:`_calculate_pvlib_solarposition()`.

    Returns
    -------
    solarposition : dataframe
        Dataframe containing solar zenith angle for a given time and set of coordinates.
    """
    nt, ny, nx = ds.sizes['time'], ds.sizes['y'], ds.sizes['x']
    time_expanded = np.broadcast_to(ds.time.values[:, None, None], (nt, ny, nx)).ravel()
    yy, xx = np.meshgrid(ds.y, ds.x, indexing="ij")
    x_expanded = np.tile(xx.ravel(), nt)
    y_expanded = np.tile(yy.ravel(), nt)
    solarposition = get_solarposition(time_expanded, y_expanded, x_expanded) # might return a pandas DataFrame or a NDArray
    multi_index = pd.MultiIndex.from_arrays([time_expanded, y_expanded, x_expanded], names=['time', 'y', 'x'])
    if isinstance(solarposition, pd.DataFrame):
        solarposition = solarposition.set_index(multi_index)
    else:
        solarposition = pd.DataFrame(solarposition)
        solarposition.index = multi_index
    return solarposition

def _solpos_flat(times: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    """Apparent and true zenith (degrees) for flat arrays of UTC times and coordinates."""
    sp = get_solarposition(pd.DatetimeIndex(times), lat, lon)
    return (
        np.array(sp["apparent_zenith"], dtype=float),
        np.array(sp["zenith"], dtype=float),
    )


def sun_position_offsets(
    time,
    lat,
    lon,
    mode: str = "mid_hour",
    transition_band_deg: float = 10.0,
    n_sub: int = 6,
):
    """Time at which to evaluate the sun's position for each hour, cell by cell.

    ERA5 radiation at a time stamp is the average over the hour ending at the stamp,
    so the sun's position should describe that hour, not the stamp itself
    (Plan 4.3). With ``mode="mid_hour"`` the position is taken 30 minutes before
    the stamp, as in atlite. In sunrise and sunset hours the sun is up for only
    part of the hour, so the position is taken at the middle of the sunlit part
    instead: the hour is sampled every ``60 / n_sub`` minutes and the sunlit
    samples (apparent zenith below 90 degrees) are averaged. Only hours whose
    mid-hour apparent zenith is within ``transition_band_deg`` of 90 degrees are
    sampled; the sun moves at most 7.5 degrees in half an hour, so all other
    hours are entirely day or night.

    ``mode="stamp"`` reproduces geodata's earlier behaviour (position at the stamp).

    Parameters
    ----------
    time : array of datetime64, shape (nt,)
        Hour-ending time stamps in UTC.
    lat, lon : arrays, shapes (ny,) and (nx,)
        Cell-centre coordinates in degrees.

    Returns
    -------
    offset_min, apparent_zenith, zenith : arrays of shape (nt, ny, nx)
        Minutes before the stamp at which the position is evaluated, and the
        apparent and true solar zenith (degrees) at that time.
    """
    time = np.asarray(time, dtype="datetime64[ns]")
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    shape = (time.size, lat.size, lon.size)
    t = np.broadcast_to(time[:, None, None], shape).ravel()
    la = np.broadcast_to(lat[None, :, None], shape).ravel()
    lo = np.broadcast_to(lon[None, None, :], shape).ravel()

    def minutes(m):
        return (np.asarray(m, dtype=float) * 60e9).astype("timedelta64[ns]")

    if mode == "stamp":
        offset = np.zeros(t.size)
        app, zen = _solpos_flat(t, la, lo)
    elif mode == "mid_hour":
        offset = np.full(t.size, 30.0)
        app, zen = _solpos_flat(t - minutes(30.0), la, lo)
        tr = np.abs(app - 90.0) < transition_band_deg
        if tr.any():
            before = 60.0 - (np.arange(n_sub) + 0.5) * 60.0 / n_sub  # e.g. 55, 45, ..., 5
            up = np.zeros((n_sub, int(tr.sum())), dtype=bool)
            for k, m in enumerate(before):
                a, _ = _solpos_flat(t[tr] - minutes(m), la[tr], lo[tr])
                up[k] = a < 90.0
            n_up = up.sum(axis=0)
            eff = np.where(
                n_up > 0, (before[:, None] * up).sum(axis=0) / np.maximum(n_up, 1), 30.0
            )
            offset[tr] = eff
            a, z = _solpos_flat(t[tr] - minutes(eff), la[tr], lo[tr])
            app[tr], zen[tr] = a, z
    else:
        raise ValueError(f"sun_position must be 'mid_hour' or 'stamp', got {mode!r}")

    return offset.reshape(shape), app.reshape(shape), zen.reshape(shape)


def calculate_ghi(
    ds: xr.Dataset, 
    zenith: pd.Series
) -> xr.DataArray:
    """
    Calculates global horizontal irradiance (ghi). Negative values are clipped.

    ERA5's ``influx_direct`` (``fdir``) is the direct irradiance received on a
    HORIZONTAL plane (beam horizontal irradiance, BHI), not the direct NORMAL
    irradiance (DNI). GHI is therefore the plain sum of the two horizontal
    components -- no zenith projection is involved:

    .. math::

    GHI = DHI + BHI

    (Multiplying ``influx_direct`` by cos(zenith), as this function previously
    did, double-applies the projection and understates GHI everywhere the sun
    is not at the zenith.) If the dataset still carries ERA5's ``ssrd``
    (surface solar radiation downwards, which *is* GHI by definition), that
    value is used directly instead of re-deriving it.

    Parameters
    ----------
    ds : xarray dataset
        An xarray dataset containing series for influx diffuse (DHI) and
        influx direct (BHI), and optionally ssrd.
    zenith : numeric
        Zenith angle of the sun in degrees, as calculated by
        :code:`calculate_pvlib_solarposition()`. Only used to validate that
        the dataset is non-empty; the GHI arithmetic itself is
        zenith-independent.

    Returns
    -------
    ghi : numeric
        Global horizontal irradiance (ghi) [W m**-2].

    """
    # Convert zenith to numpy array if it's a pandas Series or xarray DataArray
    if isinstance(zenith, (pd.Series, xr.DataArray)):
        zenith_vals = zenith.values  # type: ignore[union-attr]
    else:
        zenith_vals = zenith
    # Ensure zenith_vals is a numpy array for type checking
    zenith_vals = np.asarray(zenith_vals)

    dhi = ds.influx_diffuse.values.ravel()
    bhi = ds.influx_direct.values.ravel()

    if zenith_vals.size == 0:
        x_coord = ds.coords.get("x")
        if x_coord is None:
            x_coord = ds.coords.get("longitude") if "longitude" in ds.coords else ds.coords.get("lon")
        y_coord = ds.coords.get("y")
        if y_coord is None:
            y_coord = ds.coords.get("latitude") if "latitude" in ds.coords else ds.coords.get("lat")
        x_vals = np.asarray(x_coord.values) if x_coord is not None else np.array([])
        y_vals = np.asarray(y_coord.values) if y_coord is not None else np.array([])
        time_size = ds.sizes.get("time", 0)

        def _fmt_coord(arr: np.ndarray, max_show: int = 20) -> str:
            if len(arr) == 0:
                return "[]"
            if len(arr) <= max_show:
                return str(arr.tolist())
            return f"[{arr.min():g}..{arr.max():g}] (length={len(arr)})"

        raise ValueError(
            "Cannot calculate GHI: dataset has no data points. "
            "This typically occurs when xs/ys slices do not overlap with the dataset's "
            "coordinates, or when the loaded dataset is empty. "
            f"Dataset dimensions: time={time_size}, x={_fmt_coord(x_vals)}, "
            f"y={_fmt_coord(y_vals)}. "
            "Check that your xs and ys values (in degrees) overlap with these coordinate ranges."
        )

    if "ssrd" in ds.data_vars:
        # ERA5's ssrd IS the global horizontal irradiance; use it directly.
        ghi = np.clip(ds.ssrd.values.ravel(), 0, np.inf)
    else:
        # Both components are horizontal-plane fluxes, so GHI is their sum.
        ghi = np.clip(
            dhi + bhi,
            0,
            np.inf # `np.Inf` was removed in the NumPy 2.0 release.
        )

    reshaped_ghi = ghi.reshape(
        ds.sizes['time'], 
        ds.sizes['y'], 
        ds.sizes['x']
    )
    
    ghi = xr.DataArray(
        reshaped_ghi,
        dims=("time", "y", "x"),
        coords={
            "time": ds['time'].values, 
            "y": ds['y'].values, 
            "x": ds['x'].values
        },
        name="ghi"
    )

    ghi.name = "ghi"
    ghi.attrs["units"] = "W m**-2"
    ghi.attrs["description"] = (
        "GHI taken from ssrd when present, otherwise the sum of influx "
        "diffuse (DHI) and influx direct (BHI, horizontal)."
    )
    return ghi


def calculate_dni(
    ds: xr.Dataset,
    zenith: pd.Series,
    zenith_max_deg: float = 88.0,
) -> xr.DataArray:
    """
    Derives direct normal irradiance (DNI) from ERA5's horizontal beam
    irradiance.

    ERA5's ``influx_direct`` (``fdir``) is the direct irradiance received on a
    HORIZONTAL plane (BHI), while pvlib's ModelChain expects true DNI (the
    flux on a plane normal to the sun). The two are related through the solar
    zenith angle Z:

    .. math::

    DNI = BHI / cos(Z)

    Passing ``influx_direct`` to pvlib as ``dni`` unconverted (as this module
    previously did) understates DNI by the factor cos(Z) -- e.g. by half at
    Z = 60 degrees -- so plane-of-array irradiance and AC output are biased
    low everywhere the sun is off-zenith.

    To avoid the 1/cos(Z) blow-up at grazing angles (where ERA5's
    hourly-integrated fdir and an instantaneous zenith angle are numerically
    incompatible), DNI is set to 0 wherever Z >= ``zenith_max_deg`` (default
    88 degrees, the same cutoff pvlib.irradiance.dni uses). Negative input
    values are clipped to 0.

    Parameters
    ----------
    ds : xarray dataset
        An xarray dataset containing influx direct (BHI).
    zenith : numeric
        Zenith angle of the sun in degrees, as calculated by
        :code:`calculate_pvlib_solarposition()`.
    zenith_max_deg : float
        Zenith angle in degrees at and beyond which DNI is set to 0.

    Returns
    -------
    dni : numeric
        Direct normal irradiance (dni) [W m**-2].

    """
    if isinstance(zenith, (pd.Series, xr.DataArray)):
        zenith_vals = zenith.values  # type: ignore[union-attr]
    else:
        zenith_vals = zenith
    # zenith comes from calculate_pvlib_solarposition(), which wraps
    # pvlib.solarposition.get_solarposition() -- pvlib documents this as
    # always returning angles in degrees.
    zenith_vals = np.asarray(zenith_vals, dtype=float)

    bhi = np.clip(ds.influx_direct.values.ravel().astype(float), 0, np.inf)

    sun_up = zenith_vals < zenith_max_deg
    dni = np.zeros_like(bhi)
    np.divide(bhi, np.cos(np.deg2rad(zenith_vals)), out=dni, where=sun_up)

    reshaped_dni = dni.reshape(
        ds.sizes['time'],
        ds.sizes['y'],
        ds.sizes['x']
    )

    dni = xr.DataArray(
        reshaped_dni,
        dims=("time", "y", "x"),
        coords={
            "time": ds['time'].values,
            "y": ds['y'].values,
            "x": ds['x'].values
        },
        name="dni"
    )

    dni.attrs["units"] = "W m**-2"
    dni.attrs["description"] = (
        "DNI derived from influx direct (BHI, horizontal) as BHI / cos(zenith), "
        f"set to 0 for zenith >= {zenith_max_deg} degrees."
    )
    return dni

def calculate_relative_humidity(
    temperature: xr.DataArray, 
    dewpoint_temperature: xr.DataArray
) -> xr.DataArray:
        """
        Calculates relative humidity based on air temperature and dewpoint temperature.
        Needed in order to calculate precipitable water using pvlib's :code:`gueymard94_pw()` function.

        Relative humidity is calculated using a version of the 
        August-Roche-Magnus equation as follows: 
        
        .. math::

            RH = 100 \cdot \frac{{\exp\left(\frac{{17.625 \cdot TD}}{{243.04 + TD}}\right)}}{{\exp\left(\frac{{17.625 \cdot T}}{{243.04 + T}}\right)}}

        where, RH is % relative humidity, TD is dew-point temperature (celsius), and T is air temperature (celsius).[#1]_ [#2]_

        Parameters
        ----------
        temperature : numeric
            Ambient air temperature at the surface. [C]
        dewpoint_temperature : numeric
            Dewpoint temperature at the surface. [C]

        Returns
        -------
        relative_humidity : numeric
            Percent relative humidity. [%]

        References
        ----------
        .. [#1] `United States Environmental Protection Agency. Hydrologic Micro Services. Meteorology - Humidity.  <https://qed.epa.gov/hms/meteorology/humidity/algorithms/>`_

        .. [#2] `University of Miami. Calculate Temperature, Dewpoint, or Relative Humidity. <https://bmcnoldy.earth.miami.edu/Humidity.html>` 

        """
        relative_humidity = 100 * (
            np.exp((17.625 * dewpoint_temperature) / (243.04 + dewpoint_temperature)) /
            np.exp((17.625 * temperature) / (243.04 + temperature))
        )
        
        # Ensure result is xarray DataArray (arithmetic operations preserve xarray types)
        if not isinstance(relative_humidity, xr.DataArray):
            relative_humidity = xr.DataArray(relative_humidity)

        relative_humidity.name = "relative_humidity"
        relative_humidity.attrs["units"] = "%"
        relative_humidity.attrs["description"] = "Relative humidity, calculated using temperature and dewpoint temperature."

        return relative_humidity

def calculate_precipitable_water(
    temperature: xr.DataArray, 
    relative_humidity: xr.DataArray
) -> xr.DataArray:
    """
    Calculates precipitable water (cm) from ambient air temperature (C) and relative humidity (%) using 
    :code:`pvlib.atmosphere.gueymard94_pw()`.  

    Precipitable water (cm) is a required input for models using CEC modules from :code:`pvlib`.
    For full documentation on how :code:`pvlib.atmosphere.gueymard94_pw()` calculates precipitable water,
    see: `the pvlib API reference for pvlib.atmosphere.gueymard94_pw() <https://pvlib-python.readthedocs.io/en/v0.4.2/generated/pvlib.atmosphere.gueymard94_pw.html>`.

    Parameters
    ----------
    temperature : numeric
        Ambient air temperature at the surface. [C]
    relative_humidity : numeric
        Percent relative humidity. [%]

    Returns
    -------
    precipitable_water : numeric
        Precipitable water (cm) calculated from ambient air temperature (C) and relative humidity (%). [cm]

    """
    # Use xarray's apply_ufunc to preserve DataArray type when calling external function
    precipitable_water = xr.apply_ufunc(
        gueymard94_pw,
        temperature,
        relative_humidity,
        dask="allowed",
        output_dtypes=[float]
    )
    precipitable_water.name = "precipitable_water"
    precipitable_water.attrs["units"] = "cm"
    precipitable_water.attrs["description"] = "Precipitable water (cm) calculated from ambient air temperature (C) and relative humidity (%)."

    return precipitable_water

def convert_kelvin_to_celsius(
    ds: xr.DataArray
) -> xr.DataArray:
    """
    Converts a temperature in Kelvin to a temperature in Celsius.

    Parameters
    ----------
    ds : numeric or xarray.DataArray
        A temperature in Kelvin [K].

    Returns
    -------
    temperature : numeric or xarray.DataArray
        A temperature in Celsius [C].
    """
    return ds - 273.15


def latitude_optimal_orientation(lat: float) -> dict:
    """
    Recommends a fixed ``surface_tilt``/``surface_azimuth`` pair (in the
    degrees convention expected by :code:`pvlib.pvsystem.PVSystem`) for a
    latitude-optimal, equator-facing panel.

    Tilt follows the rule-of-thumb piecewise fit used in gsee [1]_: below
    25 degrees of absolute latitude, tilt scales at 0.87x latitude;
    between 25 and 50 degrees, at 0.76x latitude plus 3.1 degrees; above
    50 degrees, a flat 40 degrees. Unlike geodata's legacy implementation
    (:code:`geodata.pv.orientation.make_latitude_optimal`), this works for
    both hemispheres: a panel should face the equator, so ``azimuth`` is
    180 (south-facing) in the northern hemisphere but 0 (north-facing) in
    the southern hemisphere -- the legacy function instead raises
    ``NotImplementedError`` for any negative latitude.

    This single tilt/azimuth pair is meant for callers that need one
    representative orientation for a whole :code:`pvlib.pvsystem.PVSystem`
    (see :code:`Pvlib.init_pv_system()`), which applies uniformly to every
    coordinate processed in one :code:`Pvlib.estimate()` call. It is only
    a good approximation for a run whose bounding box sits within a single
    hemisphere; a bounding box straddling the equator (e.g. Indonesia)
    should be split into per-hemisphere (or per-province) runs, each
    calling this function with its own representative latitude, rather
    than averaging across the equator.

    References
    ----------
    .. [1] https://github.com/renewables-ninja/gsee/blob/master/gsee/pv.py

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees. Positive for the northern hemisphere,
        negative for the southern hemisphere.

    Returns
    -------
    orientation : dict
        Dictionary with ``surface_tilt`` and ``surface_azimuth`` in
        decimal degrees, ready to pass as keyword arguments to
        :code:`Pvlib.init_pv_system()`.
    """
    abs_lat = abs(lat)

    if abs_lat <= 25:
        tilt = 0.87 * abs_lat
    elif abs_lat <= 50:
        tilt = 0.76 * abs_lat + 3.1
    else:
        tilt = 40.0

    azimuth = 180.0 if lat >= 0 else 0.0

    return {"surface_tilt": tilt, "surface_azimuth": azimuth}