# Copyright 2016-2017 Gorm Andresen (Aarhus University), Jonas Hoersch (FIAS), Tom Brown (FIAS)

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


"""
GEODATA

Geospatial Data Collection and "Pre-Analysis" Tools
"""

import logging
from operator import itemgetter


import numpy as np
import yaml
from scipy.signal import fftconvolve
from geodata.config import SRC_ROOT


logger = logging.getLogger(name=__name__)


def get_windturbineconfig(turbine: str):
    """Load the 'turbine'.yaml file from local disk and provide a turbine dict.

    Args:
        turbine (str): Name of the wind turbine configuration file (without .yaml).

    Returns:
        dict: Dictionary containing the wind turbine configuration.
            - V: Wind speed array
            - POW: Power output array
            - hub_height: Hub height
            - P: Rated power output
    """

    res_name = "resources/windturbine/" + turbine + ".yaml"
    with open(SRC_ROOT / res_name, "r") as resource_file:
        turbineconf = yaml.safe_load(resource_file)
    V, POW, hub_height = itemgetter("V", "POW", "HUB_HEIGHT")(turbineconf)

    # Let's make sure that the wind speed and power output arrays are sorted
    V = np.asarray(V)
    POW = np.asarray(POW)

    sorted_indices = np.argsort(V)
    V = V[sorted_indices]
    POW = POW[sorted_indices]

    return {"V": V, "POW": POW, "hub_height": hub_height, "P": POW.max()}


GENERIC_CURVE_TABLE = "resources/windturbine/generic/sam_generic_power_curves.csv"
DEFAULT_CUT_IN = 3.0  # m/s, PLUSWIND's default
DEFAULT_CUT_OUT = 25.0  # m/s, PLUSWIND's default


def _apply_cut_speeds(V: np.ndarray, P: np.ndarray, cut_in: float, cut_out: float):
    """Zero the curve at or below cut-in and at or above cut-out (SAM's convention)."""
    P = np.where((V > cut_in) & (V < cut_out), P, 0.0)
    return V, P


def get_generic_windturbineconfig(
    specific_power: float,
    hub_height: float,
    cut_in: float = DEFAULT_CUT_IN,
    cut_out: float = DEFAULT_CUT_OUT,
) -> dict:
    """Generic power curve for a turbine of a given specific power (PLUSWIND method).

    Curves come from a table generated with NREL's SAM (via PySAM) using PLUSWIND's
    settings: maximum power coefficient 0.45, maximum tip-speed ratio 8, maximum tip
    speed 80 m/s, sea-level air density (see ``scripts/make_generic_power_curves.py``).
    Normalised by rated power, such a curve depends only on the specific power, so
    the table holds one curve per integer W/m2 from 150 to 600; non-integer values
    are interpolated linearly between neighbours. Power is zero at or below
    ``cut_in`` and at or above ``cut_out``.

    Args:
        specific_power: Rated power divided by rotor swept area, in W/m2.
        hub_height: Hub height in metres.
        cut_in: Cut-in wind speed in m/s.
        cut_out: Cut-out wind speed in m/s.

    Returns:
        dict: Same keys as :func:`get_windturbineconfig`, with POW as a fraction of
        rated power (so ``P`` = 1.0).
    """
    table = _generic_curve_table()
    sps = np.array([float(c) for c in table.columns[1:]])
    sp = float(specific_power)
    if not sps[0] <= sp <= sps[-1]:
        raise ValueError(
            f"Specific power {sp:.1f} W/m2 is outside the generic-curve table "
            f"({sps[0]:.0f}-{sps[-1]:.0f} W/m2). Extend SP_RANGE in "
            "scripts/make_generic_power_curves.py and regenerate the table."
        )
    lo = int(np.floor(sp))
    hi = min(lo + 1, int(sps[-1]))
    w = sp - lo
    P = (1 - w) * table[str(lo)].to_numpy() + w * table[str(hi)].to_numpy()
    V = table["wind_speed_ms"].to_numpy(dtype=float)
    if not cut_in < cut_out:
        raise ValueError(f"cut_in ({cut_in}) must be below cut_out ({cut_out}).")
    V, P = _apply_cut_speeds(V, P, float(cut_in), float(cut_out))
    return {"V": V, "POW": P, "hub_height": float(hub_height), "P": 1.0}


_GENERIC_TABLE_CACHE = {}


def _generic_curve_table():
    if "table" not in _GENERIC_TABLE_CACHE:
        import pandas as pd

        _GENERIC_TABLE_CACHE["table"] = pd.read_csv(SRC_ROOT / GENERIC_CURVE_TABLE)
    return _GENERIC_TABLE_CACHE["table"]


def get_windturbineconfig_from_csv(
    path, hub_height: float, cut_in: float | None = None, cut_out: float | None = None
) -> dict:
    """Power curve of a specific turbine model from a CSV file.

    The CSV needs columns ``wind_speed_ms`` and ``power_kw``. Power is returned as a
    fraction of the curve's maximum (taken as rated power). If ``cut_in`` or
    ``cut_out`` is given, power is set to zero at or below / at or above it.
    """
    import pandas as pd

    df = pd.read_csv(path)
    missing = {"wind_speed_ms", "power_kw"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
    df = df.sort_values("wind_speed_ms")
    V = df["wind_speed_ms"].to_numpy(dtype=float)
    POW = df["power_kw"].to_numpy(dtype=float)
    rated = POW.max()
    if rated <= 0:
        raise ValueError(f"{path}: power curve has no positive power")
    P = POW / rated
    if cut_in is not None or cut_out is not None:
        V, P = _apply_cut_speeds(
            V,
            P,
            -np.inf if cut_in is None else float(cut_in),
            np.inf if cut_out is None else float(cut_out),
        )
    return {"V": V, "POW": P, "hub_height": float(hub_height), "P": 1.0}


def resolve_windturbine(turbine) -> dict:
    """Turn any supported turbine definition into a power-curve dict.

    Accepted forms:

    - ``str``: name of a turbine YAML in geodata's library (uses its own hub height).
    - ``dict`` with ``specific_power`` (W/m2) and ``hub_height`` (m), optional
      ``cut_in`` / ``cut_out`` (m/s): generic SAM curve (PLUSWIND method).
    - ``dict`` with ``curve`` and ``hub_height``: a specific model's curve, where
      ``curve`` is a library YAML name or a CSV path (columns ``wind_speed_ms``,
      ``power_kw``); optional ``cut_in`` / ``cut_out``.
    - ``dict`` with ``name`` (library YAML) and optional ``hub_height`` override.

    Returns:
        dict with keys V, POW, hub_height, P (see :func:`get_windturbineconfig`).
    """
    if isinstance(turbine, str):
        return get_windturbineconfig(turbine)
    if not isinstance(turbine, dict):
        raise TypeError(f"turbine must be a str or dict, got {type(turbine).__name__}")

    t = dict(turbine)
    if "specific_power" in t or "sp" in t:
        sp = t.get("specific_power", t.get("sp"))
        if "hub_height" not in t:
            raise ValueError("A generic turbine needs 'hub_height'.")
        return get_generic_windturbineconfig(
            sp,
            t["hub_height"],
            cut_in=t.get("cut_in", DEFAULT_CUT_IN),
            cut_out=t.get("cut_out", DEFAULT_CUT_OUT),
        )
    curve = t.get("curve", t.get("name"))
    if curve is None:
        raise ValueError(
            "turbine dict needs 'specific_power' (generic curve) or 'curve'/'name' "
            "(specific model)."
        )
    curve = str(curve)
    if curve.lower().endswith(".csv"):
        if "hub_height" not in t:
            raise ValueError("A turbine defined by a CSV curve needs 'hub_height'.")
        return get_windturbineconfig_from_csv(
            curve, t["hub_height"], t.get("cut_in"), t.get("cut_out")
        )
    conf = get_windturbineconfig(curve)
    if "hub_height" in t:
        conf["hub_height"] = float(t["hub_height"])
    if t.get("cut_in") is not None or t.get("cut_out") is not None:
        V, POW = _apply_cut_speeds(
            conf["V"],
            conf["POW"],
            -np.inf if t.get("cut_in") is None else float(t["cut_in"]),
            np.inf if t.get("cut_out") is None else float(t["cut_out"]),
        )
        conf["V"], conf["POW"] = V, POW
    return conf


def get_solarpanelconfig(panel):
    res_name = "resources/solarpanel/" + panel + ".yaml"
    with open(SRC_ROOT / res_name, "r") as resource_file:
        panelconf = yaml.safe_load(resource_file)
    return panelconf


def solarpanel_rated_capacity_per_unit(panel):
    # unit is m^2 here

    if isinstance(panel, str):
        panel = get_solarpanelconfig(panel)

    model = panel.get("model", "huld")
    if model == "huld":
        return panel["efficiency"]
    elif model == "bofinger":
        # one unit in the capacity layout is interpreted as one panel of a
        # capacity (A + 1000 * B + log(1000) * C) * 1000W/m^2 * (k / 1000)
        A, B, C = itemgetter("A", "B", "C")(panel)
        return (A + B * 1000.0 + C * np.log(1000.0)) * 1e3


def windturbine_rated_capacity_per_unit(turbine):
    if isinstance(turbine, str):
        turbine = get_windturbineconfig(turbine)

    return turbine["P"]


def windturbine_smooth(turbine, params):
    """
    Smooth the powercurve in `turbine` with a gaussian kernel

    Parameters
    ----------
    turbine : dict
            Turbine config with at least V and POW
    params : dict
            Allows adjusting fleet availability eta, mean Delta_v and
            stdev sigma. Defaults to values from Andresen's paper: 0.95,
            1.27 and 2.29, respectively.

    Returns
    -------
    turbine : dict
            Turbine config with a smoothed power curve

    References
    ----------
    G. B. Andresen, A. A. Søndergaard, M. Greiner, Validation of
    Danish wind time series from a new global renewable energy atlas
    for energy system analysis, Energy 93, Part 1 (2015) 1074–1088.
    """

    if not isinstance(params, dict):
        params = {}

    eta = params.setdefault("eta", 0.95)
    Delta_v = params.setdefault("Delta_v", 1.27)
    sigma = params.setdefault("sigma", 2.29)

    def kernel(v_0):
        # all velocities in m/s
        return (
            1.0
            / np.sqrt(2 * np.pi * sigma * sigma)
            * np.exp(-(v_0 - Delta_v) * (v_0 - Delta_v) / (2 * sigma * sigma))
        )

    def smooth(velocities, power):
        # interpolate kernel and power curve to the same, regular velocity grid
        velocities_reg = np.linspace(-50.0, 50.0, 1001)
        power_reg = np.interp(velocities_reg, velocities, power)
        kernel_reg = kernel(velocities_reg)

        # convolve power and kernel
        # the downscaling is necessary because scipy expects the velocity
        # increments to be 1., but here, they are 0.1
        convolution = 0.1 * fftconvolve(power_reg, kernel_reg, mode="same")

        # sample down so power curve doesn't get too long
        velocities_new = np.linspace(0.0, 35.0, 72)
        power_new = eta * np.interp(velocities_new, velocities_reg, convolution)

        return velocities_new, power_new

    turbine = turbine.copy()
    turbine["V"], turbine["POW"] = smooth(turbine["V"], turbine["POW"])
    turbine["P"] = np.max(turbine["POW"])

    if any(turbine["POW"][np.where(turbine["V"] == 0.0)] > 1e-2):
        logger.warning(
            "Oversmoothing detected with parameters %s. Turbine generates energy at 0 m/s wind speeds",
            params,
        )

    return turbine


def get_available_windturbines():
    """Get a list of available wind turbine configurations."""
    res_path = SRC_ROOT / "resources/windturbine"
    return [p.stem for p in res_path.glob("*.yaml") if p.is_file()]


def get_available_solarpanels():
    """Get a list of available solar panel configurations."""
    res_path = SRC_ROOT / "resources/solarpanel"
    return [p.stem for p in res_path.glob("*.yaml") if p.is_file()]
