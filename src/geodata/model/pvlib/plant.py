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

"""A PV plant configuration that is turned into a pvlib system cell by cell.

A configuration is one module, one inverter, a DC/AC ratio and a mount (fixed tilt
or single-axis tracking). Fixed mounts can use each cell's latitude-optimal tilt,
facing the equator, so one configuration serves a whole country, including both
hemispheres (Plan 4.3).

The DC/AC ratio is set by scaling one inverter by a factor k (Paco, Pdco, Pso and
Pnt multiplied by k, C0 divided by k). This scales the Sandia inverter model's AC
output exactly by k, so CF does not depend on k; k only lets an integer number of
strings reach the target ratio closely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pvlib import pvsystem, temperature

from .calculations import latitude_optimal_orientation

INVERTER_SCALED_KEYS = ("Paco", "Pdco", "Pso", "Pnt")


def scale_inverter(inverter: pd.Series | dict, k: float) -> pd.Series:
    """Sandia inverter parameters for k identical inverters in parallel."""
    s = pd.Series(inverter).copy()
    for key in INVERTER_SCALED_KEYS:
        s[key] = float(s[key]) * k
    s["C0"] = float(s["C0"]) / k
    return s


def default_modules_per_string(module, inverter) -> int:
    """Largest string whose Voc (+10% cold margin) stays below the inverter's Vdcmax
    and whose Vmp stays inside the MPPT window (falls back to Vdcmax alone)."""
    vmp, voc = float(module["V_mp_ref"]), float(module["V_oc_ref"])
    vdcmax = float(inverter["Vdcmax"])
    lo = float(inverter.get("Mppt_low", 0.0) or 0.0)
    hi = float(inverter.get("Mppt_high", 0.0) or 0.0) or vdcmax
    best = None
    for n in range(1, 200):
        if n * voc * 1.1 > vdcmax or n * vmp > hi:
            break
        if n * vmp >= lo:
            best = n
    if best is None:
        best = max(1, int(vdcmax // (voc * 1.1)))
    return best


@dataclass
class PlantSpec:
    """One PV configuration. See :meth:`geodata.model.pvlib.Pvlib.init_plant`."""

    module_parameters: pd.Series
    inverter_parameters: pd.Series
    dc_ac_ratio: float = 1.25
    mount: str = "fixed"  # "fixed" or "single_axis"
    tilt: float | str = "latitude_optimal"
    azimuth: float | str = "equator"
    axis_tilt: float = 0.0
    axis_azimuth: float = 180.0
    gcr: float = 0.4
    max_angle: float = 45.0
    backtrack: bool = True
    modules_per_string: int | None = None
    module_type: str = "glass_polymer"
    racking_model: str = "open_rack"
    temperature_model_parameters: dict | None = None
    inverter_scale: float = 10.0
    max_ratio_error: float = 0.01
    # filled in by __post_init__
    strings: int = field(init=False, default=0)
    inverter: pd.Series = field(init=False, default=None)
    achieved_dc_ac: float = field(init=False, default=float("nan"))

    def __post_init__(self):
        if self.mount not in ("fixed", "single_axis"):
            raise ValueError(f"mount must be 'fixed' or 'single_axis', got {self.mount!r}")
        self.module_parameters = pd.Series(self.module_parameters)
        self.inverter_parameters = pd.Series(self.inverter_parameters)
        if self.modules_per_string is None:
            self.modules_per_string = default_modules_per_string(
                self.module_parameters, self.inverter_parameters
            )
        if self.temperature_model_parameters is None:
            glass = "glass_glass" if self.module_type == "glass_glass" else "glass_polymer"
            self.temperature_model_parameters = temperature.TEMPERATURE_MODEL_PARAMETERS[
                "sapm"
            ][f"{self.racking_model}_{glass}"]
        self._size()

    def _size(self):
        """Pick the inverter scale and string count that hit the DC/AC target."""
        stc = float(self.module_parameters["STC"])
        paco = float(self.inverter_parameters["Paco"])
        per = int(self.modules_per_string)
        k = float(self.inverter_scale)
        for _ in range(60):
            n = max(1, int(round(self.dc_ac_ratio * paco * k / (per * stc))))
            ratio = n * per * stc / (paco * k)
            if abs(ratio / self.dc_ac_ratio - 1) <= self.max_ratio_error:
                break
            k *= 2
        self.strings = n
        self.inverter = scale_inverter(self.inverter_parameters, k)
        self.achieved_dc_ac = ratio

    @property
    def dc_nameplate_w(self) -> float:
        """Module STC rating times the number of modules (W)."""
        return float(self.module_parameters["STC"]) * self.modules_per_string * self.strings

    @property
    def ac_nameplate_w(self) -> float:
        """Inverter rated AC power (W)."""
        return float(self.inverter["Paco"])

    def orientation(self, lat: float) -> tuple[float, float]:
        """(tilt, azimuth) of a fixed mount at latitude ``lat``."""
        opt = latitude_optimal_orientation(lat)
        tilt = opt["surface_tilt"] if self.tilt == "latitude_optimal" else float(self.tilt)
        az = opt["surface_azimuth"] if self.azimuth == "equator" else float(self.azimuth)
        return tilt, az

    def mount_for(self, lat: float):
        if self.mount == "single_axis":
            return pvsystem.SingleAxisTrackerMount(
                axis_tilt=self.axis_tilt,
                axis_azimuth=self.axis_azimuth,
                max_angle=self.max_angle,
                backtrack=self.backtrack,
                gcr=self.gcr,
            )
        tilt, az = self.orientation(lat)
        return pvsystem.FixedMount(surface_tilt=tilt, surface_azimuth=az)

    def system_for(self, lat: float) -> pvsystem.PVSystem:
        """The pvlib system for a cell at latitude ``lat``."""
        array = pvsystem.Array(
            mount=self.mount_for(lat),
            module_parameters=self.module_parameters,
            temperature_model_parameters=self.temperature_model_parameters,
            modules_per_string=self.modules_per_string,
            strings=self.strings,
            module_type=self.module_type,
        )
        return pvsystem.PVSystem(
            arrays=[array],
            inverter_parameters=self.inverter,
            racking_model=self.racking_model,
        )

    def describe(self) -> dict:
        """Attributes recorded in output files."""
        d = {
            "mount": self.mount,
            "dc_ac_ratio_target": float(self.dc_ac_ratio),
            "dc_ac_ratio": float(np.round(self.achieved_dc_ac, 4)),
            "module": str(self.module_parameters.get("name", self.module_parameters.name)),
            "inverter": str(self.inverter_parameters.name),
            "module_type": self.module_type,
            "modules_per_string": int(self.modules_per_string),
        }
        if self.mount == "fixed":
            d["tilt"] = str(self.tilt)
            d["azimuth"] = str(self.azimuth)
        else:
            d.update(
                axis_tilt=self.axis_tilt,
                axis_azimuth=self.axis_azimuth,
                gcr=self.gcr,
                max_angle=self.max_angle,
                backtrack=int(self.backtrack),
            )
        return d
