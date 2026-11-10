"""Regenerate geodata's table of generic wind power curves by specific power.

The curves follow PLUSWIND (Millstein et al. 2023), which builds generic curves
with NREL's System Advisor Model (SAM): maximum power coefficient 0.45, maximum
tip-speed ratio 8, maximum tip speed 80 m/s, SAM's default drivetrain, sea-level
air density (1.225 kg/m3). PLUSWIND's own curves are not published, so they are
regenerated here with PySAM.

With these settings the curve, normalised by rated power, depends only on the
specific power (rated power / swept area); rated power and rotor size cancel out.
The table stores the full curve (cut-in 0, cut-out 40 m/s) at every integer
specific power from 150 to 600 W/m2. geodata applies each turbine's own cut-in and
cut-out at run time (power is zero at or below cut-in and at or above cut-out),
which reproduces PySAM's own cut-in/cut-out handling exactly.

Requires PySAM (``pip install nrel-pysam``); geodata itself only reads the table.

Run:  python scripts/make_generic_power_curves.py
Output: src/geodata/resources/windturbine/generic/sam_generic_power_curves.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import PySAM.Windpower as wp

OUT = (
    Path(__file__).resolve().parents[1]
    / "src/geodata/resources/windturbine/generic/sam_generic_power_curves.csv"
)
SP_RANGE = range(150, 601)  # W/m2
ROTOR_D = 100  # m; any value gives the same normalised curve


def full_curve(sp_wm2: float) -> tuple[np.ndarray, np.ndarray]:
    rated_kw = sp_wm2 * np.pi * (ROTOR_D / 2) ** 2 / 1000.0
    m = wp.default("WindPowerSingleOwner")
    m.Turbine.calculate_powercurve(
        turbine_size=rated_kw,
        rotor_diameter=ROTOR_D,
        elevation=0,
        max_cp=0.45,
        max_tip_speed=80,
        max_tip_sp_ratio=8,
        cut_in=0.0,
        cut_out=40.0,
        drive_train=0,
    )
    v = np.array(m.Turbine.wind_turbine_powercurve_windspeeds, dtype=float)
    p = np.array(m.Turbine.wind_turbine_powercurve_powerout, dtype=float) / rated_kw
    return v, p


def main():
    cols = {}
    speeds = None
    for sp in SP_RANGE:
        v, p = full_curve(sp)
        if speeds is None:
            speeds = v
        elif not np.array_equal(v, speeds):
            raise RuntimeError("PySAM returned a different wind-speed grid")
        cols[str(sp)] = np.clip(p, 0.0, 1.0)
    table = pd.DataFrame(cols)
    table.insert(0, "wind_speed_ms", speeds)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False, float_format="%.5f")
    print(f"wrote {OUT} ({len(speeds)} speeds x {len(cols)} specific powers)")


if __name__ == "__main__":
    main()
