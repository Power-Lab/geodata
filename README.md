GEODATA
-

[![DOI](https://zenodo.org/badge/218690319.svg)](https://zenodo.org/badge/latestdoi/218690319)

**Geodata** is a Python library of geospatial data collection and "pre-analysis" tools. Geospatial and gridded datasets of physical variables are ubiquitous and increasingly high resolution. Long time-series gridded datasets can be generated as part of earth system models, and due to their geographic coverage they can have wider applications, including in engineering and social sciences. Geospatial (GIS) files can encode various physical, social, economic, and political data. However, working with these datasets often has significant startup costs due to their diverse sources, data formats, resolutions, and large file sizes.

Geodata streamlines the collection and use of geospatial datasets through the creation of shared scripts for “analysis-ready” physical variables. Its purpose is to make it easier for researchers to identify, download, and work with new sources of geospatial data. Additionally, with a minimal amount of data consistency checks and metadata information, when one researcher goes through this exercise, everyone benefits.

Geodata builds off the **[atlite](https://github.com/PyPSA/atlite)** library, which converts weather data (such as wind speeds, solar radiation, temperature and runoff) into power systems data (such as wind power, solar power, hydro power and heating demand time series). Geodata retains the power systems data functionality of atlite.

![png](docs/source/_static/images/geodata_workflow_chart.png)

## Repository history

This repository is part of a fork chain:

1. **[GeodataTools/geodata](https://github.com/GeodataTools/geodata)** — the original upstream project.
2. **[KULcoder/geodata](https://github.com/KULcoder/geodata)**, branch [`data-pvlib-integration`](https://github.com/KULcoder/geodata/tree/data-pvlib-integration) — the immediate parent of this repository. That branch is 157 commits ahead of `GeodataTools/geodata`'s `master`, and adds the `pvlib`-based solar `ModelChain` integration and mask/province-split tooling this repository builds on.
3. **This repository ([Power-Lab/geodata](https://github.com/Power-Lab/geodata))** — cloned from `KULcoder/geodata@data-pvlib-integration` and maintained independently for the Power Lab's own production use (not intended to be merged back upstream). Updates made in this version:
   - **Fixed a wind capacity-factor bug** where `wind_cf` returned `-inf` for wind speeds outside a turbine's tabulated power-curve range, by changing the curve interpolation's extrapolation behavior to clamp to `0` (matching `windpowerlib`'s convention) instead of extrapolating.
   - **Fixed a southern-hemisphere solar orientation bug** where panel tilt/azimuth were hardcoded to northern-hemisphere-optimal values (`azimuth=180`, `tilt=35`), which is physically backwards below the equator. Added `latitude_optimal_orientation()`, a hemisphere-aware helper that derives tilt/azimuth from latitude.
   - **Pinned the lint toolchain** (Ruff version and rule set) and switched CI to the actively maintained `astral-sh/ruff-action`, after the previous action was archived and a Ruff default-rule-set expansion caused CI to start failing on pre-existing code.
   - **Added regression and smoke tests**: a comparative test proving the hemisphere-orientation fix actually changes the ModelChain's output in the physically correct direction (not just that it runs without crashing), plus cross-country smoke tests exercising the turbine power-curve and pvlib `ModelChain` code paths for China, USA, Vietnam, and Indonesia.
   - **Corrected stale documentation**: package setup instructions and `environment.yaml` now describe this branch's actual install/dependencies (which include `pvlib`, `timezonefinder`, `geopandas`, and other packages absent from upstream master's docs), and previously-undocumented behavior (HRRR/MERRA2 experimental status, the `model.results` caching layer) is now documented.

## What this repository adds

Building on top of `GeodataTools/geodata`'s core weather-data-to-power-systems-data pipeline, this repository's key functional addition is a **`pvlib`-based solar `ModelChain` model** (`geodata.model.pvlib`) alongside the existing wind model, plus **mask/province-splitting tooling** (`geodata.mask`) for cropping and splitting national-scale output into sub-national (e.g. provincial/state) regions using elevation, slope, and protected-area (WDPA) layers.

Used together with the companion **[`geodata_helpers`](https://github.com/Power-Lab/geodata_helpers)** repository — which holds the per-country configuration (turbine/module/inverter selection, national crop bounds, technology configs) and the driver scripts (`profile_generation`, `profile_split`) that call into this library — this repository enables generating **hourly, province/state-level wind and solar capacity-factor profiles** from ERA5 weather data for a configurable set of target countries, spanning multiple years, ready for use in downstream power systems modeling.

## Installation

**Geodata** has been tested to run with python3 (>= 3.10). Read the [package setup instructions](docs/source/quick_start/packagesetup.md) to configure and install the package.

This branch's dependencies extend those of upstream `geodata` with the `pvlib` solar model and related geospatial packages. Installation will install:
* `numpy`, `scipy`, `pandas`, `bottleneck`, `numexpr`
* `xarray`, `h5netcdf`, `dask`, `boto3`, `toolz`, `pyproj`, `requests`
* `matplotlib`, `rasterio`, `rioxarray`, `shapely`, `geopandas`, `tqdm`
* `pvlib`, `timezonefinder`, `cdsapi`

See `pyproject.toml` and `environment.yaml` for exact version pins.

## Usage

1. Install the package per the [package setup instructions](docs/source/quick_start/packagesetup.md).
2. Configure API credentials for the weather data source (ERA5 via CDS API; see [Example Notebooks](example_notebooks) for setup).
3. Build a `Cutout` for the region/time range of interest, then use `WindModel`/`Pvlib` (see `docs/source/modeling`) to estimate wind or solar capacity factors.
4. For production, multi-country batch generation, use the `profile_generation` and `profile_split` scripts in the companion [`geodata_helpers`](https://github.com/Power-Lab/geodata_helpers) repository, which wrap this library's models with per-country configuration.

See the [Example Notebooks](example_notebooks) for worked examples, and `docs/source/` for full API documentation.

## Testing

Run the test suite with `pytest`:

```bash
pytest tests/
```

`tests/pr/` includes cross-country smoke tests (turbine power curves and the pvlib `ModelChain`, across China/USA/Vietnam/Indonesia) and regression tests for the wind extrapolation and hemisphere-orientation fixes described above. CI runs the test suite and Ruff lint checks on every push and pull request (see `.github/workflows/`).

## Documentation

YOU CAN FIND GEODATA's DOCUMENTATIONS HERE: ...
You may also jump directly to [Example Notebooks](example_notebooks).

## Contributing

We welcome suggestions for feature enhancements and the identification of bugs. Please make an issue or contact the [authors](https://mdavidson.org/about/) of geodata.

## License

Geodata is licensed under the GNU GENERAL PUBLIC LICENSE Version 3 (2007). This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License](/LICENSE.txt) for more details.

## Support

The Geodata team would like to thank the Center for Global Transformation at UC San Diego for providing financial support to the project.
