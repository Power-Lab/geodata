# Installation

This guide covers how to install and configure the **geodata** package for local and cloud use.

```{important}
This documentation describes **this repository's `main` branch**
([Power-Lab/geodata](https://github.com/Power-Lab/geodata)). It carries
KULcoder's `data-pvlib-integration` work — the `pvlib`-based solar model
(`geodata.model.pvlib`), the `geodata.mask` module, and the restructured
`datasets/era5`, `datasets/merra2`, `datasets/hrrr` packages, none of which are
in upstream [GeodataTools/geodata](https://github.com/GeodataTools/geodata)
`master` — **plus this repository's own fixes** (see the README), which the
parent [KULcoder/geodata](https://github.com/KULcoder/geodata/tree/data-pvlib-integration)
branch does not have. **Installing from `GeodataTools/geodata` (master) or from
`KULcoder/geodata@data-pvlib-integration` instead of this repository will
silently give you a package missing some or all of these.**
```

Make sure that you have the following **required** software set up:

* [Python 3.10+](https://www.python.org/downloads/) (this branch's `pyproject.toml` requires `python>=3.10`)

* Package Management System
  - [conda](https://docs.conda.io/projects/conda/en/latest/) (miniconda or anaconda)
  - [pip](https://pip.pypa.io/en/stable/installation/)

## Downloading Geodata

### Option A: install directly with pip (no local clone)

If you don't need to edit geodata's source, install this repository straight from GitHub
into your active environment:

```bash
pip install --force-reinstall "geodata @ git+https://github.com/Power-Lab/geodata.git@main"
```

This is the fastest path if you just want to *use* the package (e.g. from
`geodata_helpers` scripts). It matches the install pattern in the lab's own
[RE Profiles End-to-End
Workflow](https://docs.google.com/document/u/3/d/1rWgo6mjNRm7zkycCQthHBqQRv1f_NnsRbWBefU3MCSM/edit)
for generating RE profiles on TSCC — note that if that document still installs from
`KULcoder/geodata@data-pvlib-integration`, it installs this repository's *parent*,
without the fixes described in the README (`latitude_optimal_orientation` does not
exist there, so `geodata_helpers`' solar driver fails on import).

### Option B: clone the repo (for local development)

To download **geodata** for local editing, open a terminal/shell window, navigate to your
preferred working directory, and run the following. (If you do not have Git installed, you
may also directly download the repository as a
[zip archive](https://github.com/Power-Lab/geodata/archive/refs/heads/main.zip).)

```bash
git clone https://github.com/Power-Lab/geodata.git
cd geodata
```

## Configuring File Storage Location

To configure where to store downloaded and processed files, define an environment variable called `GEODATA_ROOT` and save in your shell configuration files, such as `.bashrc` or `.zshrc`:

```bash
export GEODATA_ROOT=<YOUR_PATH_HERE>
```

```{note}
If you are using a Windows machine, you can set the environment variable by running the following command in the command prompt:

```bash
setx GEODATA_ROOT <YOUR PATH HERE>
```

If you are running geodata in a Jupyter Notebook, you could define the variable by adding and running the following cell:
```
%setenv GEODATA_ROOT <YOUR PATH HERE>
```

If you do not define this variable, all datasets and cutouts will be stored under `~/.local/geodata` by default.

## Building Geodata

### Anaconda/miniconda Environment

[Anaconda](https://www.anaconda.com/download)/[miniconda](https://docs.conda.io/en/latest/miniconda.html) is a powerful package manager and environment manager for Windows, macOS or Linux, and it provides easy installation for all operating systems. It is especially convenient if you are building Geodata on the cloud with potential installation permission issues.

If you already have Anaconda/miniconda installed on your machine, jump straight to the `conda env create` step. Otherwise, you have 2 [options](https://conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda): download Anaconda or miniconda. Installing Anaconda requires >3GB disk space and takes minutes to download, so we will choose **miniconda** instead because is a small, bootstrap version of Anaconda that includes only conda, Python, the packages they depend on, and a small number of other useful packages.

From the package's root directory (ie, "geodata", from Option B above), create the environment from
`environment.yaml` — this installs the GIS-heavy dependencies (`rasterio`, `geopandas`, `pyproj`,
`libgdal`, ...) via conda-forge, which is more reliable than pip for those packages:

```bash
conda env create -f environment.yaml -n <ENVIRONMENT_NAME>
conda activate <ENVIRONMENT_NAME>
```

To use **geodata** in Python scripts by calling `import geodata`, you still need to install the
package itself on top of that environment (`environment.yaml` only installs its dependencies).
In the terminal/shell window, navigate to the package's root directory (ie, "geodata"), and run
the following (use `pip install -e .` instead of `pip install .` if you're actively editing the
source and want changes to take effect without reinstalling):

```bash
pip install .
```

**Note**: If running `pip install .` generates errors related to being unable to install the **rasterio** package due to conflicts with incompatible packages, you may need to reinstall Anaconda/miniconda depending on what you went with during setup. Then run the following commands:

```bash
conda update --all
```

```bash
conda install rasterio
```

### Installation with pip

#### macOS/Linux
In macOS or Linux's terminal, navigate to the package's root directory, and run the following:

```bash
pip install .
```

**Note**: All dependencies should install automatically upon building the package, with possible exceptions such as the **rasterio** library, which requires Cython and GDAL. For the source of these instructions and more documentation about **rasterio**, see [the rasterio documentation](https://rasterio.readthedocs.io/en/latest/installation.html).

If one of the dependency, such as **rasterio** does not install automatically (we know this through the error message from the command above), we will have to install it seperately in the terminal:

```bash
pip install rasterio
```

Once the library above is successfully installed, re-run the installation command above to build Geodata:

```bash
pip install .
```

If there is an error message regarding one of Geodata's dependency, repeat the process and use `pip install` to seperately download it.

#### Windows

In the Windows command prompt, navigate to the package's root directory, and run the following:

```bash
pip install .
```

**Note**: All dependencies should install automatically upon building the package, with possible exceptions such as the **rasterio** library, which requires other dependencies.

If one of the dependency, such as **GDAL** does not install automatically (we know this through the error message from the command above), we will have to install it seperately in the terminal. There are 2 options to solve this issue. Once we download the required dependency successfully, we can proceed by re-run the install command:

```bash
pip install .
```
##### Pipwin

This option is recommended. When we download libraries that is built on **GDAL**, we might run into this [issue](https://stackoverflow.com/q/54734667), where a GDAL API version must be specified.

**Pipwin** is a complementary tool for **pip** on Windows. **pipwin** installs unofficial python package binaries for windows provided by Christoph Gohlke [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/).

Run the following commands to download **pipwin** and acquire the dependencies: (This solution is adopted from [stackoverflow](https://stackoverflow.com/a/58943939))

```
pip install pipwin
pipwin install shapely
pipwin install gdal
pipwin install fiona
pipwin install pyproj
pipwin install six
pipwin install rtree
```

You may need to also install the **wheel** package `pip install wheel` to facilitate building the wheels.

Similarly, if you run into installation errors regarding the **rasterio** or **bottleneck** packages, you can also call **pipwin install rasterio** or **pipwin install bottleneck** to download them.

##### Direct Wheel Install

To install **rasterio** and the necessary GDAL library, we can download the appropriate binaries for your system by hand ([rasterio](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio) and [GDAL](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)) , place them into the current working directory, and run the following command in the downloads folder:

```bash
pip install -U pip
pip install {GDAL binary name here}.whl
pip install {rasterio binary name here}.whl
```

You may also need to also install the **wheel** package `pip install wheel` to facilitate building the wheels.

For more documentation/troubleshooting conda installation issues, see [here](https://github.com/conda-forge/rasterio-feedstock)
