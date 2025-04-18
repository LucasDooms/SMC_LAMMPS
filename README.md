# DNA Loop Extrusion by SMCCs in LAMMPS

## Installation

### Python

#### Using [uv](https://docs.astral.sh/uv/getting-started/installation/)
```sh
git clone https://github.com/LucasDooms/SMC_LAMMPS.git
cd SMC_LAMMPS
uv sync
source .venv/bin/activate
```
or use `uv run <command>` without activating the environment.

#### Using pip
```sh
git clone https://github.com/LucasDooms/SMC_LAMMPS.git
cd SMC_LAMMPS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### LAMMPS

You will need a LAMMPS executable with the `MOLECULE` and `RIGID` packages.  
See https://docs.lammps.org/Install.html for more information.

Simple example:
```sh
git clone https://github.com/lammps/lammps --depth=1000 mylammps
cd mylammps
git checkout stable # or release for a more recent version
mkdir build && cd build
cmake -D CMAKE_INSTALL_PREFIX="$HOME/lammps" -D PKG_MOLECULE=yes -D PKG_RIGID=yes ../cmake
cmake --build . -j8
make
make install
export PATH="$HOME/lammps/bin:$PATH"
```

### (Optional) VMD

To use the `post-process/visualize.py` script, you will need VMD, see  
https://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=VMD.

## Usage

1. Create a directory for your simulation, e.g. `hinge`
2. Define all parameters in `hinge/parameters.py` (see `generate/default_parameters.py` for all options)
3. Run `python run.py [flags] hinge`, providing the directory of the parameters file. Use the `-g` flag to generate the required parameterfile and datafile.

examples:
- `python run.py hinge -g`   to generate and run
- `python run.py hinge -gpv` to generate, run, post-process, and visualize
- `python run.py hinge -gvn` to generate, run, and visualize while ignoring errors
- `python run.py hinge -c`   to continue a run from a restart file

help:  
`python run.py --help`


## Authors

Original code by Stefanos Nomidis (https://github.com/sknomidis/SMC_LAMMPS).  
Modifications by Arwin Goossens.  
All commits in this repository by Lucas Dooms.  
Released under [MIT license](LICENSE)
