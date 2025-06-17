# DNA Loop Extrusion by SMCCs in LAMMPS

## Installation

### Python

#### From PyPI

The code is available as a [package on PyPI](https://pypi.org/project/smc-lammps/).
```sh
pip install smc-lammps
```

#### From Source Using [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended)
```sh
git clone https://github.com/LucasDooms/SMC_LAMMPS.git
cd SMC_LAMMPS
uv sync
source .venv/bin/activate
```
or use `uv run <command>` without activating the environment.

#### From Source Using pip
```sh
git clone https://github.com/LucasDooms/SMC_LAMMPS.git
cd SMC_LAMMPS
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
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

To use the `src/smc_lammps/post-process/visualize.py` script, you will need VMD,  
see https://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=VMD.

## Docker Installation

You can also use docker to run the code. First build the image
```sh
docker build -t smc-lammps .
```
Now you can run an interactive session using
```sh
docker run -it -v .:/data smc-lammps
```
Or, to run directly (see Usage)
```sh
docker run -v .:/data smc-lammps smc-lammps ...
```

Note: the docker image does not include VMD.

## Usage

1. Create a directory for your simulation using `smc-lammps mysim`.
2. Define all parameters in `mysim/parameters.py` (see `src/smc_lammps/generate/default_parameters.py` for all options).
3. Run `smc-lammps [flags] mysim`, providing the directory of the parameters file. Use the `-g` flag to generate the required parameterfile and datafile.

#### Examples
- `smc-lammps mysim -gr`   to generate and run.
- `smc-lammps mysim -grpv` to generate, run, post-process, and visualize.
- `smc-lammps mysim -grvn` to generate, run, and visualize while ignoring errors.
- `smc-lammps mysim -v`    to visualize.
- `smc-lammps mysim -vf`   to visualize a perspective following the SMC.
- `smc-lammps mysim -c`    to continue a run from a restart file.

#### Help
Show help with `smc-lammps --help`.

#### Shell Completion
To get shell completion when using `smc-lammps` on the command-line run the following:
 - For bash or zsh, use `eval "$(register-python-argcomplete smc-lammps)"`
 - For fish, use `register-python-argcomplete --shell fish smc-lammps | source`


## Authors

Original code by Stefanos Nomidis (https://github.com/sknomidis/SMC_LAMMPS).  
Modifications by Arwin Goossens.  
All commits in this repository by Lucas Dooms.  
Released under [MIT license](LICENSE)
