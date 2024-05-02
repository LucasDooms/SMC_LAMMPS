# DNA Loop Extrusion by SMCCs in LAMMPS

## Authors

Original code by Stefanos Nomidis (https://github.com/sknomidis/SMC_LAMMPS), released under [MIT license](ORIGINAL_LICENSE).  
Modifications by Arwin Goossens.  
All commits in this repository by Lucas Dooms, released under [GPL v3 license](LICENSE).

## How to run

1. Define all parameters in "parameters"
2. Run `python run.py ...`, provide the directory of the parameters file, use the `-g` flag to generate the required parameterfile and datafile.

examples:
- `python run.py 7nm_kleisin -g`   to generate and run
- `python run.py 7nm_kleisin -gpv` to generate, run, post-process, and visualize
- `python run.py 7nm_kleisin -gvn` to generate, run, and visualize while ignoring errors

help:  
`python run.py --help`
