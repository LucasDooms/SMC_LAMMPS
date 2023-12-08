import argparse
from pathlib import Path
import subprocess


parser = argparse.ArgumentParser(
    prog='Run LAMMPS',
    description='runs the LAMMPS file and optionally runs python setup scripts',
    epilog='End of help'
)


parser.add_argument('directory', help='the directory containing parameters for LAMMPS')
parser.add_argument('-g', '--generate', action='store_true', help='run the python setup scripts before executing LAMMPS')
parser.add_argument('-c', '--custom', help='run a custom python script as a setup before executing LAMMPS (runs before default -g script)')
parser.add_argument('-e', '--executable', help='name of the LAMMPS executable to use', default="lmp")
parser.add_argument('-i', '-in', '--input', help='path to input file to give to LAMMPS', default="input")

args = parser.parse_args()
path = Path(args.directory)

if args.custom:
    subprocess.run(f"python {args.custom}")

if args.generate:
    subprocess.run(f"python generate+parse.py {path}".split(" "))

subprocess.run(f"{args.executable} -in {Path(args.input).absolute()}", cwd=path.absolute())
