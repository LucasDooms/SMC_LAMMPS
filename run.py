import argparse
from functools import partial
from pathlib import Path
import subprocess


parser = argparse.ArgumentParser(
    prog='Run LAMMPS',
    description='runs the LAMMPS file and optionally runs python setup scripts',
    epilog='End'
)


parser.add_argument('directory', help='the directory containing parameters for LAMMPS')
parser.add_argument('-g', '--generate', action='store_true', help='run the python setup scripts before executing LAMMPS')
parser.add_argument('-e', '--executable', help='name of the LAMMPS executable to use', default="lmp")
parser.add_argument('-i', '-in', '--input', help='path to input file to give to LAMMPS', default="input")
parser.add_argument('-o', '--output', help='path to dump LAMMPS output to (prints to terminal by default)')

args = parser.parse_args()
path = Path(args.directory)

if args.generate:
    print("running setup file...")
    completion = subprocess.run(f"python generate+parse.py {path}".split(" "))
    if completion.returncode != 0:
        raise Exception(f"process ended with error code {completion.returncode}\n{completion}")
    print("succesfully ran setup file")


run_with_output = partial(subprocess.run, [f"{args.executable}", "-in", f"{Path(args.input).absolute()}"], cwd=path.absolute())

if args.output:
    with open(args.output, 'w') as output_file:
        print(f"running LAMMPS file {args.input}, output redirected to {args.output}")
        run_with_output(stdout=output_file)
else:
    print(f"running LAMMPS file {args.input}, printing output to terminal")
    run_with_output()
