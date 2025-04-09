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
parser.add_argument('-s', '--seed', help='set the seed to be used by LAMMPS, this takes precedence over the seed in default_parameters.py and parameters.py')
parser.add_argument('-p', '--post-process', action='store_true', help='run the post-processing scripts after running LAMMPS')
parser.add_argument('-n', '--ignore-errors', action='store_true', help='keep running even if the previous script exited with a non-zero error code')
parser.add_argument('-v', '--visualize', action='store_true', help='open VMD after all scripts have finished')
parser.add_argument('-f', '--force', action='store_true', help='don\'t prompt before overwriting existing files')
parser.add_argument('-e', '--executable', help='name of the LAMMPS executable to use', default="lmp")
parser.add_argument('-i', '-in', '--input', help='path to input file to give to LAMMPS', default="input.lmp")
parser.add_argument('-o', '--output', help='path to dump LAMMPS output to (prints to terminal by default)')

args = parser.parse_args()
path = Path(args.directory)


def run_and_handle_error(process, ignore_errors: bool):
    completion = process()
    if completion.returncode != 0:
        message = f"process ended with error code {completion.returncode}\n{completion}\n"
        print(message)
        if ignore_errors:
            print("-n (--ignore-errors) flag is set, continuing...\n")
            return
        raise Exception(message)


if args.generate:
    extra_args = []
    if args.seed:
        extra_args.append(args.seed)
    print("running setup file...")
    run_and_handle_error(lambda: subprocess.run(["python", "generate/generate+parse.py", f"{path}"] + extra_args, check=False), args.ignore_errors)
    print("succesfully ran setup file")


# check if output.lammpstrj exists
file_exists = (path / "output.lammpstrj").exists()

if args.force or not file_exists:
    run_with_output = partial(subprocess.run, [f"{args.executable}", "-in", f"{Path(args.input).absolute()}"], cwd=path.absolute())

    if args.output:
        with open(args.output, 'w', encoding="utf-8") as output_file:
            print(f"running LAMMPS file {args.input}, output redirected to {args.output}")
            run_and_handle_error(lambda: run_with_output(stdout=output_file), args.ignore_errors)
    else:
        print(f"running LAMMPS file {args.input}, printing output to terminal")
        run_and_handle_error(run_with_output, args.ignore_errors)
else:
    print("cannot run lammps script, output.lammpstrj already exists (use -f to overwrite files)")
    print("moving on...")

if args.post_process:
    print("running post processing...")
    run_and_handle_error(lambda: subprocess.run(["python", "post-process/process_displacement.py", f"{path}"], check=False), args.ignore_errors)
    print("succesfully ran post processing")

if args.visualize:
    print("starting VMD")
    run_and_handle_error(lambda: subprocess.run(["python", "post-process/visualize.py", f"{path}"], check=False), args.ignore_errors)
    print("VMD exited")

print("end of run.py")
