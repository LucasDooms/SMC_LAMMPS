import argparse
import subprocess
from functools import partial
from pathlib import Path
from warnings import warn


def parse() -> argparse.Namespace:
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
    parser.add_argument('-f', '--force', action='store_true', help='don\'t prompt before overwriting existing files / continuing empty simulation')
    parser.add_argument('-c', '--continue', dest='continue_flag', action='store_true', help='continue from restart file and append to existing simulation')
    parser.add_argument('-e', '--executable', help='name of the LAMMPS executable to use', default="lmp")
    parser.add_argument('-i', '-in', '--input', help='path to input file to give to LAMMPS', default="input.lmp")
    parser.add_argument('-o', '--output', help='path to dump LAMMPS output to (prints to terminal by default)')

    return parser.parse_args()


def run_and_handle_error(process, ignore_errors: bool):
    completion = process()
    if completion.returncode != 0:
        message = f"process ended with error code {completion.returncode}\n{completion}\n"
        print(message)
        if ignore_errors:
            print("-n (--ignore-errors) flag is set, continuing...\n")
            return
        raise ChildProcessError(message)


def generate(args, path):
    if not args.generate:
        return

    extra_args = []
    if args.seed:
        extra_args.append(args.seed)
    print("running setup file...")
    run_and_handle_error(lambda: subprocess.run(python_run + ["generate.generate+parse", f"{path}"] + extra_args, check=False), args.ignore_errors)
    print("successfully ran setup file")


def get_lammps_args_list(lammps_vars):
    out = []
    for var in lammps_vars:
        out += ["-var"] + var
    return out


def perform_run(args, path, lammps_vars=()):
    run_with_output = partial(subprocess.run, [f"{args.executable}", "-sf", "opt", "-in", f"{Path(args.input).absolute()}"] + get_lammps_args_list(lammps_vars), cwd=path.absolute())

    if args.output:
        with open(args.output, 'w', encoding="utf-8") as output_file:
            print(f"running LAMMPS file {args.input}, output redirected to {args.output}")
            run_and_handle_error(lambda: run_with_output(stdout=output_file), args.ignore_errors)
    else:
        print(f"running LAMMPS file {args.input}, printing output to terminal")
        run_and_handle_error(run_with_output, args.ignore_errors)


def restart_run(args, path, output_file) -> bool:
    if not args.continue_flag:
        return False

    file_exists = output_file.exists()
    if not file_exists:
        if args.force:
            return False
        raise FileNotFoundError("Make sure the following file exists to restart a simulation:", output_file)

    perform_run(args, path, [["is_restart", "1"]])
    return True


def run(args, path):
    # check if output.lammpstrj exists
    output_file = path / "output.lammpstrj"

    if restart_run(args, path, output_file):
        return

    if args.force:
        output_file.unlink(missing_ok=True)
        (path / "restartfile").unlink(missing_ok=True)

    if output_file.exists():
        warn("cannot run lammps script, output.lammpstrj already exists (use -f to overwrite files)")
        print("moving on...")
        return

    perform_run(args, path)


def post_process(args, path):
    if not args.post_process:
        return

    print("running post processing...")
    run_and_handle_error(lambda: subprocess.run(python_run + ["post-process/process_displacement", f"{path}"], check=False), args.ignore_errors)
    print("succesfully ran post processing")


def visualize(args, path):
    if not args.visualize:
        return

    print("starting VMD")
    run_and_handle_error(lambda: subprocess.run(python_run + ["post-process.visualize", f"{path}"], check=False), args.ignore_errors)
    print("VMD exited")


def main():
    args = parse()
    path = Path(args.directory)

    generate(args, path)
    run(args, path)
    post_process(args, path)
    visualize(args, path)

    print("end of run.py")


if __name__ == "__main__":
    # set PYTHONUNBUFFERED=1 if python is not printing correctly
    python_run = ["python", "-m"]
    main()
