import argparse
import subprocess
from functools import partial
from pathlib import Path
from warnings import warn


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='runs setup scripts, LAMMPS script, post-processing, and visualization',
        epilog='visit https://github.com/LucasDooms/SMC_LAMMPS for more info'
    )

    parser.add_argument('directory', help='the directory containing parameters for LAMMPS')

    generate_and_run = parser.add_argument_group(title='generate & run')
    generate_and_run.add_argument('-g', '--generate', action='store_true', help='run the python setup scripts before executing LAMMPS')
    generate_and_run.add_argument('-r', '--run', action='store_true', help='run the LAMMPS script')
    generate_and_run.add_argument('-s', '--seed', help='set the seed to be used by LAMMPS, this takes precedence over the seed in default_parameters.py and parameters.py')

    gar_mods = parser.add_argument_group(title='modifiers')
    gar_mods.add_argument('-e', '--executable', help='name of the LAMMPS executable to use, default: \'lmp\'', default='lmp')
    gar_mods.add_argument('-f', '--force', action='store_true', help='don\'t prompt before overwriting existing files / continuing empty simulation')
    gar_mods.add_argument('-c', '--continue', dest='continue_flag', action='store_true', help='continue from restart file and append to existing simulation')
    gar_mods.add_argument('-o', '--output', help='path to dump LAMMPS output to (prints to terminal by default)')
    gar_mods.add_argument('-sf', '--suffix', help='variant of LAMMPS styles to use, default: \'opt\' (see https://docs.lammps.org/Run_options.html#suffix)', default='opt')

    post_processing = parser.add_argument_group(title='post-processing')
    post_processing.add_argument('-p', '--post-process', action='store_true', help='run the post-processing scripts after running LAMMPS')
    post_processing.add_argument('-v', '--visualize', action='store_true', help='open VMD after all scripts have finished')

    other = parser.add_argument_group(title='other options')
    other.add_argument('-n', '--ignore-errors', action='store_true', help='keep running even if the previous script exited with a non-zero error code')
    other.add_argument('-i', '-in', '--input', help='path to input file to give to LAMMPS', default='lammps/input.lmp')

    return parser.parse_args()


def run_and_handle_error(process, ignore_errors: bool):
    completion: subprocess.CompletedProcess = process()
    if completion.returncode != 0:
        message = (
            f"\n\nprocess ended with error code {completion.returncode}\n{completion}\n"
        )
        print(message)
        if ignore_errors:
            print("-n (--ignore-errors) flag is set, continuing...\n")
            return
        raise ChildProcessError()


class TaskDone:
    def __init__(self, skipped=False) -> None:
        self.skipped = skipped


def generate(args, path) -> TaskDone:
    if not args.generate:
        if args.seed is not None:
            warn("seed argument is ignored when -g flag is not used!")
        return TaskDone(skipped=True)

    extra_args = []
    if args.seed:
        extra_args.append(args.seed)
    print("running setup file...")
    run_and_handle_error(
        lambda: subprocess.run(
            python_run + ["generate.generate+parse", f"{path}"] + extra_args,
            check=False,
        ),
        args.ignore_errors,
    )
    print("successfully ran setup file")

    return TaskDone()


def get_lammps_args_list(lammps_vars):
    out = []
    for var in lammps_vars:
        out += ["-var"] + var
    return out


def perform_run(args, path, lammps_vars=()):
    command = [
        f"{args.executable}",
        "-sf",
        f"{args.suffix}",
        "-in",
        f"{Path(args.input).absolute()}",
    ] + get_lammps_args_list(lammps_vars)
    if args.suffix == "kk":
        command += ["-kokkos", "on"]

    run_with_output = partial(subprocess.run, command, cwd=path.absolute())

    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            print(
                f"running LAMMPS file {args.input}, output redirected to {args.output}"
            )
            print(command)
            run_and_handle_error(
                lambda: run_with_output(stdout=output_file), args.ignore_errors
            )
    else:
        print(f"running LAMMPS file {args.input}, printing output to terminal")
        print(command)
        run_and_handle_error(run_with_output, args.ignore_errors)


def restart_run(args, path, output_file) -> TaskDone:
    if not args.continue_flag:
        return TaskDone(skipped=True)

    file_exists = output_file.exists()
    if not file_exists:
        if args.force:
            return TaskDone(skipped=True)
        raise FileNotFoundError(
            "Make sure the following file exists to restart a simulation:", output_file
        )

    perform_run(args, path, [["is_restart", "1"]])

    return TaskDone()


def run(args, path) -> TaskDone:
    if not args.run:
        return TaskDone(skipped=True)

    # check if output.lammpstrj exists
    output_file = path / "output.lammpstrj"

    if not restart_run(args, path, output_file).skipped:
        return TaskDone()

    if args.force:
        output_file.unlink(missing_ok=True)
        (path / "restartfile").unlink(missing_ok=True)

    if output_file.exists():
        warn(
            "cannot run lammps script, output.lammpstrj already exists (use -f to overwrite files)"
        )
        print("moving on...")
        return TaskDone()

    perform_run(args, path)

    return TaskDone()


def post_process(args, path) -> TaskDone:
    if not args.post_process:
        return TaskDone(skipped=True)

    print("running post processing...")
    run_and_handle_error(
        lambda: subprocess.run(
            python_run + ["post-process.process_displacement", f"{path}"], check=False
        ),
        args.ignore_errors,
    )
    print("succesfully ran post processing")

    return TaskDone()


def visualize(args, path) -> TaskDone:
    if not args.visualize:
        return TaskDone(skipped=True)

    print("starting VMD")
    run_and_handle_error(
        lambda: subprocess.run(
            python_run + ["post-process.visualize", f"{path}"], check=False
        ),
        args.ignore_errors,
    )
    print("VMD exited")

    return TaskDone()


def main():
    args = parse()
    path = Path(args.directory)

    tasks = [
        generate(args, path),
        run(args, path),
        post_process(args, path),
        visualize(args, path),
    ]

    if all(map(lambda task: task.skipped, tasks)):
        print("nothing to do, use -gr to generate and run")

    print("end of run.py")


if __name__ == "__main__":
    # set PYTHONUNBUFFERED=1 if python is not printing correctly
    python_run = ["python", "-m"]
    main()
