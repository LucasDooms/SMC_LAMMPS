#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

# Copyright (c) 2024-2026 Lucas Dooms

import argparse
import subprocess
from argparse import Namespace
from functools import partial
from pathlib import Path
from re import compile as compile_regex
from sys import exit
from typing import Callable, Iterator, Sequence

import argcomplete
from click import confirm

from smc_lammps.console import warn
from smc_lammps.generate.util import get_project_root
from smc_lammps.post_process.util import merge_lammpstrj

PYRUN = ["python", "-m"]


class MaxIterationExceeded(RuntimeError):
    """
    Raised when a loop exceeds a large number of iterations.
    """

    MAX_ITER: int = 10000  # arbitrary limit to prevent infinite loops
    "Number of iterations after which a `MaxIterationExceeded` should be raised."

    def __str__(self):
        return (
            f"MAX_ITER ({self.MAX_ITER}) exceeded while performing the following action:\n\t"
            + super().__str__()
        )


def parse_with_double_dash(
    parser: argparse.ArgumentParser, args: list[str]
) -> tuple[Namespace, list[str]]:
    """Parse arguments with '--' splitting.

    Parses arguments by splitting before and after '--'.
    Arguments before '--' are parsed by the provided parser,
    everything after is collected into a separate, non-parsed list

    Args:
        parser: Parser for arguments before '--'.
        args: Argument list.

    Returns:
        Tuple of (parsed arguments before '--', unparsed arguments after '--').
    """
    try:
        separator_index = args.index("--")
    except ValueError:
        normal_args = args
        extra_args: list[str] = []
    else:
        normal_args = args[:separator_index]
        extra_args = args[separator_index + 1 :]  # skip over the '--'

    return parser.parse_args(normal_args), extra_args


def get_parser() -> argparse.ArgumentParser:
    """Returns the parser for the smc-lammps cli.

    Returns:
        parser
    """
    # fmt: off
    parser = argparse.ArgumentParser(
        description='runs setup scripts, LAMMPS script, post-processing, and visualization',
        epilog='visit https://github.com/LucasDooms/SMC_LAMMPS for more info'
    )

    parser.add_argument('directory', help='the directory containing parameters for LAMMPS')

    generate_and_run = parser.add_argument_group(title='generate & run')
    generate_and_run.add_argument('-g', '--generate', action='store_true', help='run the python setup scripts before executing LAMMPS')
    generate_and_run.add_argument('-r', '--run', action='store_true', help='run the LAMMPS script')
    generate_and_run.add_argument('-c', '--continue', dest='continue_flag', action='store_true', help='continue from restart file and append to existing simulation')

    gar_mods = parser.add_argument_group(title='modifiers')
    gar_mods.add_argument('-s', '--seed', help='set the seed to be used by LAMMPS, this takes precedence over the seed in default_parameters.py and parameters.py (Note: currently only works with the --generate flag)')
    gar_mods.add_argument('-e', '--executable', help='name of the LAMMPS executable to use, default: \'lmp\'', default='lmp')
    gar_mods.add_argument('-f', '--force', action='store_true', help='don\'t prompt before overwriting existing files / continuing empty simulation')
    gar_mods.add_argument('-o', '--output', help='path to dump LAMMPS output to (prints to terminal by default)')
    gar_mods.add_argument('-sf', '--suffix', help='variant of LAMMPS styles to use, default: \'opt\' (see https://docs.lammps.org/Run_options.html#suffix)', default='opt')

    post_processing = parser.add_argument_group(title='post-processing')
    post_processing.add_argument('-p', '--post-process', action='store_true', help='run the post-processing scripts after running LAMMPS')
    post_processing.add_argument('--merge', action='store_true', help='merge all created lammpstrj files into one file')
    pp_vis = post_processing.add_mutually_exclusive_group()
    pp_vis.add_argument('-v', '--visualize', action='store_true', help='open VMD after all scripts have finished')
    pp_vis.add_argument('-vd', '--visualize-datafile', action='store_true', help='shows the initial structure in VMD')
    pp_vis.add_argument('-vf', '--visualize-follow', nargs='?', choices=['arms', 'kleisin'], help='same as --visualize, but follows the SMC tracking either the arms or kleisin, default: \'arms\'', const='arms', default=None)

    other = parser.add_argument_group(title='other options')
    other.add_argument('-n', '--ignore-errors', action='store_true', help='keep running even if the previous script exited with a non-zero error code')
    other.add_argument('-i', '-in', '--input', help='path to input file to give to LAMMPS')
    other.add_argument('--clean', action='store_true', help='remove all files except parameters.py from the directory')
    other.add_argument('-q', '--quiet', action='store_true', help='print less output to the console (use --output to redirect LAMMPS output)')
    # fmt: on

    # shell autocompletion
    argcomplete.autocomplete(parser)

    return parser


def quiet_print(quiet: bool, *args, **kwargs):
    """Calls print if `quiet` is `False`.

    Args:
        quiet: if `True`, do not print.
    """
    if not quiet:
        print(*args, **kwargs)


def run_and_handle_error(
    process: Callable[[], subprocess.CompletedProcess], ignore_errors: bool, quiet: bool
):
    """Runs a process and checks the exit code.

    Runs a process and handles any non-zero exit code.
    If the exit code is non-zero and `ignore_errors` is false, the script exits.

    Args:
        process: A function that runs a process when called.
        ignore_errors: If `True`, do not exit when the process exit code is non-zero.
        quiet: Passed to `quiet_print`.
    """
    completion = process()
    if completion.returncode != 0:
        quiet_print(
            quiet, f"\n\nprocess ended with error code {completion.returncode}\n{completion}\n"
        )
        if ignore_errors:
            quiet_print(quiet, "-n (--ignore-errors) flag is set, continuing...\n")
            return

        quiet_print(
            quiet,
            "\n\n"
            "------------------------------------------------\n"
            "-- exiting smc-lammps due to subprocess error --\n"
            "------------------------------------------------\n",
        )
        exit(completion.returncode)


def find_simulation_base_directory(path: Path) -> tuple[Path, Path | None]:
    """Finds the base of a simulation directory.

    Finds the base of a simulation directory by traversing up the file tree.
    The base directory is the (first) directory containing a valid `parameters.py` file.

    .. attention::
        The validity of the `parameters.py` file is not checked by this function!

    Args:
        path: Path to start from, may be a file.

    Returns:
        Tuple of (base simulation directory, subdirectory to `path` relative to the base directory if any).

    Raises:
        FileNotFoundError: Root directory was reached without any `parameters.py` files along the way.
        MaxIterationExceeded: Exceeded maximum amount of file tree traversal.
    """
    subdir = None
    # use absolute path to allow finding parents
    try_path = path.absolute()

    def get_file_names(dir: Path) -> Iterator[str]:
        try:
            lst = (p.name for p in dir.iterdir())
        except (NotADirectoryError, FileNotFoundError):
            lst = iter(())
        return lst

    not_found_error = FileNotFoundError(
        f"Could not find 'parameters.py' in '{path}' (or its parent directories).\n"
        "Did initialization fail?"
    )

    for _ in range(MaxIterationExceeded.MAX_ITER):
        file_names = get_file_names(try_path)

        if "parameters.py" in file_names:
            break

        if subdir is None:
            subdir = Path(try_path.name)
        else:
            subdir = try_path.name / subdir
        if (
            ".git" in file_names  # do no got beyond base git directory
            or try_path == try_path.parent  # reached fixed point, cannot go further
        ):
            raise not_found_error

        # go up one
        try_path = try_path.parent
    else:
        raise not_found_error from MaxIterationExceeded("Searching for 'parameters.py'.")

    return try_path, subdir


class TaskDone:
    """
    A task.

    Tasks may be skipped, e.g. if the corresponding flag is not set.
    """

    def __init__(self, skipped: bool = False) -> None:
        self.skipped = skipped


def initialize(args: Namespace, path: Path) -> TaskDone:
    """Initializes a simulation directory.

    Creates the directory if it does not exist yet,
    and places a template `parameters.py` file inside.

    Args:
        args: Parsed arguments.
        path: Simulation base path.

    Returns:
        Task completion information.

    Raises:
        FileExistsError: The `path` is non-empty and the `--force` flag is not set.
    """
    # skip initialization for files
    if path.is_file():
        return TaskDone(skipped=True)

    destination = path / "parameters.py"
    if destination.exists():
        return TaskDone(skipped=True)

    if not path.exists():
        action_flags = [
            args.clean,
            args.generate,
            args.run,
            args.post_process,
            args.visualize_datafile,
            args.visualize_follow,
            args.visualize,
        ]
        # if any flags were specified, ask for confirmation
        if (
            not args.force
            and any(action_flags)
            and not confirm(
                f"Looks like '{path}' is not a simulation directory, do you want to create it?",
                default=False,
            )
        ):
            quiet_print(args.quiet, "no simulation directory set, exiting")
            exit(1)

        path.mkdir(parents=True)
        quiet_print(args.quiet, f"created new directory: {path.absolute()}")
    elif not args.force:
        # only initialize if empty!
        if any(path.iterdir()):
            raise FileExistsError(
                f"Cannot initialize a simulation directory inside of '{path}' because it is not empty.\n"
                "Use --force to override this behavior."
            )

    root = get_project_root()
    template_path = root / "generate" / "parameters_template.py"

    # copy file
    destination.write_bytes(template_path.read_bytes())
    quiet_print(args.quiet, f"created template parameters file: {destination.absolute()}")

    return TaskDone()


def clean(args: Namespace, path: Path) -> TaskDone:
    """Cleans a simulation directory.

    Removes all files in a simulation directory, keeping only `paremeters.py`.

    Args:
        args: Parsed arguments.
        path: Simulation base path.

    Returns:
        Task completion information.
    """
    if not args.clean:
        return TaskDone(skipped=True)

    warn(f'--clean will delete all files in "{path}" except parameters.py')
    if not confirm("Are you sure?", default=False):
        return TaskDone()

    # list of regexes for files to delete
    # (assume posix style path)
    safe_to_delete = [
        r".*\.lammpstrj$",
        r".*\.lammpstrj\.\d+$",
        r".*/log\.lammps$",
        r".*/log\.lammps\.\d+$",
        r"^lammps/.*",
        r"^post_processing_parameters\.py$",
        r"^tmp\.lammps\.variable$",
        r"^vmd/vmd\.tcl$",
        r"^vmd/vmd\.init$",
        r"^bead_id_in_time\.\w+$",
        r"^bead_indices\d+\.npz$",
    ]
    safe_to_delete = [compile_regex(string) for string in safe_to_delete]

    def is_safe_to_delete(path: Path) -> bool:
        return any(regex.match(path.as_posix()) for regex in safe_to_delete)

    def remove_recursively(child: Path, base: Path):
        if child.relative_to(base) == Path("parameters.py"):
            return

        if child.is_dir():
            for subchild in child.iterdir():
                remove_recursively(subchild, base)
            try:
                child.rmdir()
            except OSError:
                pass
            else:
                quiet_print(args.quiet, f"deleted empty directory '{child}'")
        else:
            if not is_safe_to_delete(child.relative_to(base)):
                quiet_print(args.quiet, f"unrecognized file or folder '{child}', skipping...")
                return
            child.unlink()
            quiet_print(args.quiet, f"deleted '{child}' successfully")

    remove_recursively(path, path)

    return TaskDone()


def generate(args: Namespace, path: Path) -> TaskDone:
    """Runs the generation script.

    Runs the :py:func:`smc_lammps.generate.generate` script,
    passing the `args.seed` as an argument if set.

    Args:
        args: Parsed arguments.
        path: Simulation base path.

    Returns:
        Task completion information.
    """

    if not args.generate:
        if args.seed is not None:
            warn("seed argument is ignored when -g flag is not used!")
        return TaskDone(skipped=True)

    extra_args = []
    if args.seed:
        extra_args.append(args.seed)
    quiet_print(args.quiet, "running setup file...")
    run_and_handle_error(
        lambda: subprocess.run(
            PYRUN + ["smc_lammps.generate.generate", f"{path}"] + extra_args,
            check=False,
        ),
        args.ignore_errors,
        args.quiet,
    )
    quiet_print(args.quiet, "successfully ran setup file")

    return TaskDone()


def get_lammps_args_list(lammps_vars: Sequence[list[str]]) -> list[str]:
    """Converts argument list of variables to the LAMMPS format.

    :Example:
        >>> from smc_lammps.run import get_lammps_args_list
        >>> get_lammps_args_list([['is_restart', '1'], ['seed', '1234']])
        ['-var', 'is_restart', '1', '-var', 'seed', '1234']

    Args:
        lammps_vars: Variable names and values.

    Returns:
        LAMMPS variable definitions which can be passed via the cli.
    """
    out: list[str] = []
    for var in lammps_vars:
        out += ["-var"] + var
    return out


def perform_run(
    args: Namespace, path: Path, log_file_name: str = "log.lammps", **kwargs: str | list[str]
):
    if args.input is None:
        project_root = get_project_root()
        args.input = project_root / "lammps" / "input.lmp"

    lammps_script = Path(args.input)

    # set default lammps variables if not set yet
    kwargs.setdefault("lammps_root_dir", str(lammps_script.parent.absolute()))
    output_path = kwargs.setdefault("output_path", "output")
    kwargs.setdefault("output_file_name", "output.lammpstrj")

    # check for spaces in output_path (for LAMMPS compatibility)
    assert isinstance(output_path, str)
    if " " in output_path:
        raise ValueError(
            f"Found spaces in path '{output_path}', this not supported by the LAMMPS script."
        )

    output_path = path / Path(output_path)
    output_path.mkdir(exist_ok=True)

    parsed_kwargs = [[k, *([v] if isinstance(v, str) else v)] for k, v in kwargs.items()]
    lammps_vars = get_lammps_args_list(parsed_kwargs)

    command = [
        f"{args.executable}",
        "-sf",
        f"{args.suffix}",
        "-in",
        f"{lammps_script.absolute()}",
        "-log",
        f"{output_path / log_file_name}",
    ]
    if args.suffix == "kk":
        command += ["-kokkos", "on"]
    # add variables (-var) at the end
    command += lammps_vars

    run_with_output = partial(subprocess.run, command, cwd=path.absolute())

    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            quiet_print(
                args.quiet, f"running LAMMPS file {args.input}, output redirected to {args.output}"
            )
            quiet_print(args.quiet, command)
            run_and_handle_error(
                lambda: run_with_output(stdout=output_file), args.ignore_errors, args.quiet
            )
    else:
        quiet_print(args.quiet, f"running LAMMPS file {args.input}, printing output to terminal")
        quiet_print(args.quiet, command)
        run_and_handle_error(run_with_output, args.ignore_errors, args.quiet)


def restart_run(args: Namespace, path: Path, output_file: Path) -> TaskDone:
    if not args.continue_flag:
        return TaskDone(skipped=True)

    # TODO: check that all necessary files have been generated?
    checks: list[tuple[Path, bool]] = [
        (output_file, True),
        (path / "lammps" / "restartfile", False),
    ]

    for file, allow_reroute in checks:
        if not file.exists():
            if allow_reroute and args.force:
                return TaskDone(skipped=True)
            raise FileNotFoundError(
                f"Make sure the following file exists to restart a simulation: {output_file}"
            )

    # find an output file name that is not taken yet
    for suffix in range(1, MaxIterationExceeded.MAX_ITER):
        new_output_file = output_file.with_name(f"{output_file.name}.{suffix}")
        if not new_output_file.exists():
            # use same suffix for log.lammps
            new_log_file_name = f"log.lammps.{suffix}"
            new_log_file = output_file.parent / new_log_file_name
            if new_log_file.exists():
                warn(f"log file '{new_log_file}' already exists and will get overwritten")
            break
    else:
        raise FileExistsError(
            f"Could not create new '{output_file}.x' file."
        ) from MaxIterationExceeded("Searching for available file name.")

    quiet_print(
        args.quiet,
        f"your run will continue and the output trajectory will be placed into {new_output_file}",
    )
    perform_run(
        args,
        path,
        log_file_name=new_log_file_name,
        is_restart="1",
        output_path=new_output_file.parent.relative_to(path).as_posix(),
        output_file_name=new_output_file.name,
    )

    return TaskDone()


def run(args: Namespace, path: Path) -> TaskDone:
    if not args.run:
        return TaskDone(skipped=True)

    output_path = path / "output"
    # check if output.lammpstrj exists
    output_file = output_path / "output.lammpstrj"

    if not restart_run(args, path, output_file).skipped:
        return TaskDone()

    if args.force:
        output_file.unlink(missing_ok=True)
        (path / "lammps" / "restartfile").unlink(missing_ok=True)
        for perspective_file in output_path.glob("perspective.*.lammpstrj"):
            perspective_file.unlink()
        for snapshot_file in (output_path / "snapshots").glob("*.lammpstrj"):
            snapshot_file.unlink()

    if output_file.exists():
        warn(
            f"cannot run lammps script, '{output_file}' already exists (use -f to overwrite files)"
        )
        quiet_print(args.quiet, "moving on...")
        return TaskDone()

    perform_run(
        args,
        path,
        output_path=output_path.relative_to(path).as_posix(),
        output_file_name=output_file.name,
    )

    return TaskDone()


def merge(args: Namespace, path: Path) -> TaskDone:
    """Merges lammpstrj files together.

    Merges lammpstrj files together by calling :py:func:`merge_lammpstrj`
    on any files that match the `output.lammpstrj*` glob pattern.

    This can be useful after a restart run.

    Args:
        args: Parsed arguments.
        path: Simulation base path.

    Returns:
        Task completion information.
    """
    if not args.merge:
        return TaskDone(skipped=True)

    merge_path = path / "output"
    base_file_name = "output.lammpstrj"

    files = filter(lambda p: not p.is_dir(), merge_path.glob(f"{base_file_name}*"))
    files = sorted(files)

    if len(files) <= 1:
        quiet_print(
            args.quiet, f"need at least two files in order to merge, found {len(files)} file(s)"
        )
        return TaskDone(skipped=True)

    base_file, files = files[0], files[1:]

    for other_file in files:
        quiet_print(
            args.quiet,
            f"merging '{base_file.relative_to(path)}' and '{other_file.relative_to(path)}'",
        )
        merge_lammpstrj(base_file, other_file, delete_after=True)

    return TaskDone()


def post_process(args: Namespace, path: Path) -> TaskDone:
    if not args.post_process:
        return TaskDone(skipped=True)

    quiet_print(args.quiet, "running post processing...")
    run_and_handle_error(
        lambda: subprocess.run(
            PYRUN + ["smc_lammps.post_process.process_displacement", f"{path}"],
            check=False,
        ),
        args.ignore_errors,
        args.quiet,
    )
    quiet_print(args.quiet, "successfully ran post processing")

    return TaskDone()


def visualize_datafile(args: Namespace, path: Path, subdir: Path | None) -> TaskDone:
    if not args.visualize_datafile:
        return TaskDone(skipped=True)

    vmd_dir = path / "vmd"
    vmd_dir.mkdir(exist_ok=True)

    if subdir is None:
        subdir = Path("lammps/datafile_positions")

    # create VMD tcl script to automatically run topotools command
    tcl_script = path / "vmd" / "vmd.tcl"
    with open(tcl_script, "w", encoding="utf-8") as vmdfile:
        vmdfile.write(f"topo readlammpsdata {path / subdir}\n")
        vmdfile.write("mol modstyle 0 0 cpk\n")

    quiet_print(args.quiet, "starting VMD")
    run_and_handle_error(
        lambda: subprocess.run(["vmd", "-e", f"{tcl_script.absolute()}"], check=False),
        args.ignore_errors,
        args.quiet,
    )
    quiet_print(args.quiet, "VMD exited")

    return TaskDone()


def create_perspective_file(args: Namespace, path: Path, subdir: Path | None) -> Path:
    if subdir is None:
        read_from_file = path / "output" / "output.lammpstrj"
    else:
        read_from_file = path / subdir

    prefix = f"perspective.{args.visualize_follow}"
    perspective_file = read_from_file.parent / f"{prefix}.{read_from_file.name}"
    if not args.force and perspective_file.exists():
        quiet_print(args.quiet, f"found '{perspective_file}'")
        return perspective_file

    quiet_print(args.quiet, "creating new lammpstrj file")
    run_and_handle_error(
        lambda: subprocess.run(
            PYRUN
            + [
                "smc_lammps.post_process.smc_perspective",
                f"{read_from_file}",
                f"{perspective_file}",
                f"{path / 'post_processing_parameters.py'}",
                f"{args.visualize_follow}",
                f"{str(args.force).lower()}",
            ],
            check=False,
        ),
        args.ignore_errors,
        args.quiet,
    )
    quiet_print(args.quiet, f"created '{perspective_file}'")

    return perspective_file


def start_visualize_script(args: Namespace, path: Path, other_args: list[str]):
    quiet_print(args.quiet, "starting VMD")
    run_and_handle_error(
        lambda: subprocess.run(
            PYRUN + ["smc_lammps.post_process.visualize", f"{path}"] + other_args,
            check=False,
        ),
        args.ignore_errors,
        args.quiet,
    )
    quiet_print(args.quiet, "VMD exited")


def visualize_follow(args: Namespace, path: Path, subdir: Path | None) -> TaskDone:
    if args.visualize_follow is None:
        return TaskDone(skipped=True)

    perspective_file = create_perspective_file(args, path, subdir)

    file_arg = ["--file_name", perspective_file.relative_to(path)]

    start_visualize_script(args, path, file_arg + args.sub_args)

    return TaskDone()


def visualize(args: Namespace, path: Path, subdir: Path | None) -> TaskDone:
    if not visualize_datafile(args, path, subdir).skipped:
        return TaskDone()

    if not visualize_follow(args, path, subdir).skipped:
        return TaskDone()

    if not args.visualize:
        return TaskDone(skipped=True)

    file_arg = []
    if subdir is not None:
        if not (path / subdir).is_file():
            raise ValueError(f"Cannot visualize: '{path / subdir}' is not a file.")
        file_arg = ["--file_name", subdir]

    start_visualize_script(args, path, file_arg + args.sub_args)

    return TaskDone()


def execute(args: Namespace):
    """Executes a sequence of tasks depending on the provided arguments.

    The following tasks are executed (if enabled) in order:
        - :py:func:`initialize`
        - :py:func:`clean`
        - :py:func:`generate`
        - :py:func:`run`
        - :py:func:`merge`
        - :py:func:`post_process`
        - :py:func:`visualize`

    Args:
        args: Parsed arguments.
    """
    path = Path(args.directory)

    # --continue flag implies the --run flag
    if args.continue_flag:
        args.run = True

    tasks: list[TaskDone] = []

    try:
        # check if already inside of a simulation directory
        path, subdir = find_simulation_base_directory(path)
    except FileNotFoundError as e:
        if isinstance(e.__cause__, MaxIterationExceeded):
            warn(
                f"could not find base directory, see details below\n\n{e}\n**caused by**\n{e.__cause__}"
            )

        # no simulation directory found, try to create it
        tasks += [
            initialize(args, path),
        ]
        path, subdir = find_simulation_base_directory(path)

    quiet_print(args.quiet, f"using base directory '{path}'")

    tasks += [
        clean(args, path),
        generate(args, path),
        run(args, path),
        merge(args, path),
        post_process(args, path),
        visualize(args, path, subdir),
    ]

    if all(map(lambda task: task.skipped, tasks)):
        quiet_print(args.quiet, "nothing to do, use -gr to generate and run")

    quiet_print(args.quiet, "end of smc-lammps (run.py)")


def parse(argv: list[str]) -> Namespace:
    """
    Parses the argument list and returns argparse `Namespace`.

    :param argv: argument list (`from sys import argv`)
    :return: object holding command line options
    """

    parser = get_parser()
    # remove first argument from argv (name of exe)
    args, extra_args = parse_with_double_dash(parser, argv[1:])
    args.sub_args = extra_args

    return args


def main():
    """
    This is the entry point for the smc-lammps cli.

    Use `smc-lammps -h` for help.

    See :py:func:`smc_lammps.run.get_parser` for the parser definition.
    """
    from sys import argv

    args = parse(argv)
    execute(args)


if __name__ == "__main__":
    # set PYTHONUNBUFFERED=1 if python is not printing correctly
    main()
