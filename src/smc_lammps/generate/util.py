# Copyright (c) 2025-2026 Lucas Dooms

from pathlib import Path
from runpy import run_path
from typing import Sequence

import numpy as np
import numpy.typing as npt

from smc_lammps.generate.default_parameters import Parameters
from smc_lammps.generate.generator import AtomIdentifier, Generator, Nx3Array


def get_project_root() -> Path:
    """Returns the project root directory (``src/smc_lammps``).

    Returns:
        Project root path.
    """
    return Path(__file__).parent.parent


def get_parameters(file: Path) -> Parameters:
    """Reads the parameters from a python file.

    Runs the given :py:attr:`file` as a python script
    and loads the variable named ``p``.

    Args:
        file: Python script containing parameters.

    Returns:
        Loaded parameters.

    Raises:
        ValueError: Could not find variable ``p`` in the file.
        TypeError: The variable ``p`` is not an instance of :py:class:`Parameters`.
    """
    raw = run_path(file.as_posix())

    try:
        par = raw["p"]
    except KeyError:
        raise ValueError(
            f"Invalid parameters.py file: '{file}'.\nCould not extract variable named 'p'."
        )

    check_type = Parameters
    if not isinstance(par, check_type):
        raise TypeError(
            f"Invalid parameters.py file: '{file}'.\n"
            f"Parameters variable 'p' has incorrect type '{type(par)}' (expected '{check_type}')."
        )

    return par


def load_parameters(path: Path) -> Parameters:
    """Reads the parameters from a python file.

    Runs the python file ``parameters.py`` under :py:attr:`path`
    and loads the variable named ``p``.

    .. Note::
        See :py:func:`get_parameters` for possible errors.

    Args:
        path: Simulation base path.

    Returns:
        Loaded parameters.

    Raises:
        FileNotFoundError: No ``parameters.py`` under :py:attr:`path`.
    """
    file = path / "parameters.py"
    if not file.exists():
        raise FileNotFoundError(f"Could not find parameters.py: '{file}' does not exist.")

    return get_parameters(file)


def create_phase(file: Path, options: Sequence[Generator.DynamicCoeffs]) -> None:
    """Creates a file containing coefficients to dynamically load in LAMMPS scripts.

    .. Attention::
        The given :py:attr:`file` is overwritten.

    Args:
        phase_path: File to write LAMMPS commands to.
        options: Coefficients that are passed to :py:meth:`smc_lammps.generate.generator.Generator.DynamicCoeffs.write_script_bai_coeffs`.
    """
    with open(file, "w", encoding="utf-8") as phase_file:
        for args in options:
            args.write_script_bai_coeffs(phase_file)


def create_phase_wrapper(
    phase_path: Path, options: Sequence[Generator.DynamicCoeffs | None]
) -> None:
    """Filters out None values and calls :py:func:`create_phase`.

    Args:
        phase_path: File to write LAMMPS commands to.
        options: Coefficients that are passed to :py:meth:`smc_lammps.generate.generator.Generator.DynamicCoeffs.write_script_bai_coeffs`.
    """
    filtered_options = [opt for opt in options if opt is not None]
    create_phase(phase_path, filtered_options)


def get_closest(array: Nx3Array, position) -> int:
    """Returns the index in the :py:attr:`array` that is closes to the :py:attr:`position`.

    Args:
        array: An (n, 3) array of points.
        position: A single 3D point.

    Returns:
        Index.
    """
    distances = np.linalg.norm(array - position, axis=1)
    return int(np.argmin(distances))


def pos_from_id(atom_id: AtomIdentifier) -> npt.NDArray[np.float32]:
    """Returns the 3D position of an atom from its id.

    .. Note::
        This returns a copied array,
        which can be edited without affecting the original position.

    Args:
        atom_id: The atom id.

    Returns:
        A 3D Point.
    """
    return np.copy(atom_id[0].positions[atom_id[1]])
