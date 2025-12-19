# Copyright (c) 2025 Lucas Dooms

from pathlib import Path
from runpy import run_path
from typing import Sequence

import numpy as np
import numpy.typing as npt

from smc_lammps.generate.default_parameters import Parameters
from smc_lammps.generate.generator import AtomIdentifier, Generator


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_parameters(path: Path) -> Parameters:
    """Load the parameters from a parameters.py file."""
    raw = run_path(path.as_posix())

    try:
        par = raw["p"]
    except KeyError:
        raise ValueError(
            f"Invalid parameters.py file: '{path}'.\nCould not extract variable named 'p'."
        )

    check_type = Parameters
    if not isinstance(par, check_type):
        raise TypeError(
            f"Invalid parameters.py file: '{path}'.\n"
            f"Parameters variable 'p' has incorrect type '{type(par)}' (expected '{check_type}')."
        )

    return par


def create_phase(phase_path: Path, options: Sequence[Generator.DynamicCoeffs]):
    """creates a file containing coefficients to dynamically load in LAMMPS scripts"""
    with open(phase_path, "w", encoding="utf-8") as phase_file:
        for args in options:
            args.write_script_bai_coeffs(phase_file)


def create_phase_wrapper(phase_path: Path, options: Sequence[Generator.DynamicCoeffs | None]):
    """filters out None values and then calls create_phase"""
    filtered_options = [opt for opt in options if opt is not None]
    create_phase(phase_path, filtered_options)


def get_closest(array, position) -> int:
    """returns the index of the array that is closest to the given position"""
    distances = np.linalg.norm(array - position, axis=1)
    return int(np.argmin(distances))


def pos_from_id(atom_id: AtomIdentifier) -> npt.NDArray[np.float32]:
    """get the position of an atom from its identifier"""
    return np.copy(atom_id[0].positions[atom_id[1]])
