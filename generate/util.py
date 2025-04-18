from pathlib import Path
from typing import Any, List

import numpy as np

from generate.generator import Generator


def create_phase(
    generator: Generator, phase_path: Path, options: List[Generator.DynamicCoeffs]
):
    """creates a file containing coefficients to dynamically load in LAMMPS scripts"""

    def apply(function, file, list_of_args: List[Any]):
        for args in list_of_args:
            function(file, args)

    with open(phase_path, "w", encoding="utf-8") as phase_file:
        apply(generator.write_script_bai_coeffs, phase_file, options)


def get_closest(array, position) -> int:
    """returns the index of the array that is closest to the given position"""

    distances = np.linalg.norm(array - position, axis=1)
    return int(np.argmin(distances))
