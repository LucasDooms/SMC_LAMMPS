from pathlib import Path
from typing import Any, List

from generator import Generator


def create_phase(
    generator: Generator, phase_path: Path, options: List[Generator.DynamicCoeffs]
):
    def apply(function, file, list_of_args: List[Any]):
        for args in list_of_args:
            function(file, args)

    with open(phase_path, "w", encoding="utf-8") as phase_file:
        apply(generator.write_script_bai_coeffs, phase_file, options)
