# Copyright (c) 2026 Lucas Dooms

from pathlib import Path

from smc_lammps.generate.default_parameters import Parameters
from smc_lammps.generate.util import get_project_root, load_parameters


def test_root():
    """Checks that the root directory contains the expected files and directories."""
    root = get_project_root()
    assert (root / "run.py").is_file()
    assert (root / "lammps").is_dir()


def test_parameters(tmp_path: Path):
    """Checks that the template parameters can be loaded."""
    root = get_project_root()
    # copy file
    (tmp_path / "parameters.py").write_bytes(
        (root / "generate" / "parameters_template.py").read_bytes()
    )
    parameters = load_parameters(tmp_path)
    assert isinstance(parameters, Parameters)
