# Copyright (c) 2026 Lucas Dooms

from pathlib import Path

from smc_lammps.generate.default_parameters import Parameters
from smc_lammps.generate.util import get_parameters
from smc_lammps.run import execute as run_execute
from smc_lammps.run import parse as run_parse


def execute(argv: list[str]):
    args = run_parse(argv)
    run_execute(args)


def test_init(tmp_path):
    execute(["smc-lammps", str(tmp_path)])
    parameters_file = Path(tmp_path) / "parameters.py"
    assert parameters_file.exists()
    par = get_parameters(parameters_file)
    assert isinstance(par, Parameters)


def test_generate(tmp_path):
    execute(["smc-lammps", str(tmp_path), "-g"])
    files = [
        "lammps/datafile_positions",
        "lammps/datafile_coeffs",
        "lammps/parameterfile",
        "post_processing_parameters.py",
    ]
    files = [Path(tmp_path) / file for file in files]
    for file in files:
        assert file.exists()
