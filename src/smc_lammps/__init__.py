"""
MD Simulations of SMC complexes using LAMMPS.

- :py:mod:`smc_lammps.generate`: Input parameter processing and file generation.
- :py:mod:`smc_lammps.post_process`: Data analysis.
- :py:mod:`smc_lammps.reader`: Lammps trajectory file parsing tools.
"""

from smc_lammps import generate, post_process, reader
from smc_lammps.run import main

__all__ = ["main", "generate", "post_process", "reader"]
