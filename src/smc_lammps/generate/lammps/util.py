from typing import Sequence

from smc_lammps.generate.generator import AtomIdentifier, Generator


def atomIds_to_LAMMPS_ids(gen: Generator, atomIds: Sequence[AtomIdentifier]) -> list[int]:
    return [gen.get_atom_index(atomId) for atomId in atomIds]
