from typing import List, Sequence

from smc_lammps.generate.generator import AtomIdentifier, Generator


def atomIds_to_LAMMPS_ids(gen: Generator, lst: Sequence[AtomIdentifier]) -> List[int]:
    return [gen.get_atom_index(atomId) for atomId in lst]
