from dataclasses import dataclass

import numpy as np

from smc_lammps.generate.generator import BAI, COORD_TYPE, AtomGroup, AtomType, MoleculeId, Nx3Array
from smc_lammps.generate.structures.structure_creator import get_circle_segment_unit_radius


def get_structure() -> Nx3Array:
    positions = get_circle_segment_unit_radius(
        20,
        end_inclusive=True,
        theta_start=0,
        theta_end=2 * np.pi,
        normal_direction=[1, 0, 0],
    )

    positions *= 4  # nm

    positions += np.array([10, -2, 0], dtype=COORD_TYPE)

    return positions


@dataclass
class InertSMC:
    positions: Nx3Array

    def __post_init__(self) -> None:
        self.group = AtomGroup(self.positions, AtomType(mass=1), MoleculeId.get_next())

    def get_bonds(self) -> list[BAI]:
        return []
