from typing import List, Tuple

import numpy as np

from smc_lammps.generate.generator import AtomGroup, AtomIdentifier, Nx3Array


class Polymer:
    """One connected polymer / strand, comprised of any number of atom groups"""

    def __init__(self, *atom_groups: AtomGroup) -> None:
        self.atom_groups: List[AtomGroup] = []
        self.add(*atom_groups)

    def add(self, *atom_groups: AtomGroup) -> None:
        self.atom_groups += atom_groups

    def split(self, split: AtomIdentifier) -> Tuple[AtomGroup, AtomGroup]:
        """split the polymer in two pieces, with the split atom id part of the second group.
        Note: this simply changes the underlying atom groups"""
        id = self.atom_groups.index(split[0])
        self.atom_groups.remove(split[0])
        pos1 = split[0].positions[: split[1]]
        pos2 = split[0].positions[split[1] :]

        if len(pos1) == 0 or len(pos2) == 0:
            raise ValueError("Empty group produced by split!")

        args = (
            split[0].type,
            split[0].molecule_index,
            split[0].polymer_bond_type,
            split[0].polymer_angle_type,
        )
        groups = (
            AtomGroup(pos1, *args),
            AtomGroup(pos2, *args),
        )
        for grp in groups[::-1]:
            self.atom_groups.insert(id, grp)

        return groups

    def full_list(self) -> Nx3Array:
        return np.concatenate([grp.positions for grp in self.atom_groups])

    def full_list_length(self) -> int:
        return len(self.full_list())

    def handle_negative_index(self, index: int) -> int:
        if index >= 0:
            return index
        index += self.full_list_length()
        if index < 0:
            raise IndexError("Index out of range.")
        return index

    def get_id_from_list_index(self, index: int) -> AtomIdentifier:
        index = self.handle_negative_index(index)

        for grp in self.atom_groups:
            if index < len(grp.positions):
                return (grp, index)
            index -= len(grp.positions)

        raise IndexError(f"index {index} out of bounds for atom groups.")

    def all_indices_list(
        self,
    ) -> List[Tuple[AtomIdentifier, AtomIdentifier]]:
        return [((dna_grp, 0), (dna_grp, -1)) for dna_grp in self.atom_groups]

    def indices_list_from_to(self, from_index: int, to_index: int) -> List[Tuple[AtomIdentifier, AtomIdentifier]]:
        from_index = self.handle_negative_index(from_index)
        to_index = self.handle_negative_index(to_index)

        if from_index > to_index:
            return []

        lst = []
        # go to from_index
        i = 0
        for i in range(len(self.atom_groups)):
            grp = self.atom_groups[i]
            if from_index < len(grp.positions):
                if to_index < len(grp.positions):
                    lst.append(((grp, from_index), (grp, to_index)))
                    return lst
                lst.append(((grp, from_index), (grp, -1)))
                break
            from_index -= len(grp.positions)
            to_index -= len(grp.positions)

        for j in range(i + 1, len(self.atom_groups)):
            grp = self.atom_groups[j]
            lst.append(((grp, 0), (grp, -1)))
            if to_index < len(grp.positions):
                lst.pop()
                lst.append(((grp, 0), (grp, to_index)))
                return lst
            to_index -= len(grp.positions)

        raise IndexError(f"index range ({from_index}, {to_index}) out of bounds for atom groups.")

    def indices_list_to(self, index: int) -> List[Tuple[AtomIdentifier, AtomIdentifier]]:
        index = self.handle_negative_index(index)

        lst = []
        for grp in self.atom_groups:
            lst.append(((grp, 0), (grp, -1)))
            if index < len(grp.positions):
                lst.pop()
                lst.append(((grp, 0), (grp, index)))
                return lst
            index -= len(grp.positions)

        raise IndexError(f"index {index} out of bounds for atom groups.")

    def convert_ratio(self, ratio: float) -> int:
        if ratio < 0.0 or ratio > 1.0:
            raise IndexError(f"index ratio {ratio} is invalid.")

        index = int(ratio * self.full_list_length())
        if index == self.full_list_length():
            # case where ratio is (almost) equal to 1
            index -= 1

        return index

    def indices_list_from_to_percent(self, from_ratio: float, to_ratio: float) -> List[Tuple[AtomIdentifier, AtomIdentifier]]:
        from_index = self.convert_ratio(from_ratio)
        to_index = self.convert_ratio(to_ratio)

        return self.indices_list_from_to(from_index, to_index)

    def indices_list_to_percent(self, ratio: float) -> List[Tuple[AtomIdentifier, AtomIdentifier]]:
        index = self.convert_ratio(ratio)

        return self.indices_list_to(index)

    def first_id(self) -> AtomIdentifier:
        return self.get_id_from_list_index(0)

    def last_id(self) -> AtomIdentifier:
        return self.get_id_from_list_index(-1)

    def get_percent_id(self, ratio: float) -> AtomIdentifier:
        return self.get_id_from_list_index(int(ratio * self.full_list_length()))
