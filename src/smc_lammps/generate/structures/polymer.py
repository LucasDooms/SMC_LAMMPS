from __future__ import annotations

import numpy as np

from smc_lammps.generate.generator import AtomGroup, AtomIdentifier, Nx3Array


class Polymer:
    """One connected polymer / strand, comprised of any number of atom groups"""

    def __init__(self, *atom_groups: AtomGroup) -> None:
        self.atom_groups: list[AtomGroup] = []
        # atoms that should move whenever the polymer is moved
        self.tagged_atoms: dict[
            int, list[AtomIdentifier]
        ] = {}  # maps an absolute index in the polymer to a list of tagged atoms
        self.add(*atom_groups)

    def add(self, *atom_groups: AtomGroup) -> None:
        self.atom_groups += atom_groups

    def add_tagged_atoms(self, polymer_atom: AtomIdentifier, *atom_ids: AtomIdentifier) -> None:
        id = self.get_absolute_index(polymer_atom)
        if id not in self.tagged_atoms:
            self.tagged_atoms[id] = []
        self.tagged_atoms[id] += list(atom_ids)

    def split(self, split: AtomIdentifier) -> tuple[AtomGroup, AtomGroup]:
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

    def move(self, shift: Nx3Array, rng: tuple[int, int]) -> Polymer:
        index = self.handle_negative_index(rng[0])
        last_index = self.handle_negative_index(rng[1]) - index

        if last_index < 0:
            raise ValueError(f"Invalid range {rng}.")

        absolute_shift = 0

        remaining = [*self.atom_groups]
        while index >= 0:
            grp = remaining.pop(0)
            if index < grp.n:
                remaining.insert(0, grp)
                start_offset = index
                break
            index -= grp.n
            absolute_shift += grp.n
        else:
            raise IndexError(f"Start of range {rng} is out of bounds.")

        def update_tagged(rng: slice, start_id: int) -> None:
            for id in self.tagged_atoms:
                if rng.start <= id - start_id < rng.stop:
                    for atom_id in self.tagged_atoms[id]:
                        atom_id[0].positions[atom_id[1]] += shift

        for grp in remaining:
            if last_index < len(grp.positions):
                update_range = slice(start_offset, last_index + 1)
                update_tagged(update_range, absolute_shift)
                grp.positions[update_range] += shift
                break
            update_range = slice(start_offset, grp.n)
            update_tagged(update_range, absolute_shift)
            grp.positions[update_range] += shift
            start_offset = 0
            last_index -= grp.n
            absolute_shift += grp.n

        return self

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

        raise IndexError(f"Index {index} out of bounds for atom groups.")

    def get_absolute_index(self, atom_id: AtomIdentifier) -> int:
        group = atom_id[0]
        rel_index = atom_id[1]
        index = 0
        for grp in self.atom_groups:
            if grp == group:
                if rel_index < 0:
                    rel_index += group.n
                assert rel_index >= 0
                index += rel_index
                break
            index += grp.n
        else:
            raise IndexError(f"Atom index of {atom_id} out of bounds.")

        return index

    def all_indices_list(
        self,
    ) -> list[tuple[AtomIdentifier, AtomIdentifier]]:
        return [((dna_grp, 0), (dna_grp, -1)) for dna_grp in self.atom_groups]

    def indices_list_from_to(
        self, from_index: int, to_index: int
    ) -> list[tuple[AtomIdentifier, AtomIdentifier]]:
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

    def indices_list_to(self, index: int) -> list[tuple[AtomIdentifier, AtomIdentifier]]:
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

    def indices_list_from_to_percent(
        self, from_ratio: float, to_ratio: float
    ) -> list[tuple[AtomIdentifier, AtomIdentifier]]:
        from_index = self.convert_ratio(from_ratio)
        to_index = self.convert_ratio(to_ratio)

        return self.indices_list_from_to(from_index, to_index)

    def indices_list_to_percent(self, ratio: float) -> list[tuple[AtomIdentifier, AtomIdentifier]]:
        index = self.convert_ratio(ratio)

        return self.indices_list_to(index)

    def first_id(self) -> AtomIdentifier:
        return self.get_id_from_list_index(0)

    def last_id(self) -> AtomIdentifier:
        return self.get_id_from_list_index(-1)

    def get_percent_id(self, ratio: float) -> AtomIdentifier:
        return self.get_id_from_list_index(int(ratio * self.full_list_length()))
