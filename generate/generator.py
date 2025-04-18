# Copyright (c) 2024 Lucas Dooms

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Any
from enum import Enum
import numpy as np


"""
generator.py
------------
How to use:
    - create a Generator instance
    - create AtomType instances for each type you need
    - create BAI_Type for Bond/Angle/Improper types
    - define AtomGroup with positions and type
    - define BAI for interactions between specific atoms
    - define PairWise, and use add_interaction to define global interactions
    - append all defined AtomGroup, BAI, and PairWise instances to the lists in Generator
    - call Generator.write(file) to create the final datafile
"""

class AtomType:

    __index = 1

    def __init__(self, mass: float = 1.0) -> None:
        self._index = None
        self.mass = mass

    @property
    def index(self) -> int:
        if self._index is None:
            self._index = self.__class__.__index
            self.__class__.__index += 1
        return self._index


class MoleculeId:

    __index = 0

    @classmethod
    def get_next(cls) -> int:
        cls.__index += 1
        return cls.__index


class BAI_Kind(Enum):
    BOND = 1
    ANGLE = 2
    IMPROPER = 3


# number of arguments (atom ids) needed to define a BAI interaction
length_lookup = dict({
    BAI_Kind.BOND: 2,
    BAI_Kind.ANGLE: 3,
    BAI_Kind.IMPROPER: 4
})


class BAI_Type:

    indices = {kind: 1 for kind in BAI_Kind}

    def __init__(self, kind: BAI_Kind, coefficients: str = "") -> None:
        """coefficients will be printed after the index in a datafile
        e.g. 3 harmonic 2 5
        (where 3 is the index and the rest is the coefficients string)"""
        self._index = None
        self.kind = kind
        self.coefficients = coefficients

    @property
    def index(self) -> int:
        if self._index is None:
            self._index = self.indices[self.kind]
            self.indices[self.kind] += 1
        return self._index


class AtomGroup:

    """
        positions: list of 3d positions of (n) atoms [[x, y, z], ...]
        atom_type: type of the atoms
        molecule_index (int): the molecule index which is needed for atom_style molecule
        polymer_bond_type: if None -> no bonds, otherwise all atoms will form bonds as a polymer
    """

    def __init__(self, positions: List[List[float]],
                 atom_type: AtomType, molecule_index: int, polymer_bond_type: BAI_Type | None = None, polymer_angle_type: BAI_Type | None = None) -> None:
        self.n = len(positions)
        self.positions = positions
        self.type = atom_type
        self.molecule_index = molecule_index
        if polymer_bond_type is not None:
            assert(polymer_bond_type.kind == BAI_Kind.BOND)
        self.polymer_bond_type = polymer_bond_type
        if polymer_angle_type is not None:
            assert(polymer_angle_type.kind == BAI_Kind.ANGLE)
        self.polymer_angle_type = polymer_angle_type


AtomIdentifier = Tuple[AtomGroup, int]


class BAI:

    """
        Represents a Bond/Angle/Improper interaction between a certain number of atoms
    """

    def __init__(self, type: BAI_Type, *atoms: AtomIdentifier) -> None:
        """Length of atoms should be 2 for Bond, 3 for Angle, 4 for Improper"""
        self.type = type
        self.atoms: List[AtomIdentifier] = list(atoms)


class PairWise:

    """
        Represents pair interactions between all atoms of two atoms ids.

        header: string defining the interaction style, e.g. "PairIJ Coeffs # hybrid"
        template: a string with empty formatters `{}` for the interaction parameters, e.g. "lj/cut {} {} {}"
        default: a list of default parameters (used to fill out all interactions, since LAMMPS requires them
                                               to all be explicitly defined)
    """

    def __init__(self, header: str, template: str, default: List[Any] | None) -> None:
        """if default is None -> don't insert missing interactions"""
        self.header = header
        self.template = template
        self.default = default
        self.pairs: List[Tuple[AtomType, AtomType, List[Any]]] = []

    def add_interaction(self, atom_type1: AtomType, atom_type2: AtomType, *args: Any) -> PairWise:
        """Add an iteraction. Will sort the indices automatically."""
        ordered = sorted([atom_type1, atom_type2], key=lambda at: at.index)
        self.pairs.append(
            (ordered[0], ordered[1], list(args))
        )

        return self

    def write(self, file, atom_types: List[AtomType]) -> None:
        file.write(self.header)
        for atom_type1, atom_type2, text in self.get_all_interactions(atom_types):
            file.write(f"{atom_type1.index} {atom_type2.index} " + text)
        file.write("\n")

    def get_all_interaction_pairs(self, all_atom_types: List[AtomType]) -> List[Tuple[AtomType, AtomType]]:
        """Returns all possible interactions, whether they are set by the user or not."""
        present_atom_types = set()
        for pair in self.pairs:
            present_atom_types.add(pair[0])
            present_atom_types.add(pair[1])

        all_inters: List[Tuple[AtomType, AtomType]] = []
        all_atom_types = sorted(all_atom_types, key=lambda atom_type: atom_type.index)
        for i in range(len(all_atom_types)):
            for j in range(i, len(all_atom_types)):
                all_inters.append((all_atom_types[i], all_atom_types[j]))

        return all_inters

    def get_all_interactions(self, all_atom_types: List[AtomType]) -> List[Tuple[AtomType, AtomType, str]]:
        """Returns actual interactions to define,
           applying the default where no interaction was specified by the user."""
        all_inters = self.get_all_interaction_pairs(all_atom_types)
        
        def pair_in_inter(interaction: Tuple[AtomType, AtomType]) -> Tuple[AtomType, AtomType, List[Any]] | None:
            for pair in self.pairs:
                if interaction[0] == pair[0] and interaction[1] == pair[1]:
                    return pair
                if interaction[1] == pair[0] and interaction[0] == pair[1]:
                    return pair

            return None

        final_pairs: List[Tuple[AtomType, AtomType, str]] = []

        for inter in all_inters:
            pair = pair_in_inter(inter)
            if pair is None:
                if self.default is not None:
                    final_pairs.append(
                        (inter[0], inter[1], self.template.format(*self.default))
                    )
            else:
                final_pairs.append(
                    (pair[0], pair[1], self.template.format(*pair[2]))
                )

        return final_pairs


class Generator:

    def __init__(self) -> None:
        self.atom_groups: List[AtomGroup] = []
        self.bais: List[BAI] = []
        self.atom_group_map: List[int] = []
        self.pair_interactions: List[PairWise] = []
        self.box_width = None
        self.molecule_override: Dict[AtomIdentifier, int] = dict()

    def set_system_size(self, box_width: float) -> None:
        self.box_width = box_width

    def get_total_atoms(self) -> int:
        return sum(map(lambda atom_group: atom_group.n, self.atom_groups))

    def write_header(self, file) -> None:
        file.write("# LAMMPS data file\n")

    def get_all_atom_types(self) -> List[AtomType]:
        atom_types: Set[AtomType] = set()
        for atom_group in self.atom_groups:
            atom_types.add(atom_group.type)
        return sorted(atom_types, key=lambda atom_type: atom_type.index)

    def get_all_types(self, kind: BAI_Kind) -> List[BAI_Type]:
        bai_types: Set[BAI_Type] = set()
        for bai in filter(lambda bai: bai.type.kind == kind, self.bais):
            bai_types.add(bai.type)
        for atom_group in self.atom_groups:
            if kind == BAI_Kind.BOND and atom_group.polymer_bond_type is not None:
                bai_types.add(atom_group.polymer_bond_type)
            if kind == BAI_Kind.ANGLE and atom_group.polymer_angle_type is not None:
                bai_types.add(atom_group.polymer_angle_type)
        return sorted(bai_types, key=lambda bai_type: bai_type.index)

    def get_bai_dict_by_type(self) -> Dict[BAI_Kind, List[BAI]]:
        bai_by_kind: Dict[BAI_Kind, List[BAI]] = {kind: list() for kind in BAI_Kind}
        for bai in self.bais:
            bai_by_kind[bai.type.kind].append(bai)
        return bai_by_kind

    def write_amounts(self, file) -> None:
        file.write("%s atoms\n"       %self.get_total_atoms())

        length_lookup = {key: len(value) for (key, value) in self.get_bai_dict_by_type().items()}

        totalBonds = length_lookup[BAI_Kind.BOND]
        totalAngles = length_lookup[BAI_Kind.ANGLE]
        totalImpropers = length_lookup[BAI_Kind.IMPROPER]

        for atom_group in self.atom_groups:
            if atom_group.polymer_bond_type is not None:
                totalBonds += len(atom_group.positions) - 1
            if atom_group.polymer_angle_type is not None:
                totalAngles += len(atom_group.positions) - 2

        file.write("%s bonds\n"       %totalBonds)
        file.write("%s angles\n"      %totalAngles)
        file.write("%s impropers\n" %totalImpropers)

        file.write("\n")

    def write_types(self, file) -> None:
        file.write("%s atom types\n"       %len(self.get_all_atom_types()))
        file.write("%s bond types\n"       %len(self.get_all_types(BAI_Kind.BOND)))
        file.write("%s angle types\n"      %len(self.get_all_types(BAI_Kind.ANGLE)))
        file.write("%s improper types\n" %len(self.get_all_types(BAI_Kind.IMPROPER)))

        file.write("\n")

    def write_system_size(self, file) -> None:
        file.write("# System size\n")

        if self.box_width is None:
            raise Exception("box_width was not set")
        half_width = self.box_width / 2.0
        file.write("%s %s xlo xhi\n"   %(-half_width, half_width))
        file.write("%s %s ylo yhi\n"   %(-half_width, half_width))
        file.write("%s %s zlo zhi\n" %(-half_width, half_width))

        file.write("\n")

    def write_masses(self, file) -> None:
        file.write("Masses\n\n")
        for atom_type in self.get_all_atom_types():
            file.write(f"{atom_type.index} {atom_type.mass}\n")

        file.write("\n")

    @staticmethod
    def get_BAI_coeffs_header(kind: BAI_Kind) -> str:
        lookup = {
            BAI_Kind.BOND: "Bond Coeffs # hybrid\n\n",
            BAI_Kind.ANGLE: "Angle Coeffs # hybrid\n\n",
            BAI_Kind.IMPROPER: "Improper Coeffs # harmonic\n\n"
        }
        return lookup[kind]

    def write_BAI_coeffs(self, file) -> None:
        for kind in BAI_Kind:
            file.write(self.get_BAI_coeffs_header(kind))
            for bai_type in self.get_all_types(kind):
                if not bai_type.coefficients:
                    continue
                file.write(f"{bai_type.index} " + bai_type.coefficients)
            file.write("\n")

    def write_pair_interactions(self, file) -> None:
        all_atom_types = self.get_all_atom_types()
        for pair in self.pair_interactions:
            pair.write(file, all_atom_types)

    # def get_atom_index(self, atomId: AtomIdentifier) -> int:
    #     index = self.atom_groups.index(atomId[0])
    #     if len(self.atom_group_map) <= index:
    #         self.atom_group_map = [0] + list(map(lambda atom_group: len(atom_group.positions), self.atom_groups))
    #         self.atom_group_map = list(np.array(self.atom_group_map).cumsum())
    #         self.atom_group_map = [el + 1 for el in self.atom_group_map]
    #     return self.atom_group_map[index] + atomId[1]

    def get_atom_index(self, atomId: AtomIdentifier) -> int:
        if not self.atom_group_map:
            raise Exception("write_atoms must be called first")

        index = self.atom_groups.index(atomId[0])
        if atomId[1] < 0:
            atom_group_length = len(atomId[0].positions)
            atomId = (atomId[0], atomId[1] + atom_group_length)
        return self.atom_group_map[index] + atomId[1]

    def _set_up_atom_group_map(self) -> None:
        index_offset = 1
        for atom_group in self.atom_groups:
            self.atom_group_map.append(index_offset)
            index_offset += len(atom_group.positions)

    def write_atoms(self, file) -> None:
        file.write("Atoms # molecular\n\n")

        self._set_up_atom_group_map()
        molecule_override_ids = {self.get_atom_index(atomId): molId for atomId, molId in self.molecule_override.items()}

        for atom_group in self.atom_groups:
            for j, position in enumerate(atom_group.positions):
                atom_id = self.get_atom_index((atom_group, j))
                try:
                    mol_id = molecule_override_ids[atom_id]
                except KeyError:
                    mol_id = atom_group.molecule_index
                file.write(f"%s %s {atom_group.type.index} %s %s %s\n" %(atom_id, mol_id, *position) )

        file.write("\n")

    @staticmethod
    def get_BAI_header(kind: BAI_Kind) -> str:
        lookup = {
            BAI_Kind.BOND: "Bonds\n\n",
            BAI_Kind.ANGLE: "Angles\n\n",
            BAI_Kind.IMPROPER: "Impropers\n\n"
        }
        return lookup[kind]

    def write_bai(self, file) -> None:
        for kind in BAI_Kind:
            file.write(self.get_BAI_header(kind))

            global_index = 1

            for atom_group in self.atom_groups:
                if kind == BAI_Kind.BOND and atom_group.polymer_bond_type is not None:
                    for j in range(len(atom_group.positions) - 1):
                        file.write(f"%s {atom_group.polymer_bond_type.index} %s %s\n" %(global_index, self.get_atom_index((atom_group, j)), self.get_atom_index((atom_group, j + 1))) )
                        global_index += 1
                if kind == BAI_Kind.ANGLE and atom_group.polymer_angle_type is not None:
                    for j in range(len(atom_group.positions) - 2):
                        file.write(f"%s {atom_group.polymer_angle_type.index} %s %s %s\n" %(global_index, self.get_atom_index((atom_group, j)), self.get_atom_index((atom_group, j + 1)), self.get_atom_index((atom_group, j + 2))) )
                        global_index += 1

            length = length_lookup[kind]
            for bai in filter(lambda bai: bai.type.kind == kind, self.bais):    
                file.write(f"%s {bai.type.index} " %(global_index) )
                formatter = ("%s " * length)[:-1] + "\n"
                file.write(formatter %(*(self.get_atom_index(bai.atoms[i]) for i in range(length)),))
                global_index += 1

            file.write("\n")

    def write_coeffs(self, file) -> None:
        self.write_header(file)
        self.write_types(file)
        file.write("\n")
        self.write_masses(file)
        self.write_BAI_coeffs(file)
        self.write_pair_interactions(file)

    def write_positions_and_bonds(self, file) -> None:
        self.write_header(file)
        self.write_amounts(file)
        self.write_types(file)
        self.write_system_size(file)
        file.write("\n")
        self.write_atoms(file)
        self.write_bai(file)

    def write_full(self, file) -> None:
        self.write_header(file)
        self.write_amounts(file)
        self.write_types(file)
        self.write_system_size(file)
        file.write("\n")
        self.write_masses(file)
        self.write_BAI_coeffs(file)
        self.write_pair_interactions(file)
        self.write_atoms(file)
        self.write_bai(file)

    @staticmethod
    def get_script_bai_command_name(pair_or_BAI: BAI_Kind | None) -> str:
        name_dict = {
            None: "pair_coeff",
            BAI_Kind.BOND: "bond_coeff",
            BAI_Kind.ANGLE: "angle_coeff",
            BAI_Kind.IMPROPER: "improper_coeff",
        }
        return name_dict[pair_or_BAI]

    @dataclass
    class DynamicCoeffs:
        pair_or_BAI: None | BAI_Kind
        format_string: str
        args: List[Any]

    def write_script_bai_coeffs(self, file, coeffs: DynamicCoeffs) -> None:
        cmd_name = self.get_script_bai_command_name(coeffs.pair_or_BAI)
        # parse args:
        # if pair type -> get atom indices (two AtomType instances)
        # else (bai) -> get BAI_Type index
        # in both cases -> assume arguments have .index field
        format_args = [arg.index for arg in coeffs.args]
        formatted_string = coeffs.format_string.format(*format_args)
        file.write(cmd_name + " " + formatted_string)


def test_simple_atoms():
    positions = np.zeros(shape=(100, 3))
    gen = Generator()
    gen.atom_groups.append(AtomGroup(positions, AtomType(), 1))
    gen.set_system_size(10)
    with open("test.gen", 'w', encoding='utf-8') as file:
        gen.write_full(file)


def test_simple_atoms_polymer():
    positions = np.zeros(shape=(100, 3))
    gen = Generator()
    gen.atom_groups.append(AtomGroup(positions, AtomType(), 1, polymer_bond_type=BAI_Type(BAI_Kind.BOND)))
    gen.set_system_size(10)
    with open("test.gen", 'w', encoding='utf-8') as file:
        gen.write_full(file)


def test_with_bonds():
    positions = np.zeros(shape=(25, 3))

    gen = Generator()
    gen.set_system_size(10)

    bt1 = BAI_Type(BAI_Kind.BOND)
    bt2 = BAI_Type(BAI_Kind.BOND)

    group1 = AtomGroup(positions, AtomType(), 1, polymer_bond_type=bt2)
    gen.atom_groups.append(group1)

    group2 = AtomGroup(np.copy(positions), AtomType(), 3)
    gen.atom_groups.append(group2)

    gen.bais.append(
        BAI(
            bt1,
            (group1, 1),
            (group2, 0)
        )
    )

    gen.bais.append(
        BAI(
            BAI_Type(BAI_Kind.BOND),
            (group1, 5),
            (group1, 6)
        )
    )

    gen.bais.append(
        BAI(
            bt1,
            (group1, 9),
            (group1, 16)
        )
    )

    with open("test.gen", 'w', encoding='utf-8') as file:
        gen.write_full(file)


def test_with_pairs():
    positions = np.zeros(shape=(25, 3))

    gen = Generator()
    gen.set_system_size(10)

    at1 = AtomType()
    group1 = AtomGroup(positions, at1, 1)
    gen.atom_groups.append(group1)

    group2 = AtomGroup(np.copy(positions), AtomType(), 3)
    gen.atom_groups.append(group2)

    pairwise = PairWise("PairIJ Coeffs # hybrid\n", "lj/cut {} {} {}\n", [0, 0, 0])
    pairwise.add_interaction(at1, at1, 1, 2, 3)

    gen.pair_interactions.append(pairwise)

    with open("test.gen", 'w', encoding='utf-8') as file:
        gen.write_full(file)


def all_tests():
    # test_simple_atoms()
    # test_simple_atoms_polymer()
    # test_with_bonds()
    test_with_pairs()


def main():
    all_tests()

if __name__ == "__main__":
    main()
