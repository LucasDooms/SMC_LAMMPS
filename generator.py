from __future__ import annotations
from typing import List, Tuple, Set, Dict
from enum import Enum
import numpy as np


class AtomType:

    index = 1

    def __init__(self, mass: float = 1.0) -> None:
        self.index = self.__class__.index
        self.__class__.index += 1
        self.mass = mass


class BAI_Kind(Enum):
    BOND = 1
    ANGLE = 2
    IMPROPER = 3


length_lookup = dict({
    BAI_Kind.BOND: 2,
    BAI_Kind.ANGLE: 3,
    BAI_Kind.IMPROPER: 4
})


class BAI_Type:

    indices = {kind: 1 for kind in BAI_Kind}

    def __init__(self, kind: BAI_Kind, coefficients: str = "") -> None:
        self.index = self.indices[kind]
        self.indices[kind] += 1
        self.kind = kind
        self.coefficients = coefficients


class AtomGroup:

    """
        positions: list of 3d positions of (n) atoms
        mass (float): mass of each atom
        polymer_bond_type : if None -> no bonds, otherwise all atoms will from bonds as a polymer
    """

    def __init__(self, positions: np.ndarray[np.ndarray[float]],
                 atom_type: AtomType, polymer_bond_type: BondType | None = None) -> None:
        self.n = len(positions)
        self.positions = positions
        self.type = atom_type
        self.polymer_bond_type = polymer_bond_type


AtomIdentifier = Tuple[AtomGroup, int]


class BAI:

    def __init__(self, type: BAI_Type, *atoms) -> None:
        """Length of atoms should be 2 for Bond, 3 for Angle, 4 for Improper"""
        self.type = type
        self.atoms: List[AtomIdentifier] = atoms


class PairWise:

    def __init__(self, header: str) -> None:
        self.header = header
        self.pairs: List[Tuple[AtomType, AtomType, str]] = []

    def add_interaction(self, atom_type1, atom_type2, interaction: str) -> PairWise:
        self.pairs.append(
            (atom_type1, atom_type2, interaction)
        )

        return self


class Generator:

    def __init__(self) -> None:
        self.atom_groups: List[AtomGroup] = []
        self.bais: List[BAI] = []
        self.atom_group_map: List[int] = []
        self.pair_interactions: List[PairWise] = []
    
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
        if kind == BAI_Kind.BOND:
            for atom_group in self.atom_groups:
                if atom_group.polymer_bond_type is None:
                    continue
                bai_types.add(atom_group.polymer_bond_type)
        return sorted(bai_types, key=lambda bai_type: bai_type.index)

    def get_bai_dict_by_type(self) -> Dict[(BAI_Kind, List[BAI])]:
        bai_by_kind: Dict[(BAI_Kind, List[BAI])] = {kind: list() for kind in BAI_Kind}
        for bai in self.bais:
            bai_by_kind[bai.type.kind].append(bai)
        return bai_by_kind

    def write_amounts(self, file) -> None:
        file.write("%s atoms\n"       %self.get_total_atoms())

        length_lookup = {key: len(value) for (key, value) in self.get_bai_dict_by_type().items()}

        totalBonds = length_lookup[BAI_Kind.BOND]
        for atom_group in self.atom_groups:
            if atom_group.polymer_bond_type is not None:
                totalBonds += len(atom_group.positions) - 1

        totalAngles = length_lookup[BAI_Kind.ANGLE]
        totalImpropers = length_lookup[BAI_Kind.IMPROPER]

        file.write("%s bonds\n"       %totalBonds)
        file.write("%s angles\n"      %totalAngles)
        file.write("%s impropers\n\n" %totalImpropers)

    def write_types(self, file) -> None:
        file.write("%s atom types\n"       %len(self.get_all_atom_types()))
        file.write("%s bond types\n"       %len(self.get_all_types(BAI_Kind.BOND)))
        file.write("%s angle types\n"      %len(self.get_all_types(BAI_Kind.ANGLE)))
        file.write("%s improper types\n\n" %len(self.get_all_types(BAI_Kind.IMPROPER)))

    def write_masses(self, file) -> None:
        file.write("Masses\n\n")
        global_index = 1
        for atom_type in self.get_all_atom_types():
            file.write(f"{global_index} {atom_type.mass}\n")
            global_index += 1

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
            global_index = 1
            for bai_type in self.get_all_types(kind):
                if not bai_type.coefficients:
                    continue
                file.write(f"{global_index} " + bai_type.coefficients)
                global_index += 1

    def write_pair_interactions(self, file) -> None:
        for pair in self.pair_interactions:
            file.write(pair.header)
            for atom_type1, atom_type2, text in pair.pairs:
                file.write(f"{atom_type1.index}, {atom_type2.index} " + text)

    def get_atom_index(self, atomId: AtomIdentifier) -> int:
        index = self.atom_groups.index(atomId[0])
        if len(self.atom_group_map) <= index:
            self.atom_group_map = [0] + list(map(lambda atom_group: len(atom_group.positions), self.atom_groups))
            self.atom_group_map = [el + 1 for el in self.atom_group_map]
        return self.atom_group_map[index] + atomId[1]

    def write_atoms(self, file) -> None:
        file.write("\nAtoms # molecular\n\n")

        index_offset = 1
        for i, atom_group in enumerate(self.atom_groups):
            for j, position in enumerate(atom_group.positions):
                file.write(f"%s %s {i + 1} %s %s %s\n" %(j + index_offset, -1, *position) )
            index_offset += len(atom_group.positions)
    
    @staticmethod
    def get_BAI_header(kind: BAI_Kind) -> str:
        lookup = {
            BAI_Kind.BOND: "\nBonds\n\n",
            BAI_Kind.ANGLE: "\nAngles\n\n",
            BAI_Kind.IMPROPER: "\nImpropers\n\n"
        }
        return lookup[kind]

    def write_bai(self, file) -> None:
        for kind in BAI_Kind:
            file.write(self.get_BAI_header(kind))

            global_index = 1

            if kind == BAI_Kind.BOND:
                for atom_group in self.atom_groups:
                    if atom_group.polymer_bond_type is None:
                        continue
                    for j in range(len(atom_group.positions) - 1):
                        file.write(f"%s {atom_group.polymer_bond_type.index} %s %s\n" %(global_index, j + 1, j + 2) )
                        global_index += 1

            for bai in filter(lambda bai: bai.type.kind == kind, self.bais):
                length = length_lookup[kind]
                file.write(f"%s {bai.type.index} " %(global_index) )
                formatter = ("%s " * length)[:-1] + "\n"
                file.write(formatter %(*(self.get_atom_index(bai.atoms[i]) for i in range(length)),))
                global_index += 1

    def write(self, file) -> None:
        self.write_header(file)
        self.write_amounts(file)
        file.write("\n")
        self.write_types(file)
        file.write("\n")
        self.write_masses(file)
        self.write_BAI_coeffs(file)
        self.write_pair_interactions(file)
        self.write_atoms(file)
        self.write_bai(file)


def test_simple_atoms():
    positions = np.zeros(shape=(100, 3))
    gen = Generator()
    gen.atom_groups.append(AtomGroup(positions, AtomType()))
    with open("test.gen", 'w') as file:
        gen.write(file)


def test_simple_atoms_polymer():
    positions = np.zeros(shape=(100, 3))
    gen = Generator()
    gen.atom_groups.append(AtomGroup(positions, AtomType(), polymer_bond_type=BAI_Type(BAI_Kind.BOND)))
    with open("test.gen", 'w') as file:
        gen.write(file)


def test_with_bonds():
    positions = np.zeros(shape=(25, 3))

    gen = Generator()

    bt1 = BAI_Type(BAI_Kind.BOND)
    bt2 = BAI_Type(BAI_Kind.BOND)

    group1 = AtomGroup(positions, AtomType(), polymer_bond_type=bt2)
    gen.atom_groups.append(group1)

    group2 = AtomGroup(np.copy(positions), AtomType())
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

    with open("test.gen", 'w') as file:
        gen.write(file)

    
def all_tests():
    # test_simple_atoms()
    # test_simple_atoms_polymer()
    test_with_bonds()


def main():
    all_tests()

if __name__ == "__main__":
    main()
