# Copyright (c) 2024-2026 Lucas Dooms

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

from __future__ import annotations

from enum import Enum
from typing import Any, TextIO, TypeAlias

import numpy as np
import numpy.typing as npt

from smc_lammps.console import warn


class AtomType:
    __index = 0

    class UnusedIndex(AttributeError):
        pass

    def __init__(self, mass: float = 1.0, unused: bool = False) -> None:
        self._index = None
        self.mass = mass
        self.unused = unused

    @classmethod
    def _get_next(cls) -> int:
        cls.__index += 1
        return cls.__index

    @property
    def index(self) -> int:
        if self.unused:
            raise self.UnusedIndex(
                "This AtomType is marked as unused, cannot obtain a LAMMPS index!"
            )
        if self._index is None:
            self._index = self._get_next()
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

    @classmethod
    def length_lookup(cls, kind: BAI_Kind) -> int:
        """Returns the number of arguments (atom ids) needed to define a BAI interaction."""
        values = {BAI_Kind.BOND: 2, BAI_Kind.ANGLE: 3, BAI_Kind.IMPROPER: 4}
        return values[kind]


class BAI_Type:
    indices = {kind: 1 for kind in BAI_Kind}

    def __init__(self, kind: BAI_Kind, style: str, coefficients: str = "") -> None:
        """coefficients will be printed after the index in a datafile
        e.g. 3 harmonic 2 5
        (where 3 is the index and the rest is the coefficients string)"""
        self._index = None
        self.kind = kind
        self.style = style
        self.coefficients = coefficients

    @property
    def index(self) -> int:
        if self._index is None:
            self._index = self.indices[self.kind]
            self.indices[self.kind] += 1
        return self._index

    def get_string(self, omit_style=False) -> str:
        style = f" {self.style}" if not omit_style else ""
        return f"{self.index}{style} {self.coefficients}"


COORD_TYPE: TypeAlias = np.float32
Nx3Array: TypeAlias = npt.NDArray[COORD_TYPE]
"""An (N, 3) array of positions"""


class AtomGroup:
    """
    positions: numpy array with shape (n, 3) of 3d positions of (n) atoms [[x, y, z], ...]
    atom_type: type of the atoms
    molecule_index (int): the molecule index which is needed for atom_style molecule
    polymer_bond_type: if None -> no bonds, otherwise all atoms will form bonds as a polymer
    """

    def __init__(
        self,
        positions: Nx3Array,
        atom_type: AtomType,
        molecule_index: int,
        polymer_bond_type: BAI_Type | None = None,
        polymer_angle_type: BAI_Type | None = None,
        charge: float = 0.0,
    ) -> None:
        self.positions = positions
        self.type = atom_type
        self.molecule_index = molecule_index
        if polymer_bond_type is not None:
            assert polymer_bond_type.kind == BAI_Kind.BOND
        self.polymer_bond_type = polymer_bond_type
        if polymer_angle_type is not None:
            assert polymer_angle_type.kind == BAI_Kind.ANGLE
        self.polymer_angle_type = polymer_angle_type
        self.charge = charge

    @property
    def n(self) -> int:
        return len(self.positions)


AtomIdentifier: TypeAlias = tuple[AtomGroup, int]
"A unique identifier of an atom in a group"


class BAI:
    """
    Represents a Bond/Angle/Improper interaction between a certain number of atoms
    """

    def __init__(self, type_: BAI_Type, *atoms: AtomIdentifier) -> None:
        """Length of atoms should be 2 for Bond, 3 for Angle, 4 for Improper"""
        self.type = type_
        self.atoms: list[AtomIdentifier] = list(atoms)


class PairWise:
    """
    Represents pair interactions between all atoms of two atoms ids.

    header: string defining the interaction style, e.g. "PairIJ Coeffs # hybrid"
    template: a string with empty formatters `{}` for the interaction parameters, e.g. "lj/cut {} {} {}"
    default: a list of default parameters
    (used to fill out all interactions, since LAMMPS requires them to all be explicitly defined)
    """

    def __init__(self, header: str, template: str, default: list[Any] | None) -> None:
        """if default is None -> don't insert missing interactions"""
        self.header = header
        self.template = template
        self.default = default
        self.pairs: list[tuple[AtomType, AtomType, list[Any]]] = []

    def add_interaction(
        self, atom_type1: AtomType, atom_type2: AtomType, *args: Any, **kwargs: bool
    ) -> PairWise:
        """Add an iteraction. Will sort the indices automatically."""
        try:
            ordered = sorted([atom_type1, atom_type2], key=lambda at: at.index)
        except AtomType.UnusedIndex as e:
            if kwargs.get("allow_unused", False):
                # simply return without changing the self.pairs list
                return self
            raise e

        self.pairs.append((ordered[0], ordered[1], list(args)))

        return self

    def write(self, file: TextIO, atom_types: list[AtomType]) -> None:
        """Write the Pair Coeffs header and all pair interactions to a file.
        e.g.
        PairIJ Coeffs
        1 8 lj/cut 0.0 0.0 0.0
        """
        file.write(self.header)
        for atom_type1, atom_type2, text in self.get_all_interactions(atom_types):
            file.write(f"{atom_type1.index} {atom_type2.index} " + text)
        file.write("\n")

    def get_all_interaction_pairs(
        self, all_atom_types: list[AtomType]
    ) -> list[tuple[AtomType, AtomType]]:
        """Returns all possible interactions, whether they are set by the user or not."""
        present_atom_types = set()
        for pair in self.pairs:
            present_atom_types.add(pair[0])
            present_atom_types.add(pair[1])

        all_inters: list[tuple[AtomType, AtomType]] = []
        all_atom_types = sorted(all_atom_types, key=lambda atom_type: atom_type.index)
        for i in range(len(all_atom_types)):
            for j in range(i, len(all_atom_types)):
                all_inters.append((all_atom_types[i], all_atom_types[j]))

        return all_inters

    def pair_in_inter(
        self,
        interaction: tuple[AtomType, AtomType],
    ) -> tuple[AtomType, AtomType, list[Any]] | None:
        for pair in self.pairs:
            if interaction[0] == pair[0] and interaction[1] == pair[1]:
                return pair
            if interaction[1] == pair[0] and interaction[0] == pair[1]:
                return pair

        return None

    def get_all_interactions(
        self, all_atom_types: list[AtomType]
    ) -> list[tuple[AtomType, AtomType, str]]:
        """Returns actual interactions to define,
        applying the default where no interaction was specified by the user."""
        all_inters = self.get_all_interaction_pairs(all_atom_types)

        final_pairs: list[tuple[AtomType, AtomType, str]] = []

        for inter in all_inters:
            pair = self.pair_in_inter(inter)
            if pair is None:
                if self.default is not None:
                    final_pairs.append((inter[0], inter[1], self.template.format(*self.default)))
            else:
                final_pairs.append((pair[0], pair[1], self.template.format(*pair[2])))

        return final_pairs


def write_if_non_zero(file: TextIO, fmt_string: str, amount: int):
    if amount != 0:
        file.write(fmt_string.format(amount))


class Generator:
    def __init__(self) -> None:
        self._atom_groups: list[AtomGroup] = []
        self.bais: list[BAI] = []
        self.atom_group_map: list[int] = []
        self.pair_interactions: list[PairWise] = []
        self.box_width = None
        self.molecule_override: dict[AtomIdentifier, int] = {}
        self.charge_override: dict[AtomIdentifier, float] = {}
        self.use_charges = False
        self.hybrid_styles = {
            BAI_Kind.BOND: "hybrid",
            BAI_Kind.ANGLE: "hybrid",
            BAI_Kind.IMPROPER: "hybrid",
        }
        self.random_shift = lambda: np.array([0.0, 0.0, 0.0])
        """Function that returns a random shift vector.
        This is useful to avoid exact overlap, which causes LAMMPS to crash during kspace calculations (e.g. with pair_style coul)."""

    def add_atom_groups(self, *args: AtomGroup) -> None:
        """Add new atom groups, atom groups with empty positions are ignored."""
        for grp in args:
            if grp.positions.size == 0:
                continue

            self._atom_groups.append(grp)

    def move_all_atoms(self, shift: Nx3Array) -> None:
        """Move all atoms in a certain direction.
        Useful to place the system at the center of the simulation box."""
        for grp in self._atom_groups:
            grp.positions += shift

    def set_system_size(self, box_width: float) -> None:
        """Set the box size of the simulation."""
        self.box_width = box_width

    def get_total_atoms(self) -> int:
        """Get the total number of atoms across all groups."""
        return sum(map(lambda atom_group: atom_group.n, self._atom_groups))

    def write_header(self, file: TextIO) -> None:
        """Write the top header to a file.
        Note: LAMMPS always ignores the first line of a data file,
        which should be this header."""
        file.write("# LAMMPS data file\n")

    def get_all_atom_types(self) -> list[AtomType]:
        """Get a list of all atom types across all atom groups.
        The returned list is sorted based on the AtomType index."""
        atom_types: set[AtomType] = set()
        for atom_group in self._atom_groups:
            atom_types.add(atom_group.type)
        return sorted(atom_types, key=lambda atom_type: atom_type.index)

    def get_all_types(self, kind: BAI_Kind) -> list[BAI_Type]:
        """Get a list of all Bond/Angle/Improper types across all atom groups.
        The returned list is sorted based on the BAI_Type index."""
        bai_types: set[BAI_Type] = set()
        for bai in filter(lambda bai: bai.type.kind == kind, self.bais):
            bai_types.add(bai.type)
        for atom_group in self._atom_groups:
            if kind == BAI_Kind.BOND and atom_group.polymer_bond_type is not None:
                bai_types.add(atom_group.polymer_bond_type)
            if kind == BAI_Kind.ANGLE and atom_group.polymer_angle_type is not None:
                bai_types.add(atom_group.polymer_angle_type)
        return sorted(bai_types, key=lambda bai_type: bai_type.index)

    def get_bai_dict_by_type(self) -> dict[BAI_Kind, list[BAI]]:
        """Get a dictionary mapping the Bond/Angle/Improper kind
        to a list of all BAIs which have that type."""
        bai_by_kind: dict[BAI_Kind, list[BAI]] = {kind: [] for kind in BAI_Kind}
        for bai in self.bais:
            bai_by_kind[bai.type.kind].append(bai)
        return bai_by_kind

    def get_amounts(self) -> tuple[int, int, int]:
        """Return the total amount of bonds, angles, and impropers."""
        length_lookup = {key: len(value) for (key, value) in self.get_bai_dict_by_type().items()}

        total_bonds = length_lookup[BAI_Kind.BOND]
        total_angles = length_lookup[BAI_Kind.ANGLE]
        total_impropers = length_lookup[BAI_Kind.IMPROPER]

        for atom_group in self._atom_groups:
            if atom_group.polymer_bond_type is not None:
                total_bonds += max(0, len(atom_group.positions) - 1)
            if atom_group.polymer_angle_type is not None:
                total_angles += max(0, len(atom_group.positions) - 2)

        return total_bonds, total_angles, total_impropers

    def write_amounts(self, file: TextIO) -> None:
        """Write the amount of atoms, bonds, angles, and impropers to a file."""
        file.write(f"{self.get_total_atoms()} atoms\n")

        total_bonds, total_angles, total_impropers = self.get_amounts()

        write_if_non_zero(file, "{} bonds\n", total_bonds)
        write_if_non_zero(file, "{} angles\n", total_angles)
        write_if_non_zero(file, "{} impropers\n", total_impropers)

        file.write("\n")

    def write_types(self, file: TextIO) -> None:
        """Write the amount of atom types, bond types, angle types, and improper types to a file."""

        write_if_non_zero(file, "{} atom types\n", len(self.get_all_atom_types()))
        write_if_non_zero(file, "{} bond types\n", len(self.get_all_types(BAI_Kind.BOND)))
        write_if_non_zero(file, "{} angle types\n", len(self.get_all_types(BAI_Kind.ANGLE)))
        write_if_non_zero(file, "{} improper types\n", len(self.get_all_types(BAI_Kind.IMPROPER)))

        file.write("\n")

    def write_system_size(self, file: TextIO) -> None:
        """Write the system size to a file."""
        file.write("# System size\n")

        if self.box_width is None:
            raise TypeError("box_width was not set")
        half_width = self.box_width / 2.0
        lohi = (-half_width, half_width)
        file.write(f"{lohi[0]} {lohi[1]} xlo xhi\n")
        file.write(f"{lohi[0]} {lohi[1]} ylo yhi\n")
        file.write(f"{lohi[0]} {lohi[1]} zlo zhi\n")

        file.write("\n")

    def write_masses(self, file: TextIO) -> None:
        """Write the masses of all atom types to a file."""
        file.write("Masses\n\n")
        for atom_type in self.get_all_atom_types():
            file.write(f"{atom_type.index} {atom_type.mass}\n")

        file.write("\n")

    def get_all_BAI_styles(self) -> dict[BAI_Kind, list[str]]:
        """Return a list of unique styles for each BAI kind."""

        def get_unique_styles(bai_types: list[BAI_Type]) -> list[str]:
            return list(set(t.style for t in bai_types))

        return {k: get_unique_styles(self.get_all_types(k)) for k in BAI_Kind}

    @staticmethod
    def get_BAI_style_command_name(kind: BAI_Kind) -> str:
        lookup = {
            BAI_Kind.BOND: "bond_style",
            BAI_Kind.ANGLE: "angle_style",
            BAI_Kind.IMPROPER: "improper_style",
        }
        return lookup[kind]

    def get_BAI_styles_command(self) -> str:
        """Return the command to define the BAI styles in a LAMMPS script."""
        all_styles = self.get_all_BAI_styles()

        def extract_command(k: BAI_Kind) -> str:
            styles = all_styles[k]
            if len(styles) == 1:
                return styles[0]
            styles_string = " ".join(styles)
            return f"{self.hybrid_styles[k]} {styles_string}"

        strings = [f"{self.get_BAI_style_command_name(k)} {extract_command(k)}\n" for k in BAI_Kind]
        return "".join(strings)

    def get_hybrid_or_single_style(self) -> dict[BAI_Kind, str]:
        """Return the style needed for the BAI Coeffs header."""

        def extract_style(k: BAI_Kind, styles: list[str]) -> str:
            if len(styles) == 1:
                return styles[0]
            return self.hybrid_styles[k]

        all_styles = self.get_all_BAI_styles()
        return {k: extract_style(k, v) for k, v in all_styles.items()}

    def get_BAI_coeffs_header(self, kind: BAI_Kind) -> str:
        """Get the header string corresponding to a Bond/Angle/Improper kind."""
        lookup = {
            BAI_Kind.BOND: "Bond Coeffs # {}\n\n",
            BAI_Kind.ANGLE: "Angle Coeffs # {}\n\n",
            BAI_Kind.IMPROPER: "Improper Coeffs # {}\n\n",
        }
        style_strings = self.get_hybrid_or_single_style()
        lookup = {k: v.format(style_strings[k]) for k, v in lookup.items()}
        return lookup[kind]

    def write_BAI_coeffs(self, file: TextIO) -> None:
        """Write the Bond/Angle/Improper coefficients for each BAI kind to a file."""
        total_bonds, total_angles, total_impropers = self.get_amounts()
        lookup = {
            BAI_Kind.BOND: total_bonds,
            BAI_Kind.ANGLE: total_angles,
            BAI_Kind.IMPROPER: total_impropers,
        }
        for kind in BAI_Kind:
            if lookup[kind] == 0:
                # do not write anything if there are no BAIs of this kind
                continue

            file.write(self.get_BAI_coeffs_header(kind))
            all_types = self.get_all_types(kind)
            for bai_type in all_types:
                if not bai_type.coefficients:
                    continue
                omit_style = len(set(t.style for t in all_types)) == 1
                file.write(bai_type.get_string(omit_style))
            file.write("\n")

    def write_pair_interactions(self, file: TextIO) -> None:
        """Write the Pair Coeffs header(s) and corresponding pair interactions to a file."""
        all_atom_types = self.get_all_atom_types()
        for pair in self.pair_interactions:
            pair.write(file, all_atom_types)

    def get_atom_index(self, atom_id: AtomIdentifier) -> int:
        """Get the absolute LAMMPS index for an atom.
        You must call write_atoms before using this method."""
        if not self.atom_group_map:
            raise AttributeError("write_atoms must be called first")

        index = self._atom_groups.index(atom_id[0])
        if atom_id[1] < 0:
            atom_group_length = len(atom_id[0].positions)
            atom_id = (atom_id[0], atom_id[1] + atom_group_length)
        return self.atom_group_map[index] + atom_id[1]

    def _set_up_atom_group_map(self) -> None:
        index_offset = 1
        for atom_group in self._atom_groups:
            self.atom_group_map.append(index_offset)
            index_offset += len(atom_group.positions)

    def get_atom_style(self) -> str:
        """Return the atom_style for LAMMPS (based on use_charges)."""
        if self.use_charges:
            return "full"
        else:
            return "molecular"

    def get_atoms_header(self) -> str:
        return f"Atoms # {self.get_atom_style()}\n\n"

    def get_atom_style_command(self) -> str:
        return f"atom_style {self.get_atom_style()}\n"

    def check_charges(self) -> None:
        """Checks for discrepancies between the use_charges flag and the actual atom charges."""
        nonzero_charges = [grp.charge != 0.0 for grp in self._atom_groups]

        if self.use_charges:
            if not any(nonzero_charges):
                warn(
                    "Charges are enabled, but all atoms have zero charge, this may affect performance!\n"
                    "Set use_charges=False to disable charges."
                )
        else:
            if any(nonzero_charges):
                warn(
                    "Charges are disabled, but some atoms have nonzero charge!\n"
                    "Set use_charges=True to enable charges."
                )

    def write_atoms(self, file: TextIO) -> None:
        """Write the Atoms header and all atom positions to a file."""
        self.check_charges()
        file.write(self.get_atoms_header())

        self._set_up_atom_group_map()
        molecule_override_ids = {
            self.get_atom_index(atom_id): mol_id
            for atom_id, mol_id in self.molecule_override.items()
        }
        charge_override_values = {
            self.get_atom_index(atom_id): charge for atom_id, charge in self.charge_override.items()
        }

        for atom_group in self._atom_groups:
            for j, position in enumerate(atom_group.positions):
                atom_id = self.get_atom_index((atom_group, j))
                try:
                    mol_id = molecule_override_ids[atom_id]
                except KeyError:
                    mol_id = atom_group.molecule_index

                ids = f"{atom_id} {mol_id} {atom_group.type.index}"

                if self.use_charges:
                    try:
                        charge = charge_override_values[atom_id]
                    except KeyError:
                        charge = atom_group.charge
                    ids += f" {charge}"

                position += self.random_shift()
                file.write(ids + f" {position[0]} {position[1]} {position[2]}\n")

        file.write("\n")

    @staticmethod
    def get_BAI_header(kind: BAI_Kind) -> str:
        lookup = {
            BAI_Kind.BOND: "Bonds\n\n",
            BAI_Kind.ANGLE: "Angles\n\n",
            BAI_Kind.IMPROPER: "Impropers\n\n",
        }
        return lookup[kind]

    def write_bai(self, file: TextIO) -> None:
        """Write the Bond/Angle/Improper headers and all corresponding BAI interactions to a file."""
        total_bonds, total_angles, total_impropers = self.get_amounts()
        lookup = {
            BAI_Kind.BOND: total_bonds,
            BAI_Kind.ANGLE: total_angles,
            BAI_Kind.IMPROPER: total_impropers,
        }

        for kind in BAI_Kind:
            if lookup[kind] == 0:
                # do not write anything if there are no BAIs of this kind
                continue

            file.write(self.get_BAI_header(kind))

            global_index = 1

            for atom_group in self._atom_groups:
                if kind == BAI_Kind.BOND and atom_group.polymer_bond_type is not None:
                    for j in range(len(atom_group.positions) - 1):
                        file.write(
                            f"{global_index} {atom_group.polymer_bond_type.index} "
                            f"{self.get_atom_index((atom_group, j))} "
                            f"{self.get_atom_index((atom_group, j + 1))}"
                            "\n"
                        )
                        global_index += 1

                if kind == BAI_Kind.ANGLE and atom_group.polymer_angle_type is not None:
                    for j in range(len(atom_group.positions) - 2):
                        file.write(
                            f"{global_index} {atom_group.polymer_angle_type.index} "
                            f"{self.get_atom_index((atom_group, j))} "
                            f"{self.get_atom_index((atom_group, j + 1))} "
                            f"{self.get_atom_index((atom_group, j + 2))} "
                            "\n"
                        )
                        global_index += 1

            length = BAI_Kind.length_lookup(kind)
            for bai in [bai for bai in self.bais if bai.type.kind == kind]:
                file.write(f"{global_index} {bai.type.index} ")
                formatter = ("{} " * length)[:-1] + "\n"
                file.write(
                    formatter.format(*(self.get_atom_index(bai.atoms[i]) for i in range(length)))
                )
                global_index += 1

            file.write("\n")

    def write_coeffs(self, file: TextIO) -> None:
        """Write the coefficient information to a file.
        Useful when restarting a simulation."""
        self.write_header(file)
        self.write_types(file)
        file.write("\n")
        self.write_masses(file)
        self.write_BAI_coeffs(file)
        self.write_pair_interactions(file)

    def write_positions_and_bonds(self, file: TextIO) -> None:
        """Write the positions and bonds to a file."""
        self.write_header(file)
        self.write_amounts(file)
        self.write_types(file)
        self.write_system_size(file)
        file.write("\n")
        self.write_atoms(file)
        self.write_bai(file)

    def write_full(self, file: TextIO) -> None:
        """Write a full LAMMPS data file."""
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

    class DynamicCoeffs:
        """Handles coefficients dynamically set within a LAMMPS script
        using pair_coeff, bond_coeff, angle_coeff, and improper_coeff commands."""

        def __init__(self, coeff_string: str, args: BAI_Type | list[AtomType]) -> None:
            self.coeff_string = coeff_string
            self.args = args

        @staticmethod
        def get_script_bai_command_name(pair_or_BAI: BAI_Kind | None) -> str:
            """Get the command name for a pair / BAI interaction.
            Used to redefine coefficients within a LAMMPS script."""
            name_dict = {
                None: "pair_coeff",
                BAI_Kind.BOND: "bond_coeff",
                BAI_Kind.ANGLE: "angle_coeff",
                BAI_Kind.IMPROPER: "improper_coeff",
            }
            return name_dict[pair_or_BAI]

        @classmethod
        def create_from_pairwise(
            cls, pairwise: PairWise, type1: AtomType, type2: AtomType, values: list[Any] | None
        ) -> Generator.DynamicCoeffs:
            if values is not None:
                coeff_string = pairwise.template.format(*values)
            else:
                # search for values in PairWise
                # NOTE: we are not using the inferred get_all_interactions values,
                # only the explicitly user defined values!
                pair = pairwise.pair_in_inter((type1, type2))
                if pair is None:
                    raise RuntimeError(
                        "This pair interaction has no assigned values (yet).\n"
                        "To resolve this, do one of the following\n"
                        "\t- make sure the interaction has been added via PairWise.add_interaction\n"
                        "\t- define the values explicitly in create_from_pairwise (don't pass None)\n"
                    )
                coeff_string = pairwise.template.format(*pair[2])

            return cls(coeff_string, [type1, type2])

        def write_script_bai_coeffs(self, file: TextIO) -> None:
            """Write a LAMMPS command for a pair / BAI interaction to a file."""
            if isinstance(self.args, BAI_Type):
                cmd_name = self.get_script_bai_command_name(self.args.kind)
                format_args = [str(self.args.index)]
            else:  # pair interaction
                cmd_name = self.get_script_bai_command_name(None)
                format_args = [str(arg.index) for arg in self.args]

            formatted_string = " ".join(format_args) + " " + self.coeff_string
            file.write(cmd_name + " " + formatted_string)


def test_simple_atoms():
    positions = np.zeros(shape=(100, 3), dtype=np.float32)
    gen = Generator()
    gen.add_atom_groups(AtomGroup(positions, AtomType(), 1))
    gen.set_system_size(10)
    with open("test.gen", "w", encoding="utf-8") as file:
        gen.write_full(file)


def test_simple_atoms_polymer():
    positions = np.zeros(shape=(100, 3), dtype=np.float32)
    gen = Generator()
    gen.add_atom_groups(
        AtomGroup(positions, AtomType(), 1, polymer_bond_type=BAI_Type(BAI_Kind.BOND, "fene"))
    )
    gen.set_system_size(10)
    with open("test.gen", "w", encoding="utf-8") as file:
        gen.write_full(file)


def test_with_bonds():
    positions = np.zeros(shape=(25, 3), dtype=np.float32)

    gen = Generator()
    gen.set_system_size(10)

    bt1 = BAI_Type(BAI_Kind.BOND, "harmonic")
    bt2 = BAI_Type(BAI_Kind.BOND, "harmonic")

    group1 = AtomGroup(positions, AtomType(), 1, polymer_bond_type=bt2)
    gen.add_atom_groups(group1)

    group2 = AtomGroup(np.copy(positions), AtomType(), 3)
    gen.add_atom_groups(group2)

    gen.bais.append(BAI(bt1, (group1, 1), (group2, 0)))

    gen.bais.append(BAI(BAI_Type(BAI_Kind.BOND, "cosine"), (group1, 5), (group1, 6)))

    gen.bais.append(BAI(bt1, (group1, 9), (group1, 16)))

    with open("test.gen", "w", encoding="utf-8") as file:
        gen.write_full(file)


def test_with_pairs():
    positions = np.zeros(shape=(25, 3), dtype=np.float32)

    gen = Generator()
    gen.set_system_size(10)

    at1 = AtomType()
    group1 = AtomGroup(positions, at1, 1)
    gen.add_atom_groups(group1)

    group2 = AtomGroup(np.copy(positions), AtomType(), 3)
    gen.add_atom_groups(group2)

    pairwise = PairWise("PairIJ Coeffs # hybrid\n", "lj/cut {} {} {}\n", [0, 0, 0])
    pairwise.add_interaction(at1, at1, 1, 2, 3)

    gen.pair_interactions.append(pairwise)

    with open("test.gen", "w", encoding="utf-8") as file:
        gen.write_full(file)


def all_tests():
    test_simple_atoms()
    test_simple_atoms_polymer()
    test_with_bonds()
    test_with_pairs()


def main():
    all_tests()


if __name__ == "__main__":
    main()
