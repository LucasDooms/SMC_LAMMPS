# Copyright (c) 2024-2026 Lucas Dooms

# File containing different initial DNA configurations

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from smc_lammps.generate.default_parameters import Parameters
from smc_lammps.generate.generator import (
    BAI,
    COORD_TYPE,
    AtomGroup,
    AtomIdentifier,
    AtomType,
    BAI_Kind,
    BAI_Type,
    MoleculeId,
    Nx3Array,
    PairWise,
)
from smc_lammps.generate.structures import structure_creator
from smc_lammps.generate.structures.dna import dna_creator
from smc_lammps.generate.structures.polymer import Polymer
from smc_lammps.generate.structures.smc.smc import SMC
from smc_lammps.generate.util import get_closest, pos_from_id

type StrandId = tuple[int, int]  # strand_id, id


@dataclass
class DnaParameters:
    nDNA: int
    DNA_bond_length: float
    DNA_mass: float
    type: AtomType
    mol_DNA: int
    bond: BAI_Type
    angle: BAI_Type
    ssangle: BAI_Type

    def create_dna_polymer(self, dna_positions: Sequence[Nx3Array]) -> Polymer:
        return Polymer(
            *[
                AtomGroup(
                    positions=r_DNA,
                    atom_type=self.type,
                    molecule_index=self.mol_DNA,
                    polymer_bond_type=self.bond,
                    polymer_angle_type=self.angle,
                )
                for r_DNA in dna_positions
            ]
        )


@dataclass
class InteractionParameters:
    ###########
    # DNA-DNA #
    ###########

    sigma_DNA_DNA: float
    epsilon_DNA_DNA: float
    rcut_DNA_DNA: float
    k_bond_DNA_DNA: float

    ###########
    # SMC-DNA #
    ###########

    sigma_SMC_DNA: float
    epsilon_SMC_DNA: float
    rcut_SMC_DNA: float

    #############
    # Sites-DNA #
    #############

    # Sigma of LJ attraction (same as those of the repulsive SMC sites)
    sigma_upper_site_DNA: float

    # Cutoff distance of LJ attraction
    rcut_lower_site_DNA: float

    # Epsilon parameter of LJ attraction
    epsilon_upper_site_DNA: float


@dataclass
class Tether:
    class Obstacle:
        def move(self, vector: Nx3Array) -> None:
            raise NotImplementedError(f"don't use Tether.Obstacle directly {vector}")

        def get_all_groups(self) -> list[AtomGroup]:
            return []

        def add_interactions(
            self,
            pair_inter: PairWise,
            dna_type: AtomType,
            ip: InteractionParameters,
            kBT: float,
            smc: SMC,
        ) -> None:
            pass

    class Wall(Obstacle):
        def __init__(self, y_pos: float) -> None:
            super().__init__()
            self.y_pos = y_pos

        def move(self, vector: Nx3Array) -> None:
            self.y_pos += vector[1]

        def get_all_groups(self) -> list[AtomGroup]:
            return super().get_all_groups()

    class Gold(Obstacle):
        def __init__(self, group: AtomGroup, radius: float, cut: float, tether_bond: BAI) -> None:
            super().__init__()
            self.group = group
            self.radius = radius
            self.cut = cut
            self.tether_bond = tether_bond

        def move(self, vector: Nx3Array) -> None:
            self.group.positions[0] += vector

        def get_all_groups(self) -> list[AtomGroup]:
            return super().get_all_groups() + [self.group]

        def add_interactions(
            self,
            pair_inter: PairWise,
            dna_type: AtomType,
            ip: InteractionParameters,
            kBT: float,
            smc: SMC,
        ) -> None:
            super().add_interactions(pair_inter, dna_type, ip, kBT, smc)

            pair_inter.add_interaction(
                self.group.type,
                dna_type,
                ip.epsilon_DNA_DNA * kBT,
                self.radius,
                self.cut,
            )
            # Obstacle repels arms
            pair_inter.add_interaction(
                self.group.type,
                smc.t_arms_heads,
                ip.epsilon_DNA_DNA * kBT,
                self.radius,
                self.cut,
            )
            # Obstacle repels kleisin
            pair_inter.add_interaction(
                self.group.type,
                smc.t_kleisin,
                ip.epsilon_DNA_DNA * kBT,
                self.radius,
                self.cut,
            )
            if smc.has_toroidal_hinge():
                pair_inter.add_interaction(
                    self.group.type,
                    smc.t_hinge,
                    ip.epsilon_DNA_DNA * kBT,
                    self.radius,
                    self.cut,
                )

    polymer: Polymer
    dna_tether_id: StrandId | AtomIdentifier
    obstacle: Tether.Obstacle
    bonds: list[BAI]
    angles: list[BAI]

    @staticmethod
    def get_gold_mass(radius: float) -> float:
        """radius in nanometers, returns attograms"""
        density = 0.0193  # attograms per nanometer^3
        volume = 4.0 / 3.0 * np.pi * radius**3
        return density * volume

    @classmethod
    def get_obstacle(
        cls, real_obstacle: bool, ip: InteractionParameters, tether_group: AtomGroup
    ) -> Tether.Obstacle:
        if real_obstacle:
            obstacle_radius = 100  # nanometers
            obstacle_cut = obstacle_radius * 2 ** (1 / 6)
            pos = tether_group.positions[0] - np.array([0, obstacle_radius, 0], dtype=float)
            obstacle_type = AtomType(cls.get_gold_mass(obstacle_radius))
            obstacle_group = AtomGroup(
                positions=np.array([pos]),
                atom_type=obstacle_type,
                molecule_index=tether_group.molecule_index,
            )

            obstacle_bond = BAI_Type(
                BAI_Kind.BOND,
                "fene/expand",
                f"{ip.k_bond_DNA_DNA} {obstacle_radius} {0.0} {0.0} {obstacle_radius}\n",
            )
            tether_obstacle_bond = BAI(obstacle_bond, (tether_group, 0), (obstacle_group, 0))
            return Tether.Gold(obstacle_group, obstacle_radius, obstacle_cut, tether_obstacle_bond)
        else:
            return Tether.Wall(tether_group.positions[0][1])

    @classmethod
    def create_tether(
        cls,
        dna_tether_id: StrandId,
        tether_length: int,
        bond_length: float,
        mass: float,
        bond_type: BAI_Type,
        angle_type: BAI_Type,
        obstacle: Tether.Obstacle,
    ) -> Tether:
        tether_positions = (
            structure_creator.get_straight_segment(tether_length, [0, 1, 0]) * bond_length
        )
        tether_group = AtomGroup(
            positions=tether_positions,
            atom_type=AtomType(mass),
            molecule_index=MoleculeId.get_next(),
            polymer_bond_type=bond_type,
            polymer_angle_type=angle_type,
            charge=0.2,
        )
        polymer = Polymer(tether_group)

        return Tether(
            polymer=polymer, dna_tether_id=dna_tether_id, obstacle=obstacle, bonds=[], angles=[]
        )

    def move(self, vector) -> None:
        for group in self.polymer.atom_groups:
            group.positions += vector
        self.obstacle.move(vector)

    def get_all_groups(self) -> list[AtomGroup]:
        return self.polymer.atom_groups + self.obstacle.get_all_groups()

    def handle_end_points(self, end_points: list[AtomIdentifier]) -> None:
        # freeze bottom of tether if using infinite wall
        if isinstance(self.obstacle, Tether.Wall):
            end_points += [self.polymer.get_id_from_list_index(0)]

    def add_interactions(
        self,
        pair_inter: PairWise,
        ip: InteractionParameters,
        dna_type: AtomType,
        smc: SMC,
        kBT: float,
    ) -> None:
        self.obstacle.add_interactions(pair_inter, dna_type, ip, kBT, smc)

        unique_types = set()
        for grp in self.polymer.atom_groups:
            tether_type = grp.type
            if tether_type in unique_types:
                continue
            unique_types.add(tether_type)

            if tether_type.mass == dna_type.mass:
                factor = 1.0
            elif tether_type.mass == dna_type.mass / 2.0:
                factor = 2.0
            else:
                raise RuntimeError("unexpected mass for tether type")
            self.add_tether_interactions(tether_type, factor, pair_inter, ip, dna_type, smc, kBT)

    def add_tether_interactions(
        self,
        tether_type: AtomType,
        factor: float,
        pair_inter: PairWise,
        ip: InteractionParameters,
        dna_type: AtomType,
        smc: SMC,
        kBT: float,
    ) -> None:
        # tether
        pair_inter.add_interaction(
            tether_type,
            tether_type,
            ip.epsilon_DNA_DNA * kBT,
            ip.sigma_DNA_DNA,
            ip.rcut_DNA_DNA,
        )
        pair_inter.add_interaction(
            tether_type,
            dna_type,
            ip.epsilon_DNA_DNA * kBT,
            ip.sigma_DNA_DNA,
            ip.rcut_DNA_DNA,
        )
        # Tether repels arms, smaller interaction distance
        pair_inter.add_interaction(
            tether_type,
            smc.t_arms_heads,
            ip.epsilon_SMC_DNA * kBT,
            ip.sigma_SMC_DNA / factor,
            ip.rcut_SMC_DNA / factor,
        )
        # Tether repels kleisin
        pair_inter.add_interaction(
            tether_type,
            smc.t_kleisin,
            ip.epsilon_SMC_DNA * kBT,
            ip.sigma_SMC_DNA,
            ip.rcut_SMC_DNA,
        )
        if smc.has_toroidal_hinge():
            pair_inter.add_interaction(
                tether_type,
                smc.t_hinge,
                ip.epsilon_SMC_DNA * kBT,
                ip.sigma_SMC_DNA / factor,
                ip.rcut_SMC_DNA / factor,
            )
        # Optional: don't allow bridge to go through tether
        # pair_inter.add_interaction(
        #     tether_type, smc.atp_type,
        #     ip.epsilonSMCvsDNA * kBT, ip.sigmaSMCvsDNA, ip.rcutSMCvsDNA
        # )
        # Optional: allow tether to bond to siteD
        # pair_inter.add_interaction(
        #     tether_type, smc.siteD_type,
        #     ip.epsilonSiteDvsDNA * kBT, ip.sigmaSiteDvsDNA, ip.rcutSiteDvsDNA
        # )
        if isinstance(self.obstacle, Tether.Gold):
            pair_inter.add_interaction(
                self.obstacle.group.type,
                tether_type,
                ip.epsilon_DNA_DNA * kBT,
                self.obstacle.radius,
                self.obstacle.cut,
            )

    def get_bonds(self, bond_type: BAI_Type, dna_config: DnaConfiguration) -> list[BAI]:
        atom_id = dna_config.map_to_atom_id(self.dna_tether_id)
        bonds = [BAI(bond_type, self.polymer.get_id_from_list_index(-1), atom_id)]
        bonds += self.bonds
        if isinstance(self.obstacle, Tether.Gold):
            bonds += [self.obstacle.tether_bond]
        return bonds

    def get_angles(self) -> list[BAI]:
        return self.angles


# redefine a method in a class using the old_method
# this avoids infinite recursion caused by a function calling itself
def class_decorator_factory(old_method):
    def class_decorator(function):
        def new_function(*args, **kwargs):
            return function(old_method, *args, **kwargs)

        return new_function

    return class_decorator


# decorator to add tether logic to DnaConfiguration classes
def with_tether(cls):
    @class_decorator_factory(old_method=cls.get_all_groups)
    def get_all_groups(f, self) -> list[AtomGroup]:
        return f(self) + self.tether.get_all_groups()

    cls.get_all_groups = get_all_groups

    @class_decorator_factory(old_method=cls.get_post_process_parameters)
    def get_post_process_parameters(f, self) -> DnaConfiguration.PostProcessParameters:
        ppp = f(self)
        self.tether.handle_end_points(ppp.end_points)
        return ppp

    cls.get_post_process_parameters = get_post_process_parameters

    @class_decorator_factory(old_method=cls.add_interactions)
    def add_interactions(f, self, pair_inter: PairWise) -> None:
        f(self, pair_inter)
        self.tether.add_interactions(
            pair_inter,
            self.inter_par,
            self.dna_parameters.type,
            self.smc,
            self.par.kB * self.par.T,
        )

    cls.add_interactions = add_interactions

    @class_decorator_factory(old_method=cls.get_bonds)
    def get_bonds(f, self) -> list[BAI]:
        return f(self) + self.tether.get_bonds(self.dna_parameters.bond, self)

    cls.get_bonds = get_bonds

    @class_decorator_factory(old_method=cls.get_angles)
    def get_angles(f, self) -> list[BAI]:
        return f(self) + self.tether.get_angles()

    cls.get_angles = get_angles

    return cls


class DnaConfiguration:
    @dataclass
    class PostProcessParameters:
        # LAMMPS DATA

        # indices to freeze permanently
        end_points: list[AtomIdentifier]
        # indices to temporarily freeze, in order to equilibrate the system
        freeze_indices: list[AtomIdentifier]
        # forces to apply:
        # the keys are the forces (3d vectors), and the value is a list of indices to which the force will be applied
        stretching_forces_array: dict[tuple[float, float, float], list[AtomIdentifier]]

        # POST PROCESSING

        # indices to use for marked bead tracking
        dna_indices_list: dict[int, list[tuple[AtomIdentifier, AtomIdentifier]]]

    @classmethod
    def set_parameters(cls, par: Parameters, inter_par: InteractionParameters) -> None:
        cls.par = par
        cls.inter_par = inter_par

    @classmethod
    def set_smc(cls, smc: SMC) -> None:
        cls.smc = smc

    def __init__(self, dna_strands: list[Polymer], dna_parameters: DnaParameters) -> None:
        self.dna_strands = dna_strands
        self.dna_parameters = dna_parameters
        self.kBT = self.par.kB * self.par.T
        self.beads: list[AtomGroup] = []
        self.bead_sizes: list[float] = []
        self.bead_bonds: list[tuple[BAI_Type, list[StrandId | AtomIdentifier]]] = []
        self.molecule_overrides: list[tuple[int, int, int]] = []  # (strand_id, index, new_mol)
        self.tether: None | Tether = None

    @property
    def all_dna_groups(self) -> list[AtomGroup]:
        return [grp for strand in self.dna_strands for grp in strand.atom_groups]

    def get_all_groups(self) -> list[AtomGroup]:
        return self.all_dna_groups + self.beads

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, r_lower_site, par) -> DnaConfiguration:
        return NotImplemented

    def get_post_process_parameters(self) -> PostProcessParameters:
        return self.PostProcessParameters(
            end_points=[],
            freeze_indices=[],
            stretching_forces_array=dict(),
            dna_indices_list=dict(),
        )

    @staticmethod
    def strand_concat(lst: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        if not lst:
            return []

        def fuse(alist: list[tuple[int, int]], next: tuple[int, int]):
            if not alist or alist[-1][1] + 1 != next[0]:
                alist.append(next)
            else:
                alist[-1] = (alist[-1][0], next[1])

        new = []
        for item in lst:
            fuse(new, item)

        return new

    def dna_indices_list_get_all_dna(
        self,
        strand_index: int,
    ) -> list[tuple[AtomIdentifier, AtomIdentifier]]:
        return [tup for tup in self.dna_strands[strand_index].all_indices_list()]

    def dna_indices_list_get_dna_from_to(
        self, strand_index: int, from_ratio: float, to_ratio: float
    ) -> list[tuple[AtomIdentifier, AtomIdentifier]]:
        return [
            tup
            for tup in self.dna_strands[strand_index].indices_list_from_to_percent(
                from_ratio, to_ratio
            )
        ]

    def dna_indices_list_get_dna_to(
        self, strand_index: int, ratio: float
    ) -> list[tuple[AtomIdentifier, AtomIdentifier]]:
        return [tup for tup in self.dna_strands[strand_index].indices_list_to_percent(ratio)]

    def add_interactions(self, pair_inter: PairWise) -> None:
        dna_type = self.dna_parameters.type
        ip = self.inter_par
        kBT = self.par.kB * self.par.T
        pair_inter.add_interaction(
            dna_type,
            dna_type,
            ip.epsilon_DNA_DNA * kBT,
            ip.sigma_DNA_DNA,
            ip.rcut_DNA_DNA,
        )
        # DNA repels arms
        pair_inter.add_interaction(
            dna_type,
            self.smc.t_arms_heads,
            ip.epsilon_SMC_DNA * kBT,
            ip.sigma_SMC_DNA,
            ip.rcut_SMC_DNA,
        )
        # DNA repels kleisin
        pair_inter.add_interaction(
            dna_type,
            self.smc.t_kleisin,
            ip.epsilon_SMC_DNA * kBT,
            ip.sigma_SMC_DNA,
            ip.rcut_SMC_DNA,
        )
        # DNA repels shields, allow closer interaction
        pair_inter.add_interaction(
            dna_type,
            self.smc.t_shield,
            ip.epsilon_SMC_DNA * kBT,
            ip.sigma_SMC_DNA / 2.0,
            ip.rcut_SMC_DNA / 2.0,
            allow_unused=True,
        )
        pair_inter.add_interaction(
            dna_type,
            self.smc.t_hinge,
            ip.epsilon_SMC_DNA * kBT,
            ip.sigma_SMC_DNA,
            ip.rcut_SMC_DNA,
            allow_unused=True,
        )
        pair_inter.add_interaction(
            dna_type,
            self.smc.t_side_site,
            ip.epsilon_upper_site_DNA * kBT,
            ip.sigma_upper_site_DNA * 0.33,
            ip.rcut_lower_site_DNA * 0.66,
            allow_unused=True,
        )
        pair_inter.add_interaction(
            dna_type,
            self.smc.t_lower_site,
            ip.epsilon_upper_site_DNA * kBT,
            ip.sigma_upper_site_DNA,
            ip.rcut_lower_site_DNA,
        )

        # every bead should repel certain SMC groups
        for (bead, bead_size), smc_grp in product(
            zip(self.beads, self.bead_sizes), self.smc.get_repulsive_groups()
        ):
            pair_inter.add_interaction(
                bead.type,
                smc_grp.type,
                ip.epsilon_SMC_DNA * kBT,
                bead_size,
                bead_size * (2 ** (1 / 6)),
            )

    def map_to_atom_id(self, strnd: StrandId | AtomIdentifier) -> AtomIdentifier:
        x = strnd[0]
        y = strnd[1]
        if isinstance(x, AtomGroup):
            return (x, y)
        else:
            return self.dna_strands[x].get_id_from_list_index(y)

    def get_bonds(self) -> list[BAI]:
        return [
            BAI(
                bond[0],
                *[self.map_to_atom_id(strnd) for strnd in bond[1]],
            )
            for bond in self.bead_bonds
        ]

    def get_angles(self) -> list[BAI]:
        return []

    def update_tether_bond(self, old_id: AtomIdentifier, bead: None | AtomIdentifier) -> None:
        if self.tether is None:
            return

        if self.tether.dna_tether_id[0] is old_id[0]:
            if self.tether.dna_tether_id[1] == old_id[1] and bead is not None:
                self.tether.dna_tether_id = bead

        old = pos_from_id(old_id)
        new = pos_from_id(self.map_to_atom_id(self.tether.dna_tether_id))
        self.tether.move(new - old)

    def change_dna_stiffness(
        self,
        strand_index: int,
        from_id: int,
        to_id: int,
        bond: BAI_Type,
        angle: BAI_Type,
    ) -> None:
        from_ = self.dna_strands[strand_index].get_id_from_list_index(from_id)
        left, middle = self.dna_strands[strand_index].split(from_)

        # only get this id after the split, since groups change!
        to_ = self.dna_strands[strand_index].get_id_from_list_index(to_id)
        middle, right = self.dna_strands[strand_index].split(to_)

        # add interactions/exceptions
        bais: list[tuple[BAI_Type, list[AtomIdentifier | StrandId]]] = [
            (bond, [(strand_index, from_id - 1), (strand_index, from_id)]),
            (bond, [(strand_index, to_id - 1), (strand_index, to_id)]),
        ]
        assert left.polymer_angle_type is not None
        assert right.polymer_angle_type is not None
        bais = bais + [
            (
                left.polymer_angle_type,
                [(strand_index, from_id - 2), (strand_index, from_id - 1), (strand_index, from_id)],
            ),
            (
                angle,
                [(strand_index, from_id - 1), (strand_index, from_id), (strand_index, from_id + 1)],
            ),
            (angle, [(strand_index, to_id - 2), (strand_index, to_id - 1), (strand_index, to_id)]),
            (
                right.polymer_angle_type,
                [(strand_index, to_id - 1), (strand_index, to_id), (strand_index, to_id + 1)],
            ),
        ]

        middle.polymer_angle_type = angle

        self.bead_bonds += bais

    def add_bead_to_dna(
        self,
        bead_type: AtomType,
        mol_index: int,
        strand_index: int,
        dna_id: int,
        bond: None | BAI_Type,  # if None -> rigid attachment to dna_atom
        angle: None | BAI_Type,  # only used if bond is not None
        bead_size: float,
    ) -> AtomIdentifier:
        dna_atom = self.dna_strands[strand_index].get_id_from_list_index(dna_id)
        # place on a DNA bead
        location = pos_from_id(dna_atom)

        # create a bead
        bead = AtomGroup(location.reshape(1, 3), bead_type, mol_index)

        bais = []
        if bond is None:
            self.molecule_overrides.append((strand_index, dna_id, mol_index))
        else:
            first_group, second_group = self.dna_strands[strand_index].split(dna_atom)
            # dna_atom is now invalid, change it to new value
            dna_atom = (second_group, 0)

            # add interactions/exceptions
            bais += [
                (bond, [(strand_index, dna_id - 1), (bead, 0)]),
                (bond, [(strand_index, dna_id), (bead, 0)]),
            ]
            if angle is not None:
                left_angle = (
                    angle,
                    [(strand_index, dna_id - 2), (strand_index, dna_id - 1), (bead, 0)],
                )
                middle_angle = (
                    angle,
                    [(strand_index, dna_id - 1), (bead, 0), (strand_index, dna_id)],
                )
                right_angle = (
                    angle,
                    [(bead, 0), (strand_index, dna_id), (strand_index, dna_id + 1)],
                )

                if dna_id > 1:
                    bais.append(left_angle)

                bais.append(middle_angle)

                if dna_id < self.dna_strands[strand_index].full_list_length() - 1:
                    bais.append(right_angle)

            # move to correct distances
            bead.positions[0, 0] += bead_size
            # first_group.positions[:, 0] += 2 * bead_size - self.dna_parameters.DNA_bond_length
            self.dna_strands[strand_index].move(
                np.array(
                    [2 * bead_size - self.dna_parameters.DNA_bond_length, 0, 0], dtype=COORD_TYPE
                ),
                (0, self.dna_strands[strand_index].get_absolute_index((first_group, -1))),
            )

            self.update_tether_bond(dna_atom, (bead, 0))

        self.beads.append(bead)
        self.bead_sizes.append(bead_size)
        self.bead_bonds += bais

        # ensure the bead position updates with the polymer
        self.dna_strands[strand_index].add_tagged_atoms(dna_atom, (bead, 0))

        return (bead, 0)

    def get_stopper_ids(self) -> list[StrandId]:
        return []

    @staticmethod
    def str_to_config(string: str) -> DnaConfiguration:
        string = string.lower()
        return {
            "line": Line,
            "folded": Folded,
            "right_angle": RightAngle,
            "doubled": Doubled,
            "safety": Safety,
            "obstacle": Obstacle,
            "obstacle_safety": ObstacleSafety,
            "advanced_obstacle_safety": AdvancedObstacleSafety,
        }[string]


class Line(DnaConfiguration):
    """Straight line of DNA"""

    def __init__(self, dna_strands: list[Polymer], dna_parameters: DnaParameters):
        super().__init__(dna_strands, dna_parameters)

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, r_lower_site, par) -> Line:
        default_dna_pos = r_lower_site[1] + np.array([0, par.cutoff6, 0])

        # 1.
        [r_DNA] = dna_creator.get_dna_coordinates_straight(
            dna_parameters.nDNA, dna_parameters.DNA_bond_length
        )

        # 2.
        # make sure SMC contains DNA
        goal = default_dna_pos
        start = np.array(
            [
                r_DNA[int(len(r_DNA) / 1.3)][0] + 10.0 * dna_parameters.DNA_bond_length,
                r_DNA[-1][1],
                0,
            ]
        )
        shift = (goal - start).reshape(1, 3)
        r_DNA += shift

        return cls([dna_parameters.create_dna_polymer([r_DNA])], dna_parameters)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        strand = self.dna_strands[0]

        if par.force:
            ppp.stretching_forces_array[(par.force, 0, 0)] = [strand.first_id()]
            ppp.stretching_forces_array[(-par.force, 0, 0)] = [strand.last_id()]
        else:
            ppp.end_points += [strand.first_id(), strand.last_id]

        ppp.freeze_indices += [
            strand.get_id_from_list_index(
                get_closest(strand.full_list(), self.smc.pos.r_lower_site[1]),
            ),  # closest to bottom -> r_lower_site[1]
            strand.get_id_from_list_index(
                get_closest(strand.full_list(), self.smc.pos.r_middle_site[1]),
            ),  # closest to middle -> r_middle_site[1]
        ]

        ppp.dna_indices_list[0] = self.dna_indices_list_get_all_dna(0)

        return ppp

    def get_stopper_ids(self) -> list[StrandId]:
        return [(0, -1)]


class Folded(DnaConfiguration):
    def __init__(self, dna_strands: list[Polymer], dna_parameters: DnaParameters, dna_center):
        super().__init__(dna_strands, dna_parameters)
        self.dna_center = dna_center

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, r_lower_site, par) -> Folded:
        # place your DNA here, inside the SMC
        default_dna_pos = r_lower_site[1] + np.array([0, par.cutoff6, 0])

        # 1.
        [r_DNA], dna_center = dna_creator.get_dna_coordinates_twist(
            dna_parameters.nDNA, dna_parameters.DNA_bond_length, 17
        )

        # 2.
        # make sure SMC contains DNA
        goal = default_dna_pos
        start = np.array([dna_center[0] + 100.0 * dna_parameters.DNA_bond_length, r_DNA[-1][1], 0])
        shift = (goal - start).reshape(1, 3)
        r_DNA += shift

        return cls([dna_parameters.create_dna_polymer([r_DNA])], dna_parameters, dna_center)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        strand = self.dna_strands[0]

        if par.force:
            # 2 half strength forces at both ends pointing right (+x)
            ppp.stretching_forces_array[(par.force / 2.0, 0, 0)] = [
                strand.first_id(),
                strand.last_id(),
            ]
            # 1 full strength force pointing left (-x) at midway point (fold)
            ppp.stretching_forces_array[(-par.force, 0, 0)] = [
                strand.get_percent_id(0.5),
            ]
        else:
            ppp.end_points += [strand.first_id(), strand.last_id()]

        ppp.freeze_indices += [
            strand.get_id_from_list_index(
                get_closest(strand.full_list(), self.smc.pos.r_lower_site[1]),
            ),  # closest to bottom -> r_lower_site[1]
            strand.get_id_from_list_index(
                get_closest(strand.full_list(), self.smc.pos.r_middle_site[1]),
            ),  # closest to middle -> r_middle_site[1]
        ]

        ppp.dna_indices_list[0] = self.dna_indices_list_get_dna_to(0, ratio=0.5)
        ppp.dna_indices_list[1] = self.dna_indices_list_get_dna_from_to(
            0, from_ratio=0.5, to_ratio=1.0
        )

        return ppp

    def get_stopper_ids(self) -> list[StrandId]:
        # add bead at halfway point
        return [(0, self.dna_strands[0].convert_ratio(0.5))]


class RightAngle(DnaConfiguration):
    def __init__(self, dna_strands: list[Polymer], dna_parameters: DnaParameters, dna_center):
        super().__init__(dna_strands, dna_parameters)
        self.dna_center = dna_center

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, r_lower_site, par) -> RightAngle:
        # place your DNA here, inside the SMC
        default_dna_pos = r_lower_site[1] + np.array([0, par.cutoff6, 0])

        # 1.
        [r_DNA], dna_center = dna_creator.get_dna_coordinates(
            dna_parameters.nDNA, dna_parameters.DNA_bond_length, 14, 10
        )

        # 2.
        # make sure SMC touches the DNA at the lower site (siteD)
        goal = default_dna_pos
        start = np.array([dna_center[0] - 10.0 * dna_parameters.DNA_bond_length, dna_center[1], 0])
        shift = (goal - start).reshape(1, 3)
        r_DNA += shift

        return cls([dna_parameters.create_dna_polymer([r_DNA])], dna_parameters, dna_center)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        strand = self.dna_strands[0]

        if par.force:
            # FIX: total net force is nonzero, may cause issues?
            ppp.stretching_forces_array[(0, par.force, 0)] = [strand.first_id()]
            ppp.stretching_forces_array[(-par.force, 0, 0)] = [strand.last_id()]
        else:
            ppp.end_points += [strand.first_id(), strand.last_id()]

        # find closest DNA bead to siteD
        # closest_DNA_index = get_closest(self.dna_groups[0].positions, r_lower_site[1])

        ppp.dna_indices_list[0] = [(strand.first_id(), strand.get_percent_id(0.5))]

        return ppp

    def get_stopper_ids(self) -> list[StrandId]:
        return []


class Doubled(DnaConfiguration):
    def __init__(self, dna_strands: list[Polymer], dna_parameters: DnaParameters, dna_center):
        super().__init__(dna_strands, dna_parameters)
        self.dna_center = dna_center

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, r_lower_site, par) -> Doubled:
        # place your DNA here, inside the SMC
        default_dna_pos = r_lower_site[1] + np.array([0, par.cutoff6, 0])

        # 1.
        r_DNA_list, dna_center = dna_creator.get_dna_coordinates_doubled(
            dna_parameters.nDNA, dna_parameters.DNA_bond_length, 24
        )

        # 2.
        # make sure SMC contains DNA
        goal = default_dna_pos
        start = np.array(
            [
                dna_center[0] + 30.0 * dna_parameters.DNA_bond_length,
                r_DNA_list[0][-1][1],
                0,
            ]
        )
        shift = (goal - start).reshape(1, 3)
        r_DNA_list[0] += shift
        r_DNA_list[1] += shift

        return cls(
            [
                dna_parameters.create_dna_polymer(r_DNA_list[0]),
                dna_parameters.create_dna_polymer(r_DNA_list[1]),
            ],
            dna_parameters,
            dna_center,
        )

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        # get dna beads to freeze
        for strand in self.dna_strands:
            if par.force:
                # 2 half strength forces at both ends pointing right (+x)
                ppp.stretching_forces_array[(par.force / 2.0, 0, 0)] = [
                    strand.first_id(),
                    strand.last_id(),
                ]
                # 1 full strength force pointing left (-x) at midway point (fold)
                ppp.stretching_forces_array[(-par.force, 0, 0)] = [
                    strand.get_percent_id(0.5),
                ]
            else:
                ppp.end_points += [strand.first_id(), strand.last_id()]
            # TODO: fix for DOUBLED DNA, gives same bead twice
            ppp.freeze_indices += [
                strand.get_id_from_list_index(
                    get_closest(strand.full_list(), self.smc.pos.r_lower_site[1]),
                ),  # closest to bottom
                strand.get_id_from_list_index(
                    get_closest(strand.full_list(), self.smc.pos.r_middle_site[1]),
                ),  # closest to middle
            ]

        for strand_id in range(len(self.dna_strands)):
            ppp.dna_indices_list[strand_id] = self.dna_indices_list_get_dna_to(strand_id, ratio=0.5)

        return ppp

    def get_stopper_ids(self) -> list[StrandId]:
        # TODO: see todo for folded config
        return []


@with_tether
class Obstacle(DnaConfiguration):
    def __init__(
        self,
        dna_strands: list[Polymer],
        dna_parameters: DnaParameters,
        tether: Tether,
        dna_start_index: int,
    ):
        super().__init__(dna_strands, dna_parameters)
        self.tether = tether
        self.dna_start_index = dna_start_index

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, r_lower_site, par) -> Obstacle:
        # place your DNA here, inside the SMC
        default_dna_pos = r_lower_site[1] + np.array([0, par.cutoff6, 0])

        # 1.
        [r_DNA] = dna_creator.get_dna_coordinates_straight(
            dna_parameters.nDNA, dna_parameters.DNA_bond_length
        )

        # 2.
        # make sure SMC contains DNA
        goal = default_dna_pos
        dna_start_index = int(len(r_DNA) * 9 / 15)
        start = np.array(
            [
                r_DNA[dna_start_index][0] - 10.0 * dna_parameters.DNA_bond_length,
                r_DNA[dna_start_index][1],
                0,
            ]
        )
        shift = (goal - start).reshape(1, 3)
        r_DNA += shift

        dna_strand = dna_parameters.create_dna_polymer([r_DNA])

        dna_bead_to_tether_id = int(len(r_DNA) * 7.5 / 15)
        tether = Tether.create_tether(
            (0, dna_bead_to_tether_id),
            25,
            dna_parameters.DNA_bond_length,
            dna_parameters.DNA_mass / 2.0,
            dna_parameters.bond,
            dna_parameters.ssangle,
            Tether.Obstacle(),
        )
        obstacle = Tether.get_obstacle(True, cls.inter_par, tether.polymer.atom_groups[0])
        tether.obstacle = obstacle
        # place the tether next to the DNA bead
        tether.move(
            r_DNA[dna_bead_to_tether_id] - pos_from_id(tether.polymer.get_id_from_list_index(-1))
        )
        # move down a little
        tether.move(np.array([0, -dna_parameters.DNA_bond_length, 0], dtype=float))

        return cls([dna_strand], dna_parameters, tether, dna_start_index)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        strand = self.dna_strands[0]

        if par.force:
            ppp.stretching_forces_array[(par.force, 0, 0)] = [strand.first_id()]
            ppp.stretching_forces_array[(-par.force, 0, 0)] = [strand.last_id()]
        else:
            ppp.end_points += [strand.first_id(), strand.last_id()]

        ppp.dna_indices_list[0] = strand.indices_list_to(self.dna_start_index)

        return ppp

    def get_stopper_ids(self) -> list[StrandId]:
        return [(0, -1)]


class Safety(DnaConfiguration):
    def __init__(
        self,
        dna_strands: list[Polymer],
        dna_parameters: DnaParameters,
        dna_safety_belt_index,
    ):
        super().__init__(dna_strands, dna_parameters)
        self.dna_safety_belt_index = dna_safety_belt_index

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, r_lower_site, par) -> Safety:
        # 1.
        [r_DNA], belt_location, dna_safety_belt_index, _ = (
            dna_creator.get_dna_coordinates_safety_belt(
                dna_parameters.nDNA, dna_parameters.DNA_bond_length
            )
        )

        # 2.
        # make sure SMC contains DNA
        shift = r_lower_site[1] - belt_location
        shift[1] -= 0.65 * par.cutoff6 + 0.5 * par.cutoff6  # TODO: if siteDup
        r_DNA += shift

        return cls(
            [dna_parameters.create_dna_polymer([r_DNA])], dna_parameters, dna_safety_belt_index
        )

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        strand = self.dna_strands[0]

        if par.force:
            ppp.stretching_forces_array[(par.force, 0, 0)] = [strand.first_id()]
            ppp.stretching_forces_array[(-par.force, 0, 0)] = [strand.last_id()]
        else:
            ppp.end_points += [strand.first_id(), strand.last_id()]

        ppp.dna_indices_list[0] = self.dna_indices_list_get_all_dna(0)

        # prevent breaking of safety belt
        # ppp.freeze_indices += [
        #     *[(self.dna_groups[0], self.dna_safety_belt_index + i) for i in range(-6, 6)],
        #     *[(self.smc.siteD_group, i) for i in range(len(self.smc.siteD_group.positions))]
        # ]

        return ppp

    def get_stopper_ids(self) -> list[StrandId]:
        return [(0, -1)]


@with_tether
class ObstacleSafety(DnaConfiguration):
    def __init__(
        self,
        dna_strands: list[Polymer],
        dna_parameters: DnaParameters,
        tether: Tether,
        dna_safety_belt_index: int,
    ):
        super().__init__(dna_strands, dna_parameters)
        self.tether = tether
        self.dna_safety_belt_index = dna_safety_belt_index

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, r_lower_site, par) -> ObstacleSafety:
        # 1.
        [r_DNA], belt_location, dna_safety_belt_index, dna_bead_to_tether_id = (
            dna_creator.get_dna_coordinates_safety_belt(
                dna_parameters.nDNA, dna_parameters.DNA_bond_length
            )
        )

        # 2.
        # make sure SMC contains DNA
        shift = r_lower_site[1] - belt_location
        shift[1] -= 0.65 * par.cutoff6 + 0.5 * par.cutoff6  # TODO: if siteDup
        r_DNA += shift

        dna_strand = dna_parameters.create_dna_polymer([r_DNA])

        tether = Tether.create_tether(
            (0, dna_bead_to_tether_id),
            35,
            dna_parameters.DNA_bond_length,
            dna_parameters.DNA_mass / 2.0,
            dna_parameters.bond,
            dna_parameters.ssangle,
            Tether.Obstacle(),
        )
        obstacle = Tether.get_obstacle(True, cls.inter_par, tether.polymer.atom_groups[0])
        tether.obstacle = obstacle

        tether.move(
            r_DNA[dna_bead_to_tether_id] - pos_from_id(tether.polymer.get_id_from_list_index(-1))
        )
        # move down a little
        tether.move(np.array([0, -dna_parameters.DNA_bond_length, 0], dtype=float))

        return cls([dna_strand], dna_parameters, tether, dna_safety_belt_index)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        strand = self.dna_strands[0]

        if par.force:
            ppp.stretching_forces_array[(par.force, 0, 0)] = [strand.first_id()]
            ppp.stretching_forces_array[(-par.force, 0, 0)] = [strand.last_id()]
        else:
            ppp.end_points += [strand.first_id(), strand.last_id()]

        ppp.dna_indices_list[0] = self.dna_indices_list_get_all_dna(0)

        # prevent breaking of safety belt
        # ppp.freeze_indices += [
        #     *[(self.dna_groups[0], self.dna_safety_belt_index + i) for i in range(-6, 6)],
        #     *[(self.smc.siteD_group, i) for i in range(len(self.smc.siteD_group.positions))]
        # ]

        return ppp

    def get_stopper_ids(self) -> list[StrandId]:
        return [(0, -1)]


@with_tether
class AdvancedObstacleSafety(DnaConfiguration):
    def __init__(
        self,
        dna_strands: list[Polymer],
        dna_parameters: DnaParameters,
        tether: Tether,
        dna_safety_belt_index: int,
    ):
        super().__init__(dna_strands, dna_parameters)
        self.tether = tether
        self.dna_safety_belt_index = dna_safety_belt_index

    @classmethod
    def get_dna_config(
        cls, dna_parameters: DnaParameters, r_lower_site, par
    ) -> AdvancedObstacleSafety:
        # 1.
        # [rDNA], belt_location, dna_safety_belt_index, dna_bead_to_tether_id = dna_creator.get_dna_coordinates_advanced_safety_belt(dna_parameters.nDNA, dna_parameters.DNAbondLength)
        [r_DNA], belt_location, dna_safety_belt_index, dna_bead_to_tether_id = (
            dna_creator.get_dna_coordinates_advanced_safety_belt_plus_loop(
                dna_parameters.nDNA, dna_parameters.DNA_bond_length
            )
        )

        # 2.
        # make sure SMC contains DNA
        shift = r_lower_site[1] - belt_location
        shift[1] -= 1.35 * par.cutoff6 + 0.5 * par.cutoff6  # TODO: if siteDup
        r_DNA += shift

        dna_strand = dna_parameters.create_dna_polymer([r_DNA])

        tether = Tether.create_tether(
            (0, dna_bead_to_tether_id),
            35,
            dna_parameters.DNA_bond_length,
            dna_parameters.DNA_mass / 2.0,
            dna_parameters.bond,
            dna_parameters.ssangle,
            Tether.Obstacle(),
        )
        obstacle = Tether.get_obstacle(True, cls.inter_par, tether.polymer.atom_groups[0])
        tether.obstacle = obstacle

        # place the tether next to the DNA bead
        tether.move(
            r_DNA[dna_bead_to_tether_id] - pos_from_id(tether.polymer.get_id_from_list_index(-1))
        )
        # move down a little
        tether.move(np.array([0, -dna_parameters.DNA_bond_length, 0], dtype=float))

        return cls([dna_strand], dna_parameters, tether, dna_safety_belt_index)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        strand = self.dna_strands[0]

        if par.force:
            ppp.stretching_forces_array[(par.force, 0, 0)] = [strand.first_id()]
            ppp.stretching_forces_array[(-par.force, 0, 0)] = [strand.last_id()]
        else:
            ppp.end_points += [strand.first_id(), strand.last_id()]

        ppp.dna_indices_list[0] = self.dna_indices_list_get_all_dna(0)

        # prevent breaking of safety belt
        # ppp.freeze_indices += [
        #     *[(self.dna_groups[0], self.dna_safety_belt_index + i) for i in range(-6, 6)],
        #     *[(self.smc.siteD_group, i) for i in range(len(self.smc.siteD_group.positions))]
        # ]

        return ppp

    def get_stopper_ids(self) -> list[StrandId]:
        return [(0, -1)]
