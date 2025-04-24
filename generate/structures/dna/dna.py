# Copyright (c) 2024 Lucas Dooms

# File containing different initial DNA configurations

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from generate.generator import (
    BAI,
    AtomGroup,
    AtomIdentifier,
    AtomType,
    BAI_Kind,
    BAI_Type,
    MoleculeId,
    PairWise,
)
from generate.structures import structure_creator
from generate.structures.dna import dna_creator
from generate.structures.smc.smc import SMC
from generate.util import get_closest


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
    
    def create_dna(self, dna_positions) -> List[AtomGroup]:
        return [
            AtomGroup(
                positions=r_DNA,
                atom_type=self.type,
                molecule_index=self.mol_DNA,
                polymer_bond_type=self.bond,
                polymer_angle_type=self.angle
            ) for r_DNA in dna_positions
        ]


@dataclass
class InteractionParameters:
    ###########
    # DNA-DNA #
    ###########

    sigma_DNA_DNA: float
    epsilon_DNA_DNA: float
    rcut_DNA_DNA: float


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

        def move(self, vector) -> None:
            raise NotImplementedError("don't use Tether.Obstacle directly")

    class Wall(Obstacle):
        
        def __init__(self, y_pos: float) -> None:
            super().__init__()
            self.y_pos = y_pos

        def move(self, vector) -> None:
            self.y_pos += vector[1]

    class Gold(Obstacle):
        
        def __init__(self, group: AtomGroup, radius: float, cut: float, tether_bond: BAI) -> None:
            super().__init__()
            self.group = group
            self.radius = radius
            self.cut = cut
            self.tether_bond = tether_bond

        def move(self, vector) -> None:
            self.group.positions[0] += vector

    group: AtomGroup
    dna_tether_id: AtomIdentifier
    obstacle: Tether.Obstacle

    @staticmethod
    def get_gold_mass(radius: float) -> float:
        """radius in nanometers, returns attograms"""
        density = 0.0193 # attograms per nanometer^3
        volume = 4.0/3.0 * np.pi * radius**3
        return density * volume

    @classmethod
    def get_obstacle(cls, real_obstacle: bool, ip: InteractionParameters, tether_group: AtomGroup) -> Tether.Obstacle:
        if real_obstacle:
            obstacle_radius = 100 # nanometers
            obstacle_cut = obstacle_radius * 2**(1/6)
            pos = tether_group.positions[0] - np.array([0, obstacle_radius - ip.sigma_DNA_DNA, 0], dtype=float)
            obstacle_type = AtomType(cls.get_gold_mass(obstacle_radius))
            obstacle_group = AtomGroup(
                positions=np.array([pos]),
                atom_type=obstacle_type,
                molecule_index=tether_group.molecule_index
            )

            obstacle_bond = BAI_Type(BAI_Kind.BOND, "fene/expand %s %s %s %s %s\n" %(1, obstacle_radius, 0, 0, ip.sigma_DNA_DNA))
            tether_obstacle_bond = BAI(obstacle_bond, (tether_group, 0), (obstacle_group, 0))
            return Tether.Gold(obstacle_group, obstacle_radius, obstacle_cut, tether_obstacle_bond)
        else:
            return Tether.Wall(tether_group.positions[0][1])

    @classmethod
    def create_tether(cls, dna_tether_id: AtomIdentifier, tether_length: int, bond_length: float, mass: float, bond_type: BAI_Type, angle_type: BAI_Type, obstacle: Tether.Obstacle) -> Tether:
        tether_positions = structure_creator.get_straight_segment(tether_length, [0, 1, 0]) * bond_length
        tether_group = AtomGroup(
            positions=tether_positions,
            atom_type=AtomType(mass),
            molecule_index=MoleculeId.get_next(),
            polymer_bond_type=bond_type,
            polymer_angle_type=angle_type
        )

        return Tether(
            group=tether_group,
            dna_tether_id=dna_tether_id,
            obstacle=obstacle
        )

    def move(self, vector) -> None:
        self.group.positions += vector
        self.obstacle.move(vector)

    def get_all_groups(self) -> List[AtomGroup]:
        groups = [self.group]
        if isinstance(self.obstacle, Tether.Gold):
            groups += [self.obstacle.group]
        return groups

    def handle_end_points(self, end_points: List[AtomIdentifier]) -> None:
        # freeze bottom of tether if using infinite wall
        if isinstance(self.obstacle, Tether.Wall):
            end_points += [(self.group, 0)]

    def add_interactions(self, pair_inter: PairWise, ip: InteractionParameters, dna_type: AtomType, smc: SMC, kBT: float) -> None:
        tether_type = self.group.type
        # tether
        pair_inter.add_interaction(
            tether_type, tether_type,
            ip.epsilon_DNA_DNA * kBT, ip.sigma_DNA_DNA, ip.rcut_DNA_DNA
        )
        pair_inter.add_interaction(
            tether_type, dna_type,
            ip.epsilon_DNA_DNA * kBT, ip.sigma_DNA_DNA, ip.rcut_DNA_DNA
        )
        pair_inter.add_interaction(
            tether_type, smc.t_arms_heads_kleisin,
            ip.epsilon_SMC_DNA * kBT, ip.sigma_SMC_DNA, ip.rcut_SMC_DNA
        )
        pair_inter.add_interaction(
            tether_type, smc.t_hinge,
            ip.epsilon_SMC_DNA * kBT, ip.sigma_SMC_DNA / 2.0, ip.rcut_SMC_DNA / 2.0
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
            pair_inter.add_interaction(self.obstacle.group.type, dna_type, ip.epsilon_DNA_DNA * kBT, self.obstacle.radius, self.obstacle.cut)
            pair_inter.add_interaction(self.obstacle.group.type, smc.t_arms_heads_kleisin, ip.epsilon_DNA_DNA * kBT, self.obstacle.radius, self.obstacle.cut)
            pair_inter.add_interaction(self.obstacle.group.type, smc.t_hinge, ip.epsilon_DNA_DNA * kBT, self.obstacle.radius, self.obstacle.cut)
            pair_inter.add_interaction(self.obstacle.group.type, tether_type, ip.epsilon_DNA_DNA * kBT, self.obstacle.radius, self.obstacle.cut)

    def get_bonds(self, bond_type: BAI_Type) -> List[BAI]:
        bonds = [BAI(bond_type, (self.group, -1), self.dna_tether_id)]
        if isinstance(self.obstacle, Tether.Gold):
            bonds += [self.obstacle.tether_bond]
        return bonds


# decorator to add tether logic to DnaConfiguration classes
def with_tether(cls):
    def new1(f):
        def get_all_groups(self) -> List[AtomGroup]:
            return f(self) + self.tether.get_all_groups()
        return get_all_groups
    cls.get_all_groups = new1(cls.get_all_groups)

    def new2(f):
        def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
            ppp = f(self)
            self.tether.handle_end_points(ppp.end_points)
            return ppp
        return get_post_process_parameters
    cls.get_post_process_parameters = new2(cls.get_post_process_parameters)

    def new3(f):
        def add_interactions(self, pair_inter: PairWise) -> None:
            f(self, pair_inter)
            self.tether.add_interactions(pair_inter, self.inter_par, self.dna_parameters.type, self.smc, self.par.kB * self.par.T)
        return add_interactions
    cls.add_interactions = new3(cls.add_interactions)

    def new4(f):
        def get_bonds(self) -> List[BAI]:
            return f(self) + self.tether.get_bonds(self.dna_parameters.bond)
        return get_bonds
    cls.get_bonds = new4(cls.get_bonds)

    return cls


class DnaConfiguration:

    @dataclass
    class PostProcessParameters:
        # LAMMPS DATA

        # indices to freeze permanently
        end_points: List[AtomIdentifier]
        # indices to temporarily freeze, in order to equilibrate the system
        freeze_indices: List[AtomIdentifier]
        # forces to apply:
        # the keys are the forces (3d vectors), and the value is a list of indices to which the force will be applied
        stretching_forces_array: Dict[Tuple[float, float, float], List[AtomIdentifier]]
        
        # POST PROCESSING

        # indices to use for marked bead tracking
        dna_indices_list: List[Tuple[AtomIdentifier, AtomIdentifier]]

    @classmethod
    def set_parameters(cls, par, inter_par: InteractionParameters) -> None:
        cls.par = par
        cls.inter_par = inter_par

    @classmethod
    def set_smc(cls, smc: SMC) -> None:
        cls.smc = smc

    def __init__(self, dna_groups: List[AtomGroup], dna_parameters: DnaParameters) -> None:
        self.dna_groups = dna_groups
        self.dna_parameters = dna_parameters
        self.kBT = self.par.kB * self.par.T
        self.beads: List[AtomGroup] = []
        self.bead_sizes: List[int] = []
        self.bead_bonds: List[BAI] = []

    def get_all_groups(self) -> List[AtomGroup]:
        return self.dna_groups + self.beads

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, r_lower_site, par) -> DnaConfiguration:
        return NotImplemented

    def get_post_process_parameters(self) -> PostProcessParameters:
        return self.PostProcessParameters(
            end_points=[],
            freeze_indices=[],
            stretching_forces_array=dict(),
            dna_indices_list=[]
        )

    def dna_indices_list_get_all_dna(self) -> List[Tuple[AtomIdentifier, AtomIdentifier]]:
        return [
            (
                (dna_grp, 0),
                (dna_grp, -1)
            )
            for dna_grp in self.dna_groups
        ]

    def dna_indices_list_get_dna_to(self, ratio: float) -> List[Tuple[AtomIdentifier, AtomIdentifier]]:
        return [
            (
                (dna_grp, 0),
                (dna_grp, int(len(dna_grp.positions) * ratio))
            )
            for dna_grp in self.dna_groups
        ]

    def add_interactions(self, pair_inter: PairWise) -> None:
        dna_type = self.dna_parameters.type
        ip = self.inter_par
        kBT = self.par.kB * self.par.T
        pair_inter.add_interaction(dna_type, dna_type, ip.epsilon_DNA_DNA * kBT, ip.sigma_DNA_DNA, ip.rcut_DNA_DNA)
        pair_inter.add_interaction(dna_type, self.smc.t_arms_heads_kleisin, ip.epsilon_SMC_DNA * kBT, ip.sigma_SMC_DNA, ip.rcut_SMC_DNA)
        pair_inter.add_interaction(dna_type, self.smc.t_hinge, ip.epsilon_SMC_DNA * kBT, ip.sigma_SMC_DNA, ip.rcut_SMC_DNA)
        pair_inter.add_interaction(dna_type, self.smc.t_lower_site, ip.epsilon_upper_site_DNA * kBT, ip.sigma_upper_site_DNA, ip.rcut_lower_site_DNA)

        # every bead should repel every SMC group
        for bead, bead_size, smc_grp in zip(self.beads, self.bead_sizes, self.smc.get_groups()):
            pair_inter.add_interaction(
                bead.type,
                smc_grp.type,
                ip.epsilon_SMC_DNA * kBT,
                bead_size * self.dna_parameters.DNA_bond_length,
                bead_size * self.dna_parameters.DNA_bond_length * (2 ** (1/6))
            )

    def get_bonds(self) -> List[BAI]:
        return self.bead_bonds

    def split_dna(self, split: AtomIdentifier) -> Tuple[AtomGroup, AtomGroup]:
        self.dna_groups.remove(split[0])
        pos1 = split[0].positions[:split[1]]
        pos2 = split[0].positions[split[1]:]

        args = (split[0].type, split[0].molecule_index, split[0].polymer_bond_type, split[0].polymer_angle_type)
        groups = (
            AtomGroup(pos1, *args),
            AtomGroup(pos2, *args),
        )
        self.dna_groups += groups
        return groups

    def add_bead_to_dna(
        self,
        bead_type: AtomType,
        mol_index: int,
        dna_atom: AtomIdentifier,
        bond: None | BAI_Type, # if None -> rigid attachment to dna_atom
        offset: int
    ) -> None:
        # place on a DNA bead
        location = dna_atom[0].positions[dna_atom[1]]

        # create a bead
        bead = AtomGroup([location], bead_type, mol_index)

        bais = []
        if bond is None:
            # TODO:
            # gen.molecule_override[dna_atom] = mol_index
            pass
        else:
            # add interactions/exceptions
            bais += [
                BAI(bond, (dna_atom[0], dna_atom[1] - offset), (bead, 0)),
                BAI(bond, (dna_atom[0], dna_atom[1] + offset), (bead, 0)),
            ]

        self.beads.append(bead)
        self.bead_sizes.append(offset)
        self.bead_bonds += bais

    @staticmethod
    def str_to_config(string: str):
        string = string.lower()
        return {
            "line": Line,
            "folded": Folded,
            "right_angle": RightAngle,
            "doubled": Doubled,
            "obstacle": Obstacle,
            "obstacle_safety": ObstacleSafety,
            "advanced_obstacle_safety": AdvancedObstacleSafety,
        }[string]

class Line(DnaConfiguration):

    """Straight line of DNA"""

    def __init__(self, dna_groups, dna_parameters: DnaParameters):
        super().__init__(dna_groups, dna_parameters)

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, r_lower_site, par) -> Line:
        default_dna_pos = r_lower_site[1] + np.array([0, par.cutoff6, 0])

        # 1.
        [r_DNA] = dna_creator.get_dna_coordinates_straight(dna_parameters.nDNA, dna_parameters.DNA_bond_length)

        # 2.
        # make sure SMC contains DNA
        goal = default_dna_pos
        start = np.array([r_DNA[int(len(r_DNA) / 1.3)][0] + 10.0 * dna_parameters.DNA_bond_length, r_DNA[-1][1], 0])
        shift = (goal - start).reshape(1, 3)
        r_DNA += shift

        return cls(dna_parameters.create_dna([r_DNA]), dna_parameters)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        if par.force:
            ppp.stretching_forces_array[(par.force, 0, 0)] = [(self.dna_groups[0], 0)]
            ppp.stretching_forces_array[(-par.force, 0, 0)] = [(self.dna_groups[0], -1)]
        else:
            ppp.end_points += [(self.dna_groups[0], 0), (self.dna_groups[0], -1)]

        ppp.freeze_indices += [
            (self.dna_groups[0], get_closest(self.dna_groups[0].positions, self.smc.r_lower_site[1])), # closest to bottom -> r_lower_site[1]
            (self.dna_groups[0], get_closest(self.dna_groups[0].positions, self.smc.r_middle_site[1])), # closest to middle -> r_lower_site[1]
        ]

        ppp.dna_indices_list += self.dna_indices_list_get_all_dna()

        return ppp

class Folded(DnaConfiguration):

    def __init__(self, dna_groups, dna_parameters: DnaParameters, dna_center):
        super().__init__(dna_groups, dna_parameters)
        self.dna_center = dna_center

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, r_lower_site, par) -> Folded:
        # place your DNA here, inside the SMC
        default_dna_pos = r_lower_site[1] + np.array([0, par.cutoff6, 0])

        # 1.
        [r_DNA], dna_center = dna_creator.get_dna_coordinates_twist(dna_parameters.nDNA, dna_parameters.DNA_bond_length, 17)

        # 2.
        # make sure SMC contains DNA
        goal = default_dna_pos
        start = np.array([dna_center[0] + 10.0 * dna_parameters.DNA_bond_length, r_DNA[-1][1], 0])
        shift = (goal - start).reshape(1, 3)
        r_DNA += shift

        return cls(dna_parameters.create_dna([r_DNA]), dna_parameters, dna_center)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        if par.force:
            ppp.stretching_forces_array[(par.force, 0, 0)] = [(self.dna_groups[0], 0), (self.dna_groups[0], -1)]
        else:
            ppp.end_points += [(self.dna_groups[0], 0), (self.dna_groups[0], -1)]

        ppp.freeze_indices += [
            (self.dna_groups[0], get_closest(self.dna_groups[0].positions, self.smc.r_lower_site[1])), # closest to bottom -> r_lower_site[1]
            (self.dna_groups[0], get_closest(self.dna_groups[0].positions, self.smc.r_middle_site[1])), # closest to middle -> r_lower_site[1]
        ]

        ppp.dna_indices_list += self.dna_indices_list_get_dna_to(ratio=0.5)

        return ppp

class RightAngle(DnaConfiguration):

    def __init__(self, dna_groups, dna_parameters: DnaParameters, dna_center):
        super().__init__(dna_groups, dna_parameters)
        self.dna_center = dna_center

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, r_lower_site, par) -> RightAngle:
        # place your DNA here, inside the SMC
        default_dna_pos = r_lower_site[1] + np.array([0, par.cutoff6, 0])

        # 1.
        [r_DNA], dna_center = dna_creator.get_dna_coordinates(dna_parameters.nDNA, dna_parameters.DNA_bond_length, 14, 10)

        # 2.
        # make sure SMC touches the DNA at the lower site (siteD)
        goal = default_dna_pos
        start = np.array([dna_center[0] - 10.0 * dna_parameters.DNA_bond_length, dna_center[1], 0])
        shift = (goal - start).reshape(1, 3)
        r_DNA += shift

        return cls(dna_parameters.create_dna([r_DNA]), dna_parameters, dna_center)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        if par.force:
            ppp.stretching_forces_array[(0, par.force, 0)] = [(self.dna_groups[0], 0)]
            ppp.stretching_forces_array[(-par.force, 0, 0)] = [(self.dna_groups[0], -1)]
        else:
            ppp.end_points += [(self.dna_groups[0], 0), (self.dna_groups[0], -1)]
        # find closest DNA bead to siteD
        # closest_DNA_index = get_closest(self.dna_groups[0].positions, r_lower_site[1])

        ppp.dna_indices_list += self.dna_indices_list_get_dna_to(ratio=0.5)

        return ppp

class Doubled(DnaConfiguration):

    def __init__(self, dna_groups, dna_parameters: DnaParameters, dna_center):
        super().__init__(dna_groups, dna_parameters)
        self.dna_center = dna_center

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, r_lower_site, par) -> Doubled:
        # place your DNA here, inside the SMC
        default_dna_pos = r_lower_site[1] + np.array([0, par.cutoff6, 0])

        # 1.
        r_DNA_list, dna_center = dna_creator.get_dna_coordinates_doubled(dna_parameters.nDNA, dna_parameters.DNA_bond_length, 24)

        # 2.
        # make sure SMC contains DNA
        goal = default_dna_pos
        start = np.array([dna_center[0] + 30.0 * dna_parameters.DNA_bond_length, r_DNA_list[0][-1][1], 0])
        shift = (goal - start).reshape(1, 3)
        r_DNA_list[0] += shift
        r_DNA_list[1] += shift

        return cls(dna_parameters.create_dna(r_DNA_list), dna_parameters, dna_center)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        # get dna beads to freeze
        for dna_grp in self.dna_groups:
            if par.force:
                ppp.stretching_forces_array[(par.force, 0, 0)] = [(dna_grp, 0), (dna_grp, -1)]
            else:
                ppp.end_points += [(dna_grp, 0), (dna_grp, -1)]
            # TODO: fix for DOUBLED DNA, gives same bead twice
            ppp.freeze_indices += [
                (dna_grp, get_closest(dna_grp.positions, self.smc.r_lower_site[1])), # closest to bottom
                (dna_grp, get_closest(dna_grp.positions, self.smc.r_middle_site[1])), # closest to middle
            ]

        ppp.dna_indices_list += self.dna_indices_list_get_dna_to(ratio=0.5)

        return ppp

@with_tether
class Obstacle(DnaConfiguration):

    def __init__(self, dna_groups, dna_parameters: DnaParameters, tether: Tether, dna_start_index: int):
        super().__init__(dna_groups, dna_parameters)
        self.tether = tether
        self.dna_start_index = dna_start_index

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, r_lower_site, par) -> Obstacle:
        # place your DNA here, inside the SMC
        default_dna_pos = r_lower_site[1] + np.array([0, par.cutoff6, 0])

        # 1.
        [r_DNA] = dna_creator.get_dna_coordinates_straight(dna_parameters.nDNA, dna_parameters.DNA_bond_length)

        # 2.
        # make sure SMC contains DNA
        goal = default_dna_pos
        dna_start_index = int(len(r_DNA)*9/15)
        start = np.array([r_DNA[dna_start_index][0] - 10.0 * dna_parameters.DNA_bond_length, r_DNA[dna_start_index][1], 0])
        shift = (goal - start).reshape(1, 3)
        r_DNA += shift

        dna_groups = dna_parameters.create_dna([r_DNA])

        dna_bead_to_tether_id = int(len(r_DNA)*7.5/15)
        tether = Tether.create_tether(
            (dna_groups[0], dna_bead_to_tether_id), 25, dna_parameters.DNA_bond_length, dna_parameters.DNA_mass, dna_parameters.bond, dna_parameters.ssangle, Tether.Obstacle()
        )
        obstacle = Tether.get_obstacle(True, cls.inter_par, tether.group)
        tether.obstacle = obstacle
        # place the tether next to the DNA bead
        tether.move(r_DNA[dna_bead_to_tether_id] - tether.group.positions[-1])
        # move down a little
        tether.move(np.array([0, -dna_parameters.DNA_bond_length, 0], dtype=float))

        return cls(dna_groups, dna_parameters, tether, dna_start_index)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        if par.force:
            ppp.stretching_forces_array[(par.force, 0, 0)] = [(self.dna_groups[0], 0)]
            ppp.stretching_forces_array[(-par.force, 0, 0)] = [(self.dna_groups[0], -1)]
        else:
            ppp.end_points += [(self.dna_groups[0], 0), (self.dna_groups[0], -1)]

        ppp.dna_indices_list += [
            (
                (dna_grp, 0),
                (dna_grp, self.dna_start_index)
            )
            for dna_grp in self.dna_groups
        ]

        return ppp

@with_tether
class ObstacleSafety(DnaConfiguration):

    def __init__(self, dna_groups, dna_parameters: DnaParameters, tether: Tether, dna_safety_belt_index: int):
        super().__init__(dna_groups, dna_parameters)
        self.tether = tether
        self.dna_safety_belt_index = dna_safety_belt_index

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, r_lower_site, par) -> ObstacleSafety:
        # 1.
        [r_DNA], belt_location, dna_safety_belt_index, dna_bead_to_tether_id = dna_creator.get_dna_coordinates_safety_belt(dna_parameters.nDNA, dna_parameters.DNA_bond_length)
        
        # 2.
        # make sure SMC contains DNA
        shift = r_lower_site[1] - belt_location
        shift[1] -= 0.65 * par.cutoff6 + 0.5 * par.cutoff6 # TODO: if siteDup
        r_DNA += shift

        dna_groups = dna_parameters.create_dna([r_DNA])

        tether = Tether.create_tether((dna_groups[0], dna_bead_to_tether_id), 35, dna_parameters.DNA_bond_length, dna_parameters.DNA_mass, dna_parameters.bond, dna_parameters.ssangle, Tether.Obstacle())
        obstacle = Tether.get_obstacle(True, cls.inter_par, tether.group)
        tether.obstacle = obstacle

        tether.move(r_DNA[dna_bead_to_tether_id] - tether.group.positions[-1])
        # move down a little
        tether.move(np.array([0, -dna_parameters.DNA_bond_length, 0], dtype=float))

        return cls(dna_groups, dna_parameters, tether, dna_safety_belt_index)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        if par.force:
            ppp.stretching_forces_array[(par.force, 0, 0)] = [(self.dna_groups[0], 0)]
            ppp.stretching_forces_array[(-par.force, 0, 0)] = [(self.dna_groups[0], -1)]
        else:
            ppp.end_points += [(self.dna_groups[0], 0), (self.dna_groups[0], -1)]

        ppp.dna_indices_list += self.dna_indices_list_get_all_dna()

        # prevent breaking of safety belt
        # ppp.freeze_indices += [
        #     *[(self.dna_groups[0], self.dna_safety_belt_index + i) for i in range(-6, 6)],
        #     *[(self.smc.siteD_group, i) for i in range(len(self.smc.siteD_group.positions))]
        # ]

        return ppp

@with_tether
class AdvancedObstacleSafety(DnaConfiguration):

    def __init__(self, dna_groups, dna_parameters: DnaParameters, tether: Tether, dna_safety_belt_index: int):
        super().__init__(dna_groups, dna_parameters)
        self.tether = tether
        self.dna_safety_belt_index = dna_safety_belt_index

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, r_lower_site, par) -> AdvancedObstacleSafety:
        # 1.
        # [rDNA], belt_location, dna_safety_belt_index, dna_bead_to_tether_id = dna_creator.get_dna_coordinates_advanced_safety_belt(dna_parameters.nDNA, dna_parameters.DNAbondLength)
        [r_DNA], belt_location, dna_safety_belt_index, dna_bead_to_tether_id = dna_creator.get_dna_coordinates_advanced_safety_belt_plus_loop(dna_parameters.nDNA, dna_parameters.DNA_bond_length)

        # 2.
        # make sure SMC contains DNA
        shift = r_lower_site[1] - belt_location
        shift[1] -= 1.35 * par.cutoff6 + 0.5 * par.cutoff6 # TODO: if siteDup
        r_DNA += shift

        dna_groups = dna_parameters.create_dna([r_DNA])

        tether = Tether.create_tether((dna_groups[0], dna_bead_to_tether_id), 35, dna_parameters.DNA_bond_length, dna_parameters.DNA_mass, dna_parameters.bond, dna_parameters.ssangle, Tether.Obstacle())
        obstacle = Tether.get_obstacle(True, cls.inter_par, tether.group)
        tether.obstacle = obstacle

        # place the tether next to the DNA bead
        tether.move(r_DNA[dna_bead_to_tether_id] - tether.group.positions[-1])
        # move down a little
        tether.move(np.array([0, -dna_parameters.DNA_bond_length, 0], dtype=float))

        return cls(dna_groups, dna_parameters, tether, dna_safety_belt_index)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        if par.force:
            ppp.stretching_forces_array[(par.force, 0, 0)] = [(self.dna_groups[0], 0)]
            ppp.stretching_forces_array[(-par.force, 0, 0)] = [(self.dna_groups[0], -1)]
        else:
            ppp.end_points += [(self.dna_groups[0], 0), (self.dna_groups[0], -1)]

        ppp.dna_indices_list += self.dna_indices_list_get_all_dna()

        # prevent breaking of safety belt
        # ppp.freeze_indices += [
        #     *[(self.dna_groups[0], self.dna_safety_belt_index + i) for i in range(-6, 6)],
        #     *[(self.smc.siteD_group, i) for i in range(len(self.smc.siteD_group.positions))]
        # ]

        return ppp
