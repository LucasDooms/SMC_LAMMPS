# Copyright (c) 2024 Lucas Dooms

# File containing different initial DNA configurations

from __future__ import annotations
from typing import List, Tuple, Dict
from dataclasses import dataclass
from generator import BAI, BAI_Kind, BAI_Type, AtomGroup, AtomType, MoleculeId, AtomIdentifier, PairWise
from structures.dna import dna_creator
from structures import structure_creator
from structures.smc.smc import SMC
import numpy as np


def get_closest(array, position) -> int:
    """returns the index of the array that is closest to the given position"""
    distances = np.linalg.norm(array - position, axis=1)
    return int(np.argmin(distances))


@dataclass
class DnaParameters:
    nDNA: int
    DNAbondLength: float
    mDNA: float
    type: AtomType
    molDNA: int
    bond: BAI_Type
    angle: BAI_Type
    
    def create_dna(self, dna_positions) -> List[AtomGroup]:
        return [
            AtomGroup(
                positions=rDNA,
                atom_type=self.type,
                molecule_index=self.molDNA,
                polymer_bond_type=self.bond,
                polymer_angle_type=self.angle
            ) for rDNA in dna_positions
        ]


@dataclass
class InteractionParameters:
    ###########
    # DNA-DNA #
    ###########

    sigmaDNAvsDNA: float
    epsilonDNAvsDNA: float
    rcutDNAvsDNA: float


    ###########
    # SMC-DNA #
    ###########

    sigmaSMCvsDNA: float
    epsilonSMCvsDNA: float
    rcutSMCvsDNA: float


    #############
    # Sites-DNA #
    #############

    # Sigma of LJ attraction (same as those of the repulsive SMC sites)
    sigmaSiteDvsDNA: float

    # Cutoff distance of LJ attraction
    rcutSiteDvsDNA: float

    # Epsilon parameter of LJ attraction
    epsilonSiteDvsDNA: float


@dataclass
class Tether:

    class Obstacle:
        
        def move(self, vector) -> None:
            raise Exception("don't use Tether.Obstacle directly")

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
            pos = tether_group.positions[0] - np.array([0, obstacle_radius - ip.sigmaDNAvsDNA, 0], dtype=float)
            obstacle_type = AtomType(cls.get_gold_mass(obstacle_radius))
            obstacle_group = AtomGroup(
                positions=np.array([pos]),
                atom_type=obstacle_type,
                molecule_index=tether_group.molecule_index
            )

            obstacle_bond = BAI_Type(BAI_Kind.BOND, "fene/expand %s %s %s %s %s\n" %(1, obstacle_radius, 0, 0, ip.sigmaDNAvsDNA))
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
            ip.epsilonDNAvsDNA * kBT, ip.sigmaDNAvsDNA, ip.rcutDNAvsDNA
        )
        pair_inter.add_interaction(
            tether_type, dna_type,
            ip.epsilonDNAvsDNA * kBT, ip.sigmaDNAvsDNA, ip.rcutDNAvsDNA
        )
        pair_inter.add_interaction(
            tether_type, smc.armHK_type,
            ip.epsilonSMCvsDNA * kBT, ip.sigmaSMCvsDNA, ip.rcutSMCvsDNA
        )
        # Optional: don't allow bridge to go through tether
        # pair_inter.add_interaction(
        #     tether_type, smc.atp_type,
        #     ip.epsilonSMCvsDNA * kBT, ip.sigmaSMCvsDNA, ip.rcutSMCvsDNA
        # )
        # Optional: allow tether to bond to siteD
        pair_inter.add_interaction(
            tether_type, smc.siteD_type,
            ip.epsilonSiteDvsDNA * kBT, ip.sigmaSiteDvsDNA, ip.rcutSiteDvsDNA
        )
        if isinstance(self.obstacle, Tether.Gold):
            pair_inter.add_interaction(self.obstacle.group.type, dna_type, ip.epsilonDNAvsDNA * kBT, self.obstacle.radius, self.obstacle.cut)
            pair_inter.add_interaction(self.obstacle.group.type, smc.armHK_type, ip.epsilonDNAvsDNA * kBT, self.obstacle.radius, self.obstacle.cut)
            pair_inter.add_interaction(self.obstacle.group.type, tether_type, ip.epsilonDNAvsDNA * kBT, self.obstacle.radius, self.obstacle.cut)

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

    def get_all_groups(self) -> List[AtomGroup]:
        return self.dna_groups

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, rSiteD, par) -> DnaConfiguration:
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
        pair_inter.add_interaction(dna_type, dna_type, ip.epsilonDNAvsDNA * kBT, ip.sigmaDNAvsDNA, ip.rcutDNAvsDNA)
        pair_inter.add_interaction(dna_type, self.smc.armHK_type, ip.epsilonSMCvsDNA * kBT, ip.sigmaSMCvsDNA, ip.rcutSMCvsDNA)
        pair_inter.add_interaction(dna_type, self.smc.siteD_type, ip.epsilonSiteDvsDNA * kBT, ip.sigmaSiteDvsDNA, ip.rcutSiteDvsDNA)

    def get_bonds(self) -> List[BAI]:
        return []

    @staticmethod
    def str_to_config(string: str):
        string = string.lower()
        return {
            "line": Line,
            "folded": Folded,
            "right_angle": RightAngle,
            "doubled": Doubled,
            "safety_belt": SafetyBelt,
            "obstacle": Obstacle,
            "obstacle_safety": ObstacleSafety,
            "advanced_obstacle_safety": AdvancedObstacleSafety,
        }[string]

class Line(DnaConfiguration):

    def __init__(self, dna_groups, dna_parameters: DnaParameters):
        super().__init__(dna_groups, dna_parameters)

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, rSiteD, par) -> Line:
        default_dna_pos = rSiteD[1] + np.array([0, par.cutoff6, 0])

        # 1.
        [rDNA] = dna_creator.get_dna_coordinates_straight(dna_parameters.nDNA, dna_parameters.DNAbondLength)

        # 2.
        # make sure SMC contains DNA
        goal = default_dna_pos
        start = np.array([rDNA[int(len(rDNA) / 1.3)][0] + 10.0 * dna_parameters.DNAbondLength, rDNA[-1][1], 0])
        shift = (goal - start).reshape(1, 3)
        rDNA += shift

        return cls(dna_parameters.create_dna([rDNA]), dna_parameters)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        if par.force:
            ppp.stretching_forces_array[(par.force, 0, 0)] = [(self.dna_groups[0], 0)]
            ppp.stretching_forces_array[(-par.force, 0, 0)] = [(self.dna_groups[0], -1)]
        else:
            ppp.end_points += [(self.dna_groups[0], 0), (self.dna_groups[0], -1)]

        ppp.freeze_indices += [
            (self.dna_groups[0], get_closest(self.dna_groups[0].positions, self.smc.rSiteD[1])), # closest to bottom -> rSiteD[1]
            (self.dna_groups[0], get_closest(self.dna_groups[0].positions, self.smc.rSiteM[1])), # closest to middle -> rSiteM[1]
        ]

        ppp.dna_indices_list += self.dna_indices_list_get_all_dna()

        return ppp

class Folded(DnaConfiguration):

    def __init__(self, dna_groups, dna_parameters: DnaParameters, dnaCenter):
        super().__init__(dna_groups, dna_parameters)
        self.dnaCenter = dnaCenter

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, rSiteD, par) -> Folded:
        # place your DNA here, inside the SMC
        default_dna_pos = rSiteD[1] + np.array([0, par.cutoff6, 0])
        
        # 1.
        [rDNA], dnaCenter = dna_creator.get_dna_coordinates_twist(dna_parameters.nDNA, dna_parameters.DNAbondLength, 17)

        # 2.
        # make sure SMC contains DNA
        goal = default_dna_pos
        start = np.array([dnaCenter[0] + 40.0 * dna_parameters.DNAbondLength, rDNA[-1][1], 0])
        shift = (goal - start).reshape(1, 3)
        rDNA += shift

        return cls(dna_parameters.create_dna([rDNA]), dna_parameters, dnaCenter)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        if par.force:
            ppp.stretching_forces_array[(par.force, 0, 0)] = [(self.dna_groups[0], 0), (self.dna_groups[0], -1)]
        else:
            ppp.end_points += [(self.dna_groups[0], 0), (self.dna_groups[0], -1)]

        ppp.freeze_indices += [
            (self.dna_groups[0], get_closest(self.dna_groups[0].positions, self.smc.rSiteD[1])), # closest to bottom -> rSiteD[1]
            (self.dna_groups[0], get_closest(self.dna_groups[0].positions, self.smc.rSiteM[1])), # closest to middle -> rSiteM[1]
        ]

        ppp.dna_indices_list += self.dna_indices_list_get_dna_to(ratio=0.5)

        return ppp

class RightAngle(DnaConfiguration):

    def __init__(self, dna_groups, dna_parameters: DnaParameters, dnaCenter):
        super().__init__(dna_groups, dna_parameters)
        self.dnaCenter = dnaCenter

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, rSiteD, par) -> RightAngle:
        # place your DNA here, inside the SMC
        default_dna_pos = rSiteD[1] + np.array([0, par.cutoff6, 0])

        # 1.
        [rDNA], dnaCenter = dna_creator.get_dna_coordinates(dna_parameters.nDNA, dna_parameters.DNAbondLength, 14, 10)

        # 2.
        # make sure SMC touches the DNA at the lower site (siteD)
        goal = default_dna_pos
        start = np.array([dnaCenter[0] - 10.0 * dna_parameters.DNAbondLength, dnaCenter[1], 0])
        shift = (goal - start).reshape(1, 3)
        rDNA += shift

        return cls(dna_parameters.create_dna([rDNA]), dna_parameters, dnaCenter)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        if par.force:
            ppp.stretching_forces_array[(0, par.force, 0)] = [(self.dna_groups[0], 0)]
            ppp.stretching_forces_array[(-par.force, 0, 0)] = [(self.dna_groups[0], -1)]
        else:
            ppp.end_points += [(self.dna_groups[0], 0), (self.dna_groups[0], -1)]
        # find closest DNA bead to siteD
        # closest_DNA_index = get_closest(self.dna_groups[0].positions, rSiteD[1])

        ppp.dna_indices_list += self.dna_indices_list_get_dna_to(ratio=0.5)

        return ppp

class Doubled(DnaConfiguration):

    def __init__(self, dna_groups, dna_parameters: DnaParameters, dnaCenter):
        super().__init__(dna_groups, dna_parameters)
        self.dnaCenter = dnaCenter

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, rSiteD, par) -> Doubled:
        # place your DNA here, inside the SMC
        default_dna_pos = rSiteD[1] + np.array([0, 0.65 * par.cutoff6, 0])

        # 1.
        rDNAlist, dnaCenter = dna_creator.get_dna_coordinates_doubled(dna_parameters.nDNA, dna_parameters.DNAbondLength, 24)

        # 2.
        # make sure SMC contains DNA
        goal = default_dna_pos
        start = np.array([dnaCenter[0] + 30.0 * dna_parameters.DNAbondLength, rDNAlist[0][-1][1], 0])
        shift = (goal - start).reshape(1, 3)
        rDNAlist[0] += shift
        rDNAlist[1] += shift

        return cls(dna_parameters.create_dna(rDNAlist), dna_parameters, dnaCenter)

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
                (dna_grp, get_closest(dna_grp.positions, self.smc.rSiteD[1])), # closest to bottom
                (dna_grp, get_closest(dna_grp.positions, self.smc.rSiteM[1])), # closest to middle
            ]

        ppp.dna_indices_list += self.dna_indices_list_get_dna_to(ratio=0.5)

        return ppp

class SafetyBelt(DnaConfiguration):

    def __init__(self, dna_groups, dna_parameters: DnaParameters, dna_safety_belt_index: int):
        super().__init__(dna_groups, dna_parameters)
        self.dna_safety_belt_index = dna_safety_belt_index

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, rSiteD, par) -> SafetyBelt:
        # 1.
        [rDNA], belt_location, dna_safety_belt_index, _ = dna_creator.get_dna_coordinates_safety_belt(dna_parameters.nDNA, dna_parameters.DNAbondLength)

        # 2.
        # make sure SMC contains DNA
        shift = rSiteD[1] - belt_location
        shift[1] -= 0.65 * par.cutoff6 + 0.5 * par.cutoff6
        rDNA += shift

        dna_groups = dna_parameters.create_dna([rDNA])

        return cls(dna_groups, dna_parameters, dna_safety_belt_index)

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
class Obstacle(DnaConfiguration):

    def __init__(self, dna_groups, dna_parameters: DnaParameters, tether: Tether, dna_start_index: int):
        super().__init__(dna_groups, dna_parameters)
        self.tether = tether
        self.dna_start_index = dna_start_index

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, rSiteD, par) -> Obstacle:
        # place your DNA here, inside the SMC
        default_dna_pos = rSiteD[1] + np.array([0, par.cutoff6, 0])

        # 1.
        [rDNA] = dna_creator.get_dna_coordinates_straight(dna_parameters.nDNA, dna_parameters.DNAbondLength)
        
        # 2.
        # make sure SMC contains DNA
        goal = default_dna_pos
        dna_start_index = int(len(rDNA)*10.5/15)
        start = np.array([rDNA[dna_start_index][0] - 10.0 * dna_parameters.DNAbondLength, rDNA[dna_start_index][1], 0])
        shift = (goal - start).reshape(1, 3)
        rDNA += shift

        dna_groups = dna_parameters.create_dna([rDNA])

        dna_bead_to_tether_id = int(len(rDNA)*10/15)
        tether = Tether.create_tether(
            (dna_groups[0], dna_bead_to_tether_id), 25, dna_parameters.DNAbondLength, dna_parameters.mDNA, dna_parameters.bond, dna_parameters.angle, Tether.Obstacle()
        )
        obstacle = Tether.get_obstacle(par.obstacle_is_real, cls.inter_par, tether.group)
        tether.obstacle = obstacle
        # place the tether next to the DNA bead
        tether.move(rDNA[dna_bead_to_tether_id] - tether.group.positions[-1])
        # move down a little
        tether.move(np.array([0, -dna_parameters.DNAbondLength, 0], dtype=float))

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
    def get_dna_config(cls, dna_parameters: DnaParameters, rSiteD, par) -> ObstacleSafety:
        # 1.
        [rDNA], belt_location, dna_safety_belt_index, dna_bead_to_tether_id = dna_creator.get_dna_coordinates_safety_belt(dna_parameters.nDNA, dna_parameters.DNAbondLength)

        # 2.
        # make sure SMC contains DNA
        shift = rSiteD[1] - belt_location
        shift[1] -= 0.65 * par.cutoff6 + 0.5 * par.cutoff6 # TODO: if siteDup
        rDNA += shift

        dna_groups = dna_parameters.create_dna([rDNA])

        tether = Tether.create_tether((dna_groups[0], dna_bead_to_tether_id), 35, dna_parameters.DNAbondLength, dna_parameters.mDNA, dna_parameters.bond, dna_parameters.angle, Tether.Obstacle())
        obstacle = Tether.get_obstacle(par.obstacle_is_real, cls.inter_par, tether.group)
        tether.obstacle = obstacle

        tether.move(rDNA[dna_bead_to_tether_id] - tether.group.positions[-1])
        # move down a little
        tether.move(np.array([0, -dna_parameters.DNAbondLength, 0], dtype=float))

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
    def get_dna_config(cls, dna_parameters: DnaParameters, rSiteD, par) -> AdvancedObstacleSafety:
        # 1.
        # [rDNA], belt_location, dna_safety_belt_index, dna_bead_to_tether_id = dna_creator.get_dna_coordinates_advanced_safety_belt(dna_parameters.nDNA, dna_parameters.DNAbondLength)
        [rDNA], belt_location, dna_safety_belt_index, dna_bead_to_tether_id = dna_creator.get_dna_coordinates_advanced_safety_belt_plus_loop(dna_parameters.nDNA, dna_parameters.DNAbondLength)

        # 2.
        # make sure SMC contains DNA
        shift = rSiteD[1] - belt_location
        shift[1] -= 1.35 * par.cutoff6 + 0.5 * par.cutoff6 # TODO: if siteDup
        rDNA += shift

        dna_groups = dna_parameters.create_dna([rDNA])

        tether = Tether.create_tether((dna_groups[0], dna_bead_to_tether_id), 35, dna_parameters.DNAbondLength, dna_parameters.mDNA, dna_parameters.bond, dna_parameters.angle, Tether.Obstacle())
        obstacle = Tether.get_obstacle(par.obstacle_is_real, cls.inter_par, tether.group)
        tether.obstacle = obstacle

        # place the tether next to the DNA bead
        tether.move(rDNA[dna_bead_to_tether_id] - tether.group.positions[-1])
        # move down a little
        tether.move(np.array([0, -dna_parameters.DNAbondLength, 0], dtype=float))

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
