from __future__ import annotations
from typing import List, Tuple, Dict
from dataclasses import dataclass
from generator import BAI_Type, AtomGroup, AtomType, MoleculeId, AtomIdentifier
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
    dna_type: AtomType
    molDNA: int
    dna_bond: BAI_Type
    dna_angle: BAI_Type
    
    def create_dna(self, dna_positions) -> List[AtomGroup]:
        return [
            AtomGroup(
                positions=rDNA,
                atom_type=self.dna_type,
                molecule_index=self.molDNA,
                polymer_bond_type=self.dna_bond,
                polymer_angle_type=self.dna_angle
            ) for rDNA in dna_positions
        ]


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
    def set_parameters(cls, par) -> None:
        cls.par = par

    @classmethod
    def set_smc(cls, smc: SMC) -> None:
        cls.smc = smc

    def __init__(self, dna_groups: List[AtomGroup], dna_parameters: DnaParameters) -> None:
        self.dna_groups = dna_groups
        self.dna_parameters = dna_parameters

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
            "safety_loop": SafetyLoop,
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
        start = np.array([dnaCenter[0] + 10.0 * dna_parameters.DNAbondLength, rDNA[-1][1], 0])
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
        default_dna_pos = rSiteD[1] + np.array([0, par.cutoff6, 0])

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

class Obstacle(DnaConfiguration):

    def __init__(self, dna_groups, dna_parameters: DnaParameters, tether_group: AtomGroup, dna_tether_id: int, dna_start_index: int):
        super().__init__(dna_groups, dna_parameters)
        self.tether_group = tether_group
        self.dna_tether_id = dna_tether_id
        self.dna_start_index = dna_start_index

    def get_all_groups(self) -> List[AtomGroup]:
        return super().get_all_groups() + [self.tether_group]
    
    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, rSiteD, par) -> Obstacle:
        # place your DNA here, inside the SMC
        default_dna_pos = rSiteD[1] + np.array([0, par.cutoff6, 0])

        # 1.
        [rDNA] = dna_creator.get_dna_coordinates_straight(dna_parameters.nDNA, dna_parameters.DNAbondLength)
        
        # 2.
        # make sure SMC contains DNA
        goal = default_dna_pos
        dna_start_index = int(len(rDNA)*8.2/15)
        start = np.array([rDNA[dna_start_index][0] - 10.0 * dna_parameters.DNAbondLength, rDNA[dna_start_index][1], 0])
        shift = (goal - start).reshape(1, 3)
        rDNA += shift

        obstacle_length = 45
        tether_positions = structure_creator.get_straight_segment(obstacle_length, [0, 1, 0]) * dna_parameters.DNAbondLength
        # place the tether next to the DNA bead
        dna_bead_to_tether_id = int(len(rDNA)*1/2)
        tether_positions += rDNA[dna_bead_to_tether_id] - tether_positions[-1]
        # move down a little
        tether_positions += np.array([0, -dna_parameters.DNAbondLength, 0], dtype=float)

        tether_group = AtomGroup(
            positions=tether_positions,
            atom_type=AtomType(dna_parameters.mDNA),
            molecule_index=MoleculeId.get_next(),
            polymer_bond_type=dna_parameters.dna_bond,
            polymer_angle_type=dna_parameters.dna_angle
        )

        return cls(dna_parameters.create_dna([rDNA]), dna_parameters, tether_group, dna_bead_to_tether_id, dna_start_index)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        if par.force:
            ppp.stretching_forces_array[(par.force, 0, 0)] = [(self.dna_groups[0], 0)]
            ppp.stretching_forces_array[(-par.force, 0, 0)] = [(self.dna_groups[0], -1)]
        ppp.end_points += [(self.tether_group, 0)]

        ppp.dna_indices_list += [
            (
                (dna_grp, 0),
                (dna_grp, self.dna_start_index)
            )
            for dna_grp in self.dna_groups
        ]

        return ppp

class ObstacleSafety(DnaConfiguration):

    def __init__(self, dna_groups, dna_parameters: DnaParameters, tether_group: AtomGroup, dna_tether_id: int, dna_safety_belt_index: int):
        super().__init__(dna_groups, dna_parameters)
        self.tether_group = tether_group
        self.dna_tether_id = dna_tether_id
        self.dna_safety_belt_index = dna_safety_belt_index

    def get_all_groups(self) -> List[AtomGroup]:
        return super().get_all_groups() + [self.tether_group]
    
    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, rSiteD, par) -> ObstacleSafety:
        # 1.
        [rDNA], belt_location, dna_safety_belt_index = dna_creator.get_dna_coordinates_safety_belt(dna_parameters.nDNA, dna_parameters.DNAbondLength)
        
        # 2.
        # make sure SMC contains DNA
        shift = rSiteD[1] - belt_location
        shift[1] -= 0.65 * par.cutoff6 
        rDNA += shift

        tether_positions = structure_creator.get_straight_segment(35, [0, 1, 0]) * dna_parameters.DNAbondLength
        # place the tether next to the DNA bead
        dna_bead_to_tether_id = int(len(rDNA) / 3.5)
        tether_positions += rDNA[dna_bead_to_tether_id] - tether_positions[-1]
        # move down a little
        tether_positions += np.array([0, -dna_parameters.DNAbondLength, 0], dtype=float)

        tether_group = AtomGroup(
            positions=tether_positions,
            atom_type=AtomType(dna_parameters.mDNA),
            molecule_index=MoleculeId.get_next(),
            polymer_bond_type=dna_parameters.dna_bond,
            polymer_angle_type=dna_parameters.dna_angle
        )

        return cls(dna_parameters.create_dna([rDNA]), dna_parameters, tether_group, dna_bead_to_tether_id, dna_safety_belt_index)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        if par.force:
            ppp.stretching_forces_array[(par.force, 0, 0)] = [(self.dna_groups[0], 0)]
            ppp.stretching_forces_array[(-par.force, 0, 0)] = [(self.dna_groups[0], -1)]
        ppp.end_points += [(self.tether_group, 0)]

        ppp.dna_indices_list += self.dna_indices_list_get_all_dna()

        return ppp

class AdvancedObstacleSafety(DnaConfiguration):

    def __init__(self, dna_groups, dna_parameters: DnaParameters, tether_group: AtomGroup, dna_tether_id: int, dna_safety_belt_index: int):
        super().__init__(dna_groups, dna_parameters)
        self.tether_group = tether_group
        self.dna_tether_id = dna_tether_id
        self.dna_safety_belt_index = dna_safety_belt_index

    def get_all_groups(self) -> List[AtomGroup]:
        return super().get_all_groups() + [self.tether_group]
    
    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, rSiteD, par) -> AdvancedObstacleSafety:
        # 1.
        # [rDNA], belt_location, dna_safety_belt_index, dna_bead_to_tether_id = dna_creator.get_dna_coordinates_advanced_safety_belt(dna_parameters.nDNA, dna_parameters.DNAbondLength)
        [rDNA], belt_location, dna_safety_belt_index, dna_bead_to_tether_id = dna_creator.get_dna_coordinates_advanced_safety_belt_plus_loop(dna_parameters.nDNA, dna_parameters.DNAbondLength)

        # 2.
        # make sure SMC contains DNA
        shift = rSiteD[1] - belt_location
        shift[1] -= 1.35 * par.cutoff6 
        rDNA += shift

        tether_positions = structure_creator.get_straight_segment(35, [0, 1, 0]) * dna_parameters.DNAbondLength
        # place the tether next to the DNA bead
        tether_positions += rDNA[dna_bead_to_tether_id] - tether_positions[-1]
        # move down a little
        tether_positions += np.array([0, -dna_parameters.DNAbondLength, 0], dtype=float)

        tether_group = AtomGroup(
            positions=tether_positions,
            atom_type=AtomType(dna_parameters.mDNA),
            molecule_index=MoleculeId.get_next(),
            polymer_bond_type=dna_parameters.dna_bond,
            polymer_angle_type=dna_parameters.dna_angle
        )

        return cls(dna_parameters.create_dna([rDNA]), dna_parameters, tether_group, dna_bead_to_tether_id, dna_safety_belt_index)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        if par.force:
            ppp.stretching_forces_array[(par.force, 0, 0)] = [(self.dna_groups[0], 0)]
            ppp.stretching_forces_array[(-par.force, 0, 0)] = [(self.dna_groups[0], -1)]
        ppp.end_points += [(self.tether_group, 0)]

        ppp.dna_indices_list += self.dna_indices_list_get_all_dna()

        return ppp

class SafetyLoop(DnaConfiguration):

    def __init__(self, dna_groups, dna_parameters: DnaParameters, dna_safety_belt_index: int):
        super().__init__(dna_groups, dna_parameters)
        self.dna_safety_belt_index = dna_safety_belt_index

    @classmethod
    def get_dna_config(cls, dna_parameters: DnaParameters, rSiteD, par) -> SafetyLoop:
        # 1.
        [rDNA], belt_location, dna_safety_belt_index = dna_creator.get_dna_coordinates_safety_loop(dna_parameters.nDNA, dna_parameters.DNAbondLength)

        # 2.
        # make sure SMC contains DNA
        shift = rSiteD[1] - belt_location
        shift[1] -= par.cutoff6 
        rDNA += shift

        return cls(dna_parameters.create_dna([rDNA]), dna_parameters, dna_safety_belt_index)

    def get_post_process_parameters(self) -> DnaConfiguration.PostProcessParameters:
        ppp = super().get_post_process_parameters()
        par = self.par

        if par.force:
            ppp.stretching_forces_array[(par.force, 0, 0)] = [(self.dna_groups[0], 0)]
            ppp.stretching_forces_array[(0, par.force, 0)] = [(self.dna_groups[0], -1)]
        else:
            ppp.end_points += [(self.dna_groups[0], 0), (self.dna_groups[0], -1)]

        ppp.dna_indices_list += self.dna_indices_list_get_all_dna()

        return ppp
