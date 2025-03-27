# Copyright (c) 2024 Lucas Dooms

from generator import AtomType, AtomGroup, BAI_Type, BAI
from dataclasses import dataclass
from typing import List


@dataclass
class SMC:

    rArmDL : ...
    rArmUL : ...
    rArmUR : ...
    rArmDR : ...
    rATP : ...
    rHK  : ...
    rSiteU : ...
    rSiteM : ...
    rSiteD : ...
    rHinge : ...

    molArmDL: int
    molArmUL: int
    molArmUR: int
    molArmDR: int
    molHK   : int
    molATP  : int
    molHingeL: int
    molHingeR: int
    molSiteM: int
    molSiteD: int

    armHK_type : AtomType
    hinge_type: AtomType
    atp_type : AtomType
    siteU_type : AtomType
    siteM_type : AtomType
    siteD_type : AtomType
    refSite_type : AtomType

    def __post_init__(self):
        # create groups
        self.armDL_group = AtomGroup(self.rArmDL, self.armHK_type, self.molArmDL)
        self.armUL_group = AtomGroup(self.rArmUL, self.armHK_type, self.molArmUL)
        self.armUR_group = AtomGroup(self.rArmUR, self.armHK_type, self.molArmUR)
        self.armDR_group = AtomGroup(self.rArmDR, self.armHK_type, self.molArmDR)
        self.hk_group = AtomGroup(self.rHK, self.armHK_type, self.molHK)

        self.atp_group = AtomGroup(self.rATP, self.atp_type, self.molATP)

        self.hingeL_group = AtomGroup(self.rHinge[:len(self.rHinge) // 2], self.hinge_type, self.molHingeL)
        self.hingeR_group = AtomGroup(self.rHinge[len(self.rHinge) // 2:], self.hinge_type, self.molHingeR)
        self.siteU_group = AtomGroup(self.rSiteU, self.siteU_type, self.molHingeL)

        # split M in three parts

        cut = 2
        self.siteM_group = AtomGroup(self.rSiteM[:cut], self.siteM_type, self.molSiteM)
        self.siteM_atp_group = AtomGroup(self.rSiteM[cut:-1], self.atp_type, self.molSiteM)
        # ref site
        self.siteM_ref_group = AtomGroup(self.rSiteM[-1:], self.refSite_type, self.molSiteM)

        # split B in two parts

        cut = 3
        self.siteD_group = AtomGroup(self.rSiteD[:cut], self.siteD_type, self.molSiteD)
        self.siteD_arm_group = AtomGroup(self.rSiteD[cut:], self.armHK_type, self.molSiteD)

    def get_groups(self) -> List[AtomGroup]:
        return [
            self.armDL_group,
            self.armUL_group,
            self.armUR_group,
            self.armDR_group,
            self.hk_group,
            self.atp_group,
            self.siteU_group,
            self.siteM_group,
            self.siteM_atp_group,
            self.siteM_ref_group,
            self.siteD_group,
            self.siteD_arm_group,
            self.hingeL_group,
            self.hingeR_group,
        ]

    def get_bonds(self, bond_t2: BAI_Type, bond_t3: BAI_Type) -> List[BAI]:
        left_attach_hinge = len(self.hingeL_group.positions) // 2
        right_attach_hinge = len(self.hingeR_group.positions) // 2
        return [
            # attach arms together
            BAI(bond_t2, (self.armDL_group, -1), (self.armUL_group, 0)),
            BAI(bond_t2, (self.armUR_group, -1), (self.armDR_group, 0)),
            # connect hinge and arms
            BAI(bond_t2, (self.armUL_group, -1), (self.hingeL_group, left_attach_hinge)),
            BAI(bond_t2, (self.armUR_group, 0), (self.hingeR_group, right_attach_hinge)),
            # connect Left and Right hinge pieces together
            BAI(bond_t3, (self.hingeL_group, -1), (self.hingeR_group, 0)),
            BAI(bond_t3, (self.hingeL_group, 0), (self.hingeR_group, -1)),
            # connect atp bridge to arms
            BAI(bond_t2, (self.armDR_group, -1), (self.atp_group, -1)),
            BAI(bond_t2, (self.atp_group,  0), (self.armDL_group, 0)),
            # connect bridge to hk
            BAI(bond_t2, (self.atp_group, -1), (self.hk_group, 0)),
            BAI(bond_t2, (self.hk_group, -1), (self.atp_group, 0)),
        ]

    def get_angles(self, angle_t2: BAI_Type, angle_t3: BAI_Type, angle_t4: BAI_Type) -> List[BAI]:
        left_attach_hinge = len(self.hingeL_group.positions) // 2
        right_attach_hinge = len(self.hingeR_group.positions) // 2
        return [
            # keep left arms rigid (prevent too much bending)
            BAI(angle_t2, (self.armDL_group, 0), (self.armUL_group, 0), (self.armUL_group, -1)),
            # same, but for right arms
            BAI(angle_t2, (self.armUR_group, 0), (self.armUR_group, -1), (self.armDR_group, -1)),

            # keep hinge perpendicular to arms
            BAI(angle_t4, (self.armUL_group, -2), (self.armUL_group, -1), (self.hingeL_group, left_attach_hinge + 1)),
            BAI(angle_t4, (self.armUL_group, -2), (self.armUL_group, -1), (self.hingeL_group, left_attach_hinge - 1)),

            BAI(angle_t4, (self.armUR_group, 1), (self.armUR_group, 0), (self.hingeR_group, right_attach_hinge - 1)),
            BAI(angle_t4, (self.armUR_group, 1), (self.armUR_group, 0), (self.hingeR_group, right_attach_hinge + 1)),

            # prevent too much bending between lower arms and the bridge
            BAI(angle_t3, (self.armDL_group, -1), (self.armDL_group, 0), (self.atp_group, -1)),
            BAI(angle_t3, (self.armDR_group, 0), (self.armDR_group, -1), (self.atp_group, 0))
        ]

    def get_impropers(self, imp_t1: BAI_Type, imp_t2: BAI_Type, imp_t3: BAI_Type, imp_t4: BAI_Type) -> List[BAI]:
        nHK = len(self.rHK)
        left_attach_hinge = len(self.hingeL_group.positions) // 2
        right_attach_hinge = len(self.hingeR_group.positions) // 2
        return [
            # Fix orientation of ATP/kleisin bridge
            # WARNING: siteM is split into groups, be careful with index
            BAI(imp_t1, (self.armDL_group, -1), (self.armDL_group, 0), (self.atp_group, -1), (self.siteM_group, 1)),
            BAI(imp_t1, (self.armDR_group, 0), (self.armDR_group, -1), (self.atp_group, 0), (self.siteM_group, 1)),

            BAI(imp_t2, (self.armDL_group, -1), (self.armDL_group, 0), (self.atp_group, -1), (self.hk_group, nHK//2)),
            BAI(imp_t2, (self.armDR_group, 0), (self.armDR_group, -1), (self.atp_group, 0), (self.hk_group, nHK//2)),

            # prevent kleisin ring from swaying too far relative to the bridge
            BAI(imp_t3, (self.siteM_ref_group, 0), (self.armDL_group, 0), (self.armDR_group, -1), (self.hk_group, nHK//2)),

            # fix hinge to a plane
            BAI(imp_t1, (self.hingeL_group, 0), (self.hingeL_group, -1), (self.hingeR_group, 0), (self.hingeR_group, -1)),
            # fix hinge perpendicular to arms plane
            BAI(imp_t4, (self.armUL_group, -2), (self.armUL_group, -1), (self.hingeL_group, left_attach_hinge - 1), (self.hingeL_group, left_attach_hinge + 1)),
            BAI(imp_t4, (self.armUR_group, 1), (self.armUR_group, 0), (self.hingeR_group, right_attach_hinge + 1), (self.hingeR_group, right_attach_hinge - 1)),
        ]
