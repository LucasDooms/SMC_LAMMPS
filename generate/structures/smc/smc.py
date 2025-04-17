# Copyright (c) 2024 Lucas Dooms

from generator import AtomType, AtomGroup, BAI_Kind, BAI_Type, BAI, MoleculeId
from dataclasses import dataclass
from typing import List


@dataclass
class SMC:

    use_rigid_hinge: bool

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

    armHK_type : AtomType
    hinge_type: AtomType
    atp_type : AtomType
    siteU_type : AtomType
    siteM_type : AtomType
    siteD_type : AtomType
    refSite_type : AtomType

    k_bond: float
    max_bond_length: float

    def set_molecule_ids(self, use_rigid_hinge: bool) -> None:
        # Molecule for each rigid body
        self.mol_arm_dl = MoleculeId.get_next()
        self.mol_arm_ul = MoleculeId.get_next()
        self.mol_arm_ur = MoleculeId.get_next()
        self.mol_arm_dr = MoleculeId.get_next()
        self.mol_heads_kleisin = MoleculeId.get_next()
        self.mol_ATP = MoleculeId.get_next()

        self.mol_hinge_l = MoleculeId.get_next()
        if use_rigid_hinge:
            self.mol_hinge_r = self.mol_hinge_l
        else:
            self.mol_hinge_r = MoleculeId.get_next()

        self.mol_middle_site = self.mol_ATP
        self.mol_lower_site = self.mol_heads_kleisin

    def get_molecule_ids(self) -> List[int]:
        return [
            self.mol_arm_dl,
            self.mol_arm_ul,
            self.mol_arm_ur,
            self.mol_arm_dr,
            self.mol_heads_kleisin,
            self.mol_ATP,
            self.mol_hinge_l,
            self.mol_hinge_r,
            self.mol_middle_site,
            self.mol_lower_site,
        ]

    def __post_init__(self):
        self.set_molecule_ids(self.use_rigid_hinge)
        # create groups
        self.arm_dl_grp = AtomGroup(self.rArmDL, self.armHK_type, self.mol_arm_dl)
        self.arm_ul_grp = AtomGroup(self.rArmUL, self.armHK_type, self.mol_arm_ul)
        self.arm_ur_grp = AtomGroup(self.rArmUR, self.armHK_type, self.mol_arm_ur)
        self.arm_dr_grp = AtomGroup(self.rArmDR, self.armHK_type, self.mol_arm_dr)
        self.hk_grp = AtomGroup(self.rHK, self.armHK_type, self.mol_heads_kleisin)

        self.atp_grp = AtomGroup(self.rATP, self.atp_type, self.mol_ATP)

        self.hinge_l_grp = AtomGroup(self.rHinge[:len(self.rHinge) // 2], self.hinge_type, self.mol_hinge_l)
        self.hinge_r_grp = AtomGroup(self.rHinge[len(self.rHinge) // 2:], self.hinge_type, self.mol_hinge_r)
        self.upper_site_grp = AtomGroup(self.rSiteU, self.siteU_type, self.mol_hinge_l)

        # split M in three parts

        cut = 2
        self.middle_site_grp = AtomGroup(self.rSiteM[:cut], self.siteM_type, self.mol_middle_site)
        self.middle_site_atp_grp = AtomGroup(self.rSiteM[cut:-1], self.atp_type, self.mol_middle_site)
        # ref site
        self.middle_site_ref_grp = AtomGroup(self.rSiteM[-1:], self.refSite_type, self.mol_middle_site)

        # split B in two parts

        cut = 3
        self.lower_site_grp = AtomGroup(self.rSiteD[:cut], self.siteD_type, self.mol_lower_site)
        self.lower_site_arm_grp = AtomGroup(self.rSiteD[cut:], self.armHK_type, self.mol_lower_site)

    def get_groups(self) -> List[AtomGroup]:
        return [
            self.arm_dl_grp,
            self.arm_ul_grp,
            self.arm_ur_grp,
            self.arm_dr_grp,
            self.hk_grp,
            self.atp_grp,
            self.upper_site_grp,
            self.middle_site_grp,
            self.middle_site_atp_grp,
            self.middle_site_ref_grp,
            self.lower_site_grp,
            self.lower_site_arm_grp,
            self.hinge_l_grp,
            self.hinge_r_grp,
        ]

    def get_bonds(self, hinge_opening: float | None = None) -> List[BAI]:
        # Every joint is kept in place through bonds
        attach = BAI_Type(BAI_Kind.BOND, f"fene/expand {self.k_bond} {self.max_bond_length} {0.0} {0.0} {0.0}\n")

        left_attach_hinge = len(self.hinge_l_grp.positions) // 2
        right_attach_hinge = len(self.hinge_r_grp.positions) // 2
        bonds = [
            # attach arms together
            BAI(attach, (self.arm_dl_grp, -1), (self.arm_ul_grp, 0)),
            BAI(attach, (self.arm_ur_grp, -1), (self.arm_dr_grp, 0)),
            # attach hinge and arms
            BAI(attach, (self.arm_ul_grp, -1), (self.hinge_l_grp, left_attach_hinge)),
            BAI(attach, (self.arm_ur_grp, 0), (self.hinge_r_grp, right_attach_hinge)),
            # attach atp bridge to arms
            BAI(attach, (self.arm_dr_grp, -1), (self.atp_grp, -1)),
            BAI(attach, (self.atp_grp,  0), (self.arm_dl_grp, 0)),
            # attach bridge to hk
            BAI(attach, (self.atp_grp, -1), (self.hk_grp, 0)),
            BAI(attach, (self.hk_grp, -1), (self.atp_grp, 0)),
        ]
        # work-around for crash caused by
        # `bond_style     hybrid fene/expand harmonic`
        # in input.lmp
        # always add bond for now, even if it is rigid
        if not self.use_rigid_hinge or True:
            assert hinge_opening is not None
            hinge_bond = BAI_Type(BAI_Kind.BOND, f"harmonic {self.k_bond} {hinge_opening}\n")
            bonds += [
                # connect Left and Right hinge pieces together
                BAI(hinge_bond, (self.hinge_l_grp, -1), (self.hinge_r_grp, 0)),
                BAI(hinge_bond, (self.hinge_l_grp, 0), (self.hinge_r_grp, -1)),
            ]
        return bonds

    def get_angles(self, angle_t2: BAI_Type, angle_t3: BAI_Type, angle_t4: BAI_Type) -> List[BAI]:
        left_attach_hinge = len(self.hinge_l_grp.positions) // 2
        right_attach_hinge = len(self.hinge_r_grp.positions) // 2
        return [
            # keep left arms rigid (prevent too much bending)
            BAI(angle_t2, (self.arm_dl_grp, 0), (self.arm_ul_grp, 0), (self.arm_ul_grp, -1)),
            # same, but for right arms
            BAI(angle_t2, (self.arm_ur_grp, 0), (self.arm_ur_grp, -1), (self.arm_dr_grp, -1)),

            # keep hinge perpendicular to arms
            BAI(angle_t4, (self.arm_ul_grp, -2), (self.arm_ul_grp, -1), (self.hinge_l_grp, left_attach_hinge + 1)),
            BAI(angle_t4, (self.arm_ul_grp, -2), (self.arm_ul_grp, -1), (self.hinge_l_grp, left_attach_hinge - 1)),

            BAI(angle_t4, (self.arm_ur_grp, 1), (self.arm_ur_grp, 0), (self.hinge_r_grp, right_attach_hinge - 1)),
            BAI(angle_t4, (self.arm_ur_grp, 1), (self.arm_ur_grp, 0), (self.hinge_r_grp, right_attach_hinge + 1)),

            # prevent too much bending between lower arms and the bridge
            BAI(angle_t3, (self.arm_dl_grp, -1), (self.arm_dl_grp, 0), (self.atp_grp, -1)),
            BAI(angle_t3, (self.arm_dr_grp, 0), (self.arm_dr_grp, -1), (self.atp_grp, 0))
        ]

    def get_impropers(self, imp_t1: BAI_Type, imp_t2: BAI_Type, imp_t3: BAI_Type, imp_t4: BAI_Type) -> List[BAI]:
        nHK = len(self.rHK)
        left_attach_hinge = len(self.hinge_l_grp.positions) // 2
        right_attach_hinge = len(self.hinge_r_grp.positions) // 2
        return [
            # Fix orientation of ATP/kleisin bridge
            # WARNING: siteM is split into groups, be careful with index
            BAI(imp_t1, (self.arm_dl_grp, -1), (self.arm_dl_grp, 0), (self.atp_grp, -1), (self.middle_site_grp, 1)),
            BAI(imp_t1, (self.arm_dr_grp, 0), (self.arm_dr_grp, -1), (self.atp_grp, 0), (self.middle_site_grp, 1)),

            BAI(imp_t2, (self.arm_dl_grp, -1), (self.arm_dl_grp, 0), (self.atp_grp, -1), (self.hk_grp, nHK//2)),
            BAI(imp_t2, (self.arm_dr_grp, 0), (self.arm_dr_grp, -1), (self.atp_grp, 0), (self.hk_grp, nHK//2)),

            # prevent kleisin ring from swaying too far relative to the bridge
            BAI(imp_t3, (self.middle_site_ref_grp, 0), (self.arm_dl_grp, 0), (self.arm_dr_grp, -1), (self.hk_grp, nHK//2)),

            # fix hinge to a plane
            BAI(imp_t1, (self.hinge_l_grp, 0), (self.hinge_l_grp, -1), (self.hinge_r_grp, 0), (self.hinge_r_grp, -1)),
            BAI(imp_t1, (self.hinge_l_grp, 0), (self.hinge_l_grp, left_attach_hinge), (self.hinge_r_grp, 0), (self.hinge_r_grp, right_attach_hinge)),

            # fix hinge perpendicular to arms plane
            BAI(imp_t4, (self.arm_ul_grp, -2), (self.arm_ul_grp, -1), (self.hinge_l_grp, left_attach_hinge - 1), (self.hinge_l_grp, left_attach_hinge + 1)),
            BAI(imp_t4, (self.arm_ur_grp, 1), (self.arm_ur_grp, 0), (self.hinge_r_grp, right_attach_hinge + 1), (self.hinge_r_grp, right_attach_hinge - 1)),

            # # keep hinge aligned with bridge axis
            # BAI(imp_t1, (self.hingeL_group, left_attach_hinge), (self.hingeR_group, right_attach_hinge), (self.atp_group, 0), (self.atp_group, -1)),
            BAI(imp_t4, (self.upper_site_grp, 0), (self.upper_site_grp, len(self.upper_site_grp.positions) // 2), (self.atp_grp, len(self.atp_grp.positions) // 2), (self.atp_grp, -1)),
        ]
