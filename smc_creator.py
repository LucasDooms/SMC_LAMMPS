from structure_creator import get_circle_segment_unit_radius, get_straight_segment, attach_chain
from dataclasses import dataclass
import math
import numpy as np
from numpy.random import default_rng
from scipy.spatial.transform import Rotation


@dataclass
class SMC_Creator:

    SMCspacing: float

    siteUhDist: float
    siteUvDist: float
    siteMhDist: float
    siteMvDist: float
    siteDhDist: float
    siteDvDist: float

    armLength: float
    bridgeWidth: float

    HKradius: float

    foldingAngleAPO: float

    SMALL: float = 1e-9


    def get_arms(self, seed: int = 8671288977726523465):
        # Number of beads forming each arm segment (err on the high side)
        nArmSegm = math.ceil(self.armLength / (2 * self.SMCspacing))

        # z and y lengths of each arm (2 aligned segments), for the initial triangular geometry
        zArm = self.bridgeWidth / 2.0
        yArm = math.sqrt(self.armLength**2 - zArm**2)

        direction_DL = [0, yArm, zArm]
        direction_UL = [0, yArm, zArm]
        direction_UR = [0, -yArm, zArm]
        direction_DR = [0, -yArm, zArm]

        rArmDL = get_straight_segment(nArmSegm, direction_DL) / 2.0 * self.armLength / nArmSegm
        rArmUL = get_straight_segment(nArmSegm, direction_UL) / 2.0 * self.armLength / nArmSegm
        rArmUR = get_straight_segment(nArmSegm, direction_UR) / 2.0 * self.armLength / nArmSegm
        rArmDR = get_straight_segment(nArmSegm, direction_DR) / 2.0 * self.armLength / nArmSegm

        rArmDL, rArmUL, rArmUR, rArmDR = attach_chain(rArmDL, [[rArmUL, False], [rArmUR, False], [rArmDR, False]])

        # move lower center to origin (where bridge will be placed)
        center = (rArmDL[0] + rArmDR[-1]) / 2.0
        pieces = [rArmDL, rArmUL, rArmUR, rArmDR]
        for i in range(len(pieces)):
            pieces[i] -= center


        # A bit of randomness, to avoid exact overlap (pressure is messed up in LAMMPS)
        rng_arms = default_rng(seed=seed)

        rArmDL += rng_arms.standard_normal(size=rArmDL.shape) * self.SMALL
        rArmUL += rng_arms.standard_normal(size=rArmUL.shape) * self.SMALL
        rArmUR += rng_arms.standard_normal(size=rArmUR.shape) * self.SMALL
        rArmDR += rng_arms.standard_normal(size=rArmDR.shape) * self.SMALL
        
        return rArmDL, rArmUL, rArmUR, rArmDR

    def get_bridge(self, seed: int = 4685150768879447999):
        # Number of beads forming the ATP ring (err on the high side)
        nATP = math.ceil(self.bridgeWidth / self.SMCspacing)

        # We want an odd number (necessary for angle/dihedral interactions)
        if nATP % 2 == 0:
            nATP += 1

        # Positions
        rATP = get_straight_segment(nATP, [0, 0, 1])

        # use the bridgeWidth as the absolute truth (SMC spacing may be slighlty off)
        rATP *= self.bridgeWidth / nATP
        
        # move center to origin
        rATP -= (rATP[0] + rATP[-1]) / 2.0

        # A bit of randomness
        rng_atp = default_rng(seed=seed)
        rATP += rng_atp.standard_normal(rATP.shape) * self.SMALL
        
        return rATP

    def get_heads_kleisin(self, seed: int = 8305832029550348799):
        # Circle-arc radius
        # radius = (self.HKradius**2 + (self.bridgeWidth / 2.0)**2) / (2.0 * self.HKradius)
        bridgeRadius = self.bridgeWidth / 2.0
        radius = self.HKradius
        if radius < bridgeRadius:
            raise ValueError(f"The kleisin radius ({radius}) is too small (<{bridgeRadius}) based on the bridgeWidth {self.bridgeWidth}")

        # from the y-axis
        starting_angle = math.asin(bridgeRadius / radius)
        # Opening angle of circular arc (away from the bridge = 2*pi - angle towards the bridge)
        phi0 = 2.0 * math.pi - 2.0 * starting_angle

        # Number of beads forming the heads/kleisin complex (err on the high side)
        nHK = math.ceil(phi0 * radius / self.SMCspacing)

        # We want an odd number (necessary for angle/dihedral interactions)
        if nHK % 2 == 0:
            nHK += 1
        
        ending_angle = 2.0 * math.pi - starting_angle
        # add pi/2, since circle_segment calculation starts from x-axis
        starting_angle += math.pi / 2.0
        ending_angle += math.pi / 2.0

        rHK = get_circle_segment_unit_radius(nHK, end_inclusive=True, theta_start=starting_angle,
                                             theta_end=ending_angle, normal_direction=[1, 0, 0])

        rHK *= radius
        # move the bridge-gap to the origin
        rHK[:,1] -= rHK[0][1]


        # A bit of randomness
        rng_rhk = default_rng(seed=seed)
        rHK += rng_rhk.standard_normal(size=rHK.shape) * self.SMALL

        return rHK

    @staticmethod
    def shielded_site_template(nInnerBeads: int, nOuterBeadsPerInnerBead: int, innerSpacing: float, outerSpacing: float):
        """create a line of beads surrounded by a protective shell/shield"""
        axis = np.array([1, 0, 0])
        # Inner/Attractive beads
        innerBeads = get_straight_segment(nInnerBeads, direction=axis) * innerSpacing
        # put center at the origin
        innerBeads -= (innerBeads[0] + innerBeads[-1]) / 2.0

        # Repulsive/Outer beads, forming a surrounding shell
        shells = []
        for innerBead in innerBeads:
            shells.append(
                get_circle_segment_unit_radius(nOuterBeadsPerInnerBead, end_inclusive=True,
                                               theta_start=0, theta_end=np.pi,
                                               normal_direction=axis) * outerSpacing
            )
            # place center of shell at inner bead
            shells[-1] += innerBead

        # Horizontal shield at two ends
        end_first = (innerBeads[0] - innerSpacing * axis).reshape(1, 3)
        end_last = (innerBeads[-1] + innerSpacing * axis).reshape(1, 3)

        return np.concatenate([innerBeads, *shells, end_first, end_last])

    @staticmethod
    def transpose_rotate_transpose(rotation, array):
        return rotation.dot(array.transpose()).transpose()

    def get_interaction_sites(self, seed: int = 8343859591397577529):
        # U = upper  interaction site
        # M = middle interaction site
        # D = lower  interaction site

        rotate_around_x_axis = Rotation.from_rotvec(math.pi * np.array([1.0, 0.0, 0.0])).as_matrix()

        # UPPER SITE
        rSiteU = self.shielded_site_template(3, 4, self.siteUhDist, 1)
        # Inert bead connecting site to arms at top
        rSiteU = np.concatenate([rSiteU, np.array([0.0, self.siteUvDist, 0.0]).reshape(1, 3)])

        # MIDDLE SITE
        rSiteM = self.shielded_site_template(1, 4, self.siteMhDist, 1)
        rSiteM = self.transpose_rotate_transpose(rotate_around_x_axis, rSiteM)

        # take last bead and use it as an extra inner bead
        rSiteM = np.concatenate([rSiteM[:1], rSiteM[-1:], rSiteM[1:-1]])
        # move, so that this bead is at the origin
        rSiteM -= rSiteM[1]
      
        # LOWER SITE
        rSiteD = self.shielded_site_template(3, 4, self.siteDhDist, 1)
        rSiteD = self.transpose_rotate_transpose(rotate_around_x_axis, rSiteD)


        # Add randomness
        rng_sites = default_rng(seed=seed)
        rSiteU += rng_sites.standard_normal(size=rSiteU.shape) * self.SMALL
        rSiteM += rng_sites.standard_normal(size=rSiteM.shape) * self.SMALL
        rSiteD += rng_sites.standard_normal(size=rSiteD.shape) * self.SMALL

        return rSiteU, rSiteM, rSiteD

    def get_smc(self):
        rArmDL, rArmUL, rArmUR, rArmDR = self.get_arms()
        rATP = self.get_bridge()
        rHK = self.get_heads_kleisin()
        rSiteU, rSiteM, rSiteD = self.get_interaction_sites()
        
        rSiteU[:,1] -= self.siteUvDist
        # Inert bead, used for breaking folding symmetry
        rSiteM = np.concatenate([rSiteM, np.array([1.0, -1.0, 0.0]).reshape(1, 3)])
        rSiteM[:,1] += self.siteMvDist
        rSiteD[:,1] += self.siteDvDist
        
        # scale properly
        rSiteU *= self.SMCspacing
        rSiteM *= self.SMCspacing
        rSiteD *= self.SMCspacing
    
        # move into the correct location
        rSiteU += rArmUL[-1]
        rSiteM += rATP[len(rATP)//2]
        rSiteD += rHK[len(rHK)//2]
        
        ############################# Fold upper compartment ############################

        # Rotation matrix (clockwise about z axis)
        rotMat = Rotation.from_rotvec(-math.radians(self.foldingAngleAPO) * np.array([0.0, 0.0, 1.0])).as_matrix()

        # Rotations
        rArmDL = self.transpose_rotate_transpose(rotMat, rArmDL)
        rArmUL = self.transpose_rotate_transpose(rotMat, rArmUL)
        rArmUR = self.transpose_rotate_transpose(rotMat, rArmUR)
        rArmDR = self.transpose_rotate_transpose(rotMat, rArmDR)
        rSiteU = self.transpose_rotate_transpose(rotMat, rSiteU)
        rSiteM = self.transpose_rotate_transpose(rotMat, rSiteM)

        return rArmDL, rArmUL, rArmUR, rArmDR, rATP, rHK, rSiteU, rSiteM, rSiteD
