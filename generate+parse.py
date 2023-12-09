import math
import numpy as np
from numpy.random import default_rng
from scipy.spatial.transform import Rotation
from generator import Generator, BAI, BAI_Type, BAI_Kind, AtomGroup, AtomType, PairWise
from sys import argv
from pathlib import Path
from dna_creator import get_dna_coordinates
from importlib import import_module
import default_parameters
#import matplotlib.pyplot as plt


if len(argv) != 2:
    raise Exception("Provide a folder path")
path = Path(argv[1])

parameters = import_module((path / "parameters").as_posix().replace('/', '.'))

class Parameters:

    def __getattr__(self, var_name):
        return getattr(parameters, var_name, getattr(default_parameters, var_name))

    def __dir__(self):
        return list(set(dir(parameters)) | set(dir(default_parameters)))

par = Parameters()

# Name of generated data file
filename_data = 'datafile'
filepath_data = path / filename_data

# Name of generated parameter file
filename_param = 'parameterfile'
filepath_param = path / filename_param

nDNA = par.N
DNAdiscr = par.n


#################################################################################
#                               Other parameters                                #
#################################################################################


# Simulation temperature (K)
T = par.T

# Boltzmann's constant (pN nm / K)
kB = par.kB


#################################### Masses #####################################


#######
# DNA #
#######

# Mass per base pair (ag)
bpMass = 2 * 3.1575 * 5.24e-4

# Effective bead mass (ag)
mDNA = DNAdiscr * bpMass


#######
# SMC #
#######

# Total mass of SMC protein (ag)
mSMCtotal = 0.25


#################################### Lengths ####################################


################
# Interactions #
################

# DNA-DNA repulsion radius (nm)
intRadDNAvsDNA = 3.5


#######
# DNA #
#######

# Bending stiffness (nm)
DNAstiff = 50.

# Base pair step (nm)
bpStep = 0.34

# Effective bond length = DNA bead size (nm)
DNAbondLength = DNAdiscr * bpStep

# Total length of DNA (nm)
DNAcontourLength = DNAbondLength * nDNA


#######
# SMC #
#######

# Desirable SMC spacing (radius of 1 SMC bead is R = intRadSMCvsDNA)
# Equal to R:   Minimum diameter = sqrt(3)    = 1.73 R
# Equal to R/2: Minimum diameter = sqrt(15)/2 = 1.94 R
SMCspacing = par.intRadSMCvsDNA/2


#################
# Binding sites #
#################

# Vertical distance of top binding sites from hinge (units of bead spacing)
siteUvDist = 4

# Horizontal distance between top binding sites (units bead spacing)
siteUhDist = 2


# Vertical distance of middle binding sites from bridge (units of bead spacing)
siteMvDist = 1

# Horizontal distance between middle binding sites (units bead spacing)
siteMhDist = 2


# Distance of bottom binding sites from kleisin (units of bead spacing)
siteDvDist = 0.5

# Horizontal distance between bottom binding sites (units bead spacing)
siteDhDist = 2


##################
# Simulation box #
##################


# Width of cubic simulation box (nm)
boxWidth = 2*DNAcontourLength


################################## Interactions #################################


###########
# DNA-DNA #
###########

sigmaDNAvsDNA   = intRadDNAvsDNA
epsilonDNAvsDNA = par.epsilon3
rcutDNAvsDNA    = sigmaDNAvsDNA * 2**(1/6)


###########
# SMC-DNA #
###########

sigmaSMCvsDNA   = par.intRadSMCvsDNA
epsilonSMCvsDNA = par.epsilon3
rcutSMCvsDNA    = sigmaSMCvsDNA * 2**(1/6)


#############
# Sites-DNA #
#############

# Sigma of LJ attraction (same as those of the repulsive SMC sites)
sigmaSiteDvsDNA = sigmaSMCvsDNA

# Cutoff distance of LJ attraction
rcutSiteDvsDNA = par.cutoff6

# Epsilon parameter of LJ attraction
epsilonSiteDvsDNA = par.epsilon6 * kB*T


#################################################################################
#                                 SMC complex                                   #
#################################################################################


#################################### Arms #######################################


# Number of beads forming each arm segment (err on the high side)
nArmSegm = math.ceil(par.armLength / 2 / SMCspacing)

# Create list of 3 zeros for later coordinates of each arm bead for each arm segment (U=up, D=down, L=left, R=right)
rArmDL = np.zeros((nArmSegm,3))
rArmUL = np.zeros((nArmSegm,3))
rArmUR = np.zeros((nArmSegm,3))
rArmDR = np.zeros((nArmSegm,3))

# z and y lengths of each arm (2 aligned segments), for the initial triangular geometry
zArm = par.bridgeWidth / 2
yArm = ( par.armLength**2 - zArm**2 )**0.5 

# y positions
rArmDL[:,1] = np.linspace(      0,  yArm/2, nArmSegm)
rArmUL[:,1] = np.linspace( yArm/2,    yArm, nArmSegm)
rArmUR[:,1] = np.linspace(   yArm,  yArm/2, nArmSegm)
rArmDR[:,1] = np.linspace( yArm/2,       0, nArmSegm)

# z positions
rArmDL[:,2] = np.linspace(  -zArm, -zArm/2, nArmSegm)
rArmUL[:,2] = np.linspace(-zArm/2,       0, nArmSegm)
rArmUR[:,2] = np.linspace(      0,  zArm/2, nArmSegm)
rArmDR[:,2] = np.linspace( zArm/2,    zArm, nArmSegm)


# A bit of randomness, to avoid exact overlap (pressure is messed up in LAMMPS)
SMALL = 1e-9
rng_arms = default_rng(seed=8671288977726523465)

rArmDL += rng_arms.standard_normal(size=rArmDL.shape) * SMALL
rArmUL += rng_arms.standard_normal(size=rArmUL.shape) * SMALL
rArmUR += rng_arms.standard_normal(size=rArmUR.shape) * SMALL
rArmDR += rng_arms.standard_normal(size=rArmDR.shape) * SMALL


################################# ATP bridge ####################################


# Number of beads forming the ATP ring (err on the high side)
nATP = math.ceil( par.bridgeWidth / SMCspacing )

# We want an odd number (necessary for angle/dihedral interactions)
if nATP%2 == 0:
    nATP += 1


# Positions
rATP      = np.zeros((nATP,3))
rATP[:,2] = np.linspace(-par.bridgeWidth/2, par.bridgeWidth/2, nATP)


# A bit of randomness
rng_atp = default_rng(seed=4685150768879447999)
rATP += rng_atp.standard_normal(rATP.shape) * SMALL


################################ Heads/Kleisin ##################################


# Circle-arc radius
radius = ( par.HKradius**2 + (par.bridgeWidth/2)**2 ) / ( 2 * par.HKradius )

# Opening angle of circular arc
phi0 = 2 * math.asin( par.bridgeWidth / 2 / radius )
if par.HKradius > par.bridgeWidth/2:
    phi0 = 2*math.pi - phi0


# Number of beads forming the heads/kleisin complex (err on the high side)
nHK = math.ceil( phi0 * radius / SMCspacing )

# We want an odd number (necessary for angle/dihedral interactions)
if nHK%2 == 0:
    nHK += 1


# Polar angle
phi = np.linspace(-(math.pi-phi0)/2, -(math.pi+phi0)/2, nHK) 
if par.HKradius > par.bridgeWidth/2:
    phi = np.linspace((phi0-math.pi)/2, -math.pi-(phi0-math.pi)/2, nHK)


# Positions
rHK      = np.zeros((nHK,3))
rHK[:,1] = radius * np.sin(phi) - par.HKradius + radius
rHK[:,2] = radius * np.cos(phi)

# A bit of randomness
rng_rhk = default_rng(seed=8305832029550348799)
rHK += rng_rhk.standard_normal(size=rHK.shape) * SMALL


############################### Interaction sites ###############################


# U = upper  interaction site
# M = middle interaction site
# D = lower  interaction site

# Number of beads per site
nSiteU = 18
nSiteM =  8
nSiteD = 17

rSiteU = np.zeros((nSiteU,3))
rSiteM = np.zeros((nSiteM,3))
rSiteD = np.zeros((nSiteD,3))


# Polar angles of a 4-bead semicircle
phi = np.arange(4) * np.pi/3


# UPPER SITE


# Attractive beads
rSiteU[0] = rArmUL[-1] + SMCspacing * np.array([-siteUhDist, -siteUvDist, 0])
rSiteU[1] = rArmUL[-1] + SMCspacing * np.array([          0, -siteUvDist, 0])
rSiteU[2] = rArmUL[-1] + SMCspacing * np.array([+siteUhDist, -siteUvDist, 0])

# Repulsive beads, forming a surrounding shell
for index in range(len(phi)):
    rSiteU[3 +index] = rSiteU[0] + SMCspacing * np.array([ 0, np.sin(phi[index]), np.cos(phi[index]) ])
    rSiteU[7 +index] = rSiteU[1] + SMCspacing * np.array([ 0, np.sin(phi[index]), np.cos(phi[index]) ])
    rSiteU[11+index] = rSiteU[2] + SMCspacing * np.array([ 0, np.sin(phi[index]), np.cos(phi[index]) ])

# Horizontal shield at two ends
rSiteU[15] = rSiteU[0] + SMCspacing * np.array([-siteUhDist,0,0])
rSiteU[16] = rSiteU[2] + SMCspacing * np.array([ siteUhDist,0,0])

# Inert bead connecting site to arms at top
rSiteU[17] = rArmUL[-1]


# MIDDLE SITE


# Attractive beads
rSiteM[0] = rATP[nATP//2] + SMCspacing * np.array([-siteMhDist, siteMvDist, 0])
rSiteM[1] = rATP[nATP//2] + SMCspacing * np.array([          0, siteMvDist, 0])

# Inert bead, used for breaking folding symmetry
rSiteM[2] = rATP[nATP//2] + SMCspacing * np.array([ 1, 0, 0])

# Repulsive beads, forming a surrounding shell
for index in range(len(phi)):
    rSiteM[3+index] = rSiteM[0] - SMCspacing * np.array([ 0, np.sin(phi[index]), np.cos(phi[index]) ])

# Horizontal shield at one end
rSiteM[7] = rSiteM[0] + SMCspacing * np.array([-siteMhDist, 0, 0])


# LOWER SITE


# Attractive beads
rSiteD[0] = rHK[nHK//2] + SMCspacing * np.array([-siteDhDist,  siteDvDist, 0])
rSiteD[1] = rHK[nHK//2] + SMCspacing * np.array([          0,  siteDvDist, 0])
rSiteD[2] = rHK[nHK//2] + SMCspacing * np.array([+siteDhDist,  siteDvDist, 0])

# Repulsive beads, forming a surrounding shell
for index in range(len(phi)):
    rSiteD[3 +index] = rSiteD[0] - SMCspacing * np.array([ 0, np.sin(phi[index]), np.cos(phi[index]) ])
    rSiteD[7 +index] = rSiteD[1] - SMCspacing * np.array([ 0, np.sin(phi[index]), np.cos(phi[index]) ])
    rSiteD[11+index] = rSiteD[2] - SMCspacing * np.array([ 0, np.sin(phi[index]), np.cos(phi[index]) ])

# Horizontal shield at two ends
rSiteD[15] = rSiteD[0] + SMCspacing * np.array([-siteDhDist,0,0])
rSiteD[16] = rSiteD[2] + SMCspacing * np.array([ siteDhDist,0,0])


# Add randomness
rng_sites = default_rng(seed=8343859591397577529)
rSiteU += rng_sites.standard_normal(size=rSiteU.shape) * SMALL
rSiteM += rng_sites.standard_normal(size=rSiteM.shape) * SMALL
rSiteD += rng_sites.standard_normal(size=rSiteD.shape) * SMALL


############################# Fold upper compartment ############################

# Rotation matrix (clockwise about z axis)
rotMat = Rotation.from_rotvec(-math.radians(par.foldingAngleAPO) * np.array([0.0, 0.0, 1.0])).as_matrix()

# Rotations
def transpose_rotate_transpose(rotation, array):
    return rotation.dot(array.transpose()).transpose()

rArmDL = transpose_rotate_transpose(rotMat, rArmDL)
rArmUL = transpose_rotate_transpose(rotMat, rArmUL)
rArmUR = transpose_rotate_transpose(rotMat, rArmUR)
rArmDR = transpose_rotate_transpose(rotMat, rArmDR)
rSiteU = transpose_rotate_transpose(rotMat, rSiteU)
rSiteM = transpose_rotate_transpose(rotMat, rSiteM)


#################################################################################
#                                     DNA                                       #
#################################################################################

rDNA, nLowerDNA = get_dna_coordinates(nDNA, DNAbondLength, 14, 10)

#################################################################################
#                               Shift DNA to SMC                                #
#################################################################################


# make sure SMC touches the DNA at the lower site (siteD)
desired_y_pos = rSiteD[1][1] - 2.0 * par.cutoff6
shift_y = desired_y_pos - rDNA[-1][1]
desired_x_pos = rSiteD[1][0]
shift_x = desired_x_pos - rDNA[-(nLowerDNA - 3)][0]
shift = np.array([shift_x, shift_y, 0]).reshape(1, 3)
rDNA += shift

# find closest DNA bead to siteD
distances = np.linalg.norm(rDNA - rSiteD[1], axis=1)
closest_DNA_index = int(np.argmin(distances))


#################################################################################
#                                Print to file                                  #
#################################################################################

# Divide total mass evenly among the segments
mSMC = mSMCtotal / ( 4*nArmSegm + nHK + nATP + nSiteU + nSiteM + nSiteD )

# Relative bond fluctuations
bondFlDNA = 1e-2
bondFlSMC = 1e-2

# Maximum relative bond extension (units of rest length)
bondMax = 1.

# Spring constant obeying equilibrium relative bond fluctuations
kBondDNA    = 3*kB*T/(DNAbondLength*bondFlDNA)**2
kBondSMC    = 3*kB*T/(   SMCspacing*bondFlSMC)**2
kBondAlign1 =  10*kB*T / SMCspacing**2
kBondAlign2 = 200*kB*T / SMCspacing**2


indL = np.argmin(np.linalg.norm(rSiteU[-2]-rArmUL, axis=1))
indL = int(indL) # result should be an int if array is one dimensional
indR = np.argmin(np.linalg.norm(rSiteU[-2]-rArmUR, axis=1))
indR = int(indR)
bondMin1 = np.linalg.norm(rSiteU[-2]-rArmUL[indL])
bondMin2 = np.linalg.norm(rSiteU[-2]-rArmUL[-1])

# Maximum bond length
maxLengthDNA = DNAbondLength*bondMax
maxLengthSMC =    SMCspacing*bondMax

# DNA bending rigidity
kDNA = DNAstiff * kB*T / DNAbondLength

# Angular trap constants
# kElbows: Bending of elbows (kinkable arms, hence soft)
# kArms:   Arms opening angle wrt ATP bridge (should be stiff)
kElbows = par.elbowsStiffness*kB*T
kArms   =   par.armsStiffness*kB*T

# Fixes site orientation (prevents free rotation, should be stiff)
kAlignSite = par.siteStiffness*kB*T

# Folding stiffness of lower compartment (should be stiff)
kFolding = par.foldingStiffness*kB*T

# Makes folding asymmetric (should be stiff)
kAsymmetry = par.asymmetryStiffness*kB*T

# SET UP DATAFILE GENERATOR
gen = Generator()
gen.set_system_size(boxWidth)

# Molecule for each rigid body
molDNA   = 1
molArmDL = 2
molArmUL = 3
molArmUR = 4
molArmDR = 5
molHK    = 6
molATP   = 7
molSiteU = 8
molSiteM = molATP
molSiteD = molHK

dna_bond = BAI_Type(BAI_Kind.BOND, "fene/expand %s %s %s %s %s\n" %(kBondDNA, maxLengthDNA, 0, 0, DNAbondLength))
dna_type = AtomType(mDNA)
dna_group = AtomGroup(
    positions=rDNA,
    atom_type=dna_type,
    molecule_index=molDNA,
    polymer_bond_type=dna_bond
)

armHK_type = AtomType(mSMC)
atp_type = AtomType(mSMC)
siteU_type = AtomType(mSMC)
siteM_type = AtomType(mSMC)
siteD_type = AtomType(mSMC)
refSite_type = AtomType(mSMC)

armDL_group = AtomGroup(rArmDL, armHK_type, molArmDL)
armUL_group = AtomGroup(rArmUL, armHK_type, molArmUL)
armUR_group = AtomGroup(rArmUR, armHK_type, molArmUR)
armDR_group = AtomGroup(rArmDR, armHK_type, molArmDR)
hk_group = AtomGroup(rHK, armHK_type, molHK)

atp_group = AtomGroup(rATP, atp_type, molATP)

# split U in two parts

cut = 3
siteU_group = AtomGroup(rSiteU[:cut], siteU_type, molSiteU)
siteU_arm_group = AtomGroup(rSiteU[cut:], armHK_type, molSiteU)

# split M in three parts

cut = 2
siteM_group = AtomGroup(rSiteM[:cut], siteM_type, molSiteM)
# ref site
siteM_ref_group = AtomGroup(rSiteM[cut:cut+1], refSite_type, molSiteM)
siteM_atp_group = AtomGroup(rSiteM[cut+1:], atp_type, molSiteM)

# split B in two parts

cut = 3
siteD_group = AtomGroup(rSiteD[:cut], siteD_type, molSiteD)
siteD_arm_group = AtomGroup(rSiteD[cut:], armHK_type, molSiteD)

gen.atom_groups += [
    dna_group,
    armDL_group,
    armUL_group,
    armUR_group,
    armDR_group,
    hk_group,
    atp_group,
    siteU_group,
    siteU_arm_group,
    siteM_group,
    siteM_ref_group,
    siteM_atp_group,
    siteD_group,
    siteD_arm_group
]


# Pair coefficients
pair_inter = PairWise("PairIJ Coeffs # hybrid\n\n", "lj/cut {} {} {}\n", [0.0, 0.0, 0.0])

pair_inter.add_interaction(dna_type, dna_type, epsilonDNAvsDNA, sigmaDNAvsDNA, rcutDNAvsDNA)
pair_inter.add_interaction(dna_type, armHK_type, epsilonSMCvsDNA, sigmaSMCvsDNA, rcutSMCvsDNA)
pair_inter.add_interaction(dna_type, siteD_type, epsilonSiteDvsDNA, sigmaSiteDvsDNA, rcutSiteDvsDNA)

# soft interactions
pair_soft_inter = PairWise("PairIJ Coeffs # hybrid\n\n", "soft {} {}\n", [0.0, 0.0])

gen.pair_interactions.append(pair_inter)
gen.pair_interactions.append(pair_soft_inter)


# Every joint is kept in place through bonds
bond_t2 = BAI_Type(BAI_Kind.BOND, "fene/expand %s %s %s %s %s\n" %(kBondSMC, maxLengthSMC, 0, 0, 0))
bond_t3 = BAI_Type(BAI_Kind.BOND, "harmonic %s %s\n"             %(kBondAlign1, bondMin1))
bond_t4 = BAI_Type(BAI_Kind.BOND, "harmonic %s %s\n"           %(kBondAlign2, bondMin2))

bonds = [
    BAI(bond_t2, (armDL_group, -1), (armUL_group, 0)),
    BAI(bond_t2, (armUL_group, -1), (armUR_group, 0)),
    BAI(bond_t2, (armUR_group, -1), (armDR_group, 0)),
    BAI(bond_t2, (armUL_group, -1), (siteU_arm_group, -1)),
    BAI(bond_t2, (armDR_group, -1), (atp_group, -1)),
    BAI(bond_t2, (atp_group,  0), (armDL_group, 0)),
    BAI(bond_t2, (atp_group, -1), (hk_group, 0)),
    BAI(bond_t2, (hk_group, -1), (atp_group, 0)),
    BAI(bond_t3, (armUL_group, indL), (siteU_arm_group, -2)),
    BAI(bond_t3, (armUL_group, indL), (siteU_arm_group, -3)),
    BAI(bond_t3, (armUR_group, indR), (siteU_arm_group, -2)),
    BAI(bond_t3, (armUR_group, indR), (siteU_arm_group, -3)),
    BAI(bond_t4, (armUL_group, -1), (siteU_arm_group, -2)),
    BAI(bond_t4, (armUL_group, -1), (siteU_arm_group, -3)),
] 

gen.bais += bonds

angle_t1 = BAI_Type(BAI_Kind.ANGLE, "cosine %s\n"        %  kDNA )
angle_t2 = BAI_Type(BAI_Kind.ANGLE, "harmonic %s %s\n"   % ( kElbows, 180 ) )
angle_t3 = BAI_Type(BAI_Kind.ANGLE, "harmonic %s %s\n" % ( kArms,  np.rad2deg( math.acos( par.bridgeWidth / par.armLength ) ) ) )

# DNA stiffness
dna_angle_list = []
for index in range(nDNA-2):
    dna_angle_list.append(BAI(
        angle_t1,
        (dna_group, index),
        (dna_group, index + 1),
        (dna_group, index + 2)
    ))

arm_arm_angle1 = BAI(angle_t2, (armDL_group, 0), (armUL_group, 0), (armUL_group, -1))
arm_arm_angle2 = BAI(angle_t2, (armUR_group, 0), (armUR_group, -1), (armDR_group, -1))

arm_atp_angle1 = BAI(angle_t3, (armDL_group, -1), (armDL_group, 0), (atp_group, -1))
arm_atp_angle2 = BAI(angle_t3, (armDR_group, 0), (armDR_group, -1), (atp_group, 0))

gen.bais += dna_angle_list + [arm_arm_angle1, arm_arm_angle2, arm_atp_angle1, arm_atp_angle2]

# We impose zero improper angle
imp_t1 = BAI_Type(BAI_Kind.IMPROPER, "%s %s\n"   %( kAlignSite, 0 ) )
imp_t2 = BAI_Type(BAI_Kind.IMPROPER, "%s %s\n"   %( kFolding,   180 - par.foldingAngleAPO ) )
imp_t3 = BAI_Type(BAI_Kind.IMPROPER, "%s %s\n" %( kAsymmetry,  math.fabs(90 - par.foldingAngleAPO) ) )

# Fix orientation of ATP/kleisin bridge
# WARNING: siteM is split into groups, be careful with index
atp_HK_improper1 = BAI(imp_t1, (armDL_group, -1), (armDL_group, 0), (atp_group, -1), (siteM_group, 1))
atp_HK_improper2 = BAI(imp_t1, (armDR_group, 0), (armDR_group, -1), (atp_group, 0), (siteM_group, 1))

folding_angle_improper1 = BAI(imp_t2, (armDL_group, -1), (armDL_group, 0), (atp_group, -1), (hk_group, nHK//2))
folding_angle_improper2 = BAI(imp_t2, (armDR_group, 0), (armDR_group, -1), (atp_group, 0), (hk_group, nHK//2))

# WARNING: indices M changed
folding_asymmetry_improper = BAI(imp_t3, (siteM_ref_group, 0), (armDL_group, 0), (armDR_group, -1), (hk_group, nHK//2))
# datafile.write("9 3 %s %s %s %s\n\n" %(IDsiteM[2], IDarmDL[ 0], IDarmDR[-1], IDhK[nHK//2]))

gen.bais += [atp_HK_improper1, atp_HK_improper2, folding_angle_improper1, folding_angle_improper2, folding_asymmetry_improper]


with open(filepath_data, 'w') as datafile:
    gen.write(datafile)


# TODO
angle3kappa = par.armsStiffness * kB * T
angle3angleATP = par.armsAngleATP

improper2kappa = par.foldingStiffness * kB * T
improper2angleAPO = 180 - par.foldingAngleAPO

improper3kappa = par.asymmetryStiffness * kB * T
improper3angleAPO = abs(90 - par.foldingAngleAPO)

angle3angleAPO1 = 180 / math.pi * np.arccos(par.bridgeWidth / par.armLength)
angle3angleAPO1 = 180 / math.pi * np.arccos(2 * par.bridgeWidth / par.armLength)

improper2angleATP = 180 - par.foldingAngleATP
improper3angleATP = abs(90 - par.foldingAngleATP)

bridge_off = [None, "{} {} lj/cut 0 0 0\n", dna_type, atp_type]
bridge_on = [None, "{} {} " + f"lj/cut {par.epsilon3 * kB * T} {par.sigma} {par.sigma * 2**(1/6)}\n", dna_type, atp_type]

bridge_soft_off = [None, "{} {} soft 0 0\n", dna_type, atp_type]
bridge_soft_on = [None, "{} {} soft " + f"{par.epsilon3 * kB * T} {par.sigma * 2**(1/6)}\n", dna_type, atp_type]

top_site_off = [None, "{} {} lj/cut 0 0 0\n", dna_type, siteU_type]
top_site_on = [None, "{} {} " + f"lj/cut {par.epsilon4 * kB * T} {par.sigma} {par.cutoff4}\n", dna_type, siteU_type]

middle_site_off = [None, "{} {} lj/cut 0 0 0\n", dna_type, siteM_type]
middle_site_on = [None, "{} {} " + f"lj/cut {par.epsilon5 * kB * T} {par.sigma} {par.cutoff5}\n", dna_type, siteM_type]

middle_site_soft_off = [None, "{} {} soft 0 0\n", dna_type, siteM_type]
middle_site_soft_on = [None, "{} {} soft " + f"{par.epsilon5 * kB * T} {par.sigma * 2**(1/6)}\n", dna_type, siteM_type]

lower_site_off = [None, "{} {} lj/cut 0 0 0\n", dna_type, siteD_type]
lower_site_on = [None, "{} {} " + f"lj/cut {par.epsilon6 * kB * T} {par.sigma} {par.cutoff6}\n", dna_type, siteD_type]

arms_close = [BAI_Kind.ANGLE, "{} harmonic " + f"{angle3kappa} {angle3angleAPO1}\n", angle_t3]
arms_open = [BAI_Kind.ANGLE, "{} harmonic " + f"{angle3kappa} {angle3angleATP}\n", angle_t3]

lower_compartment_folds1 = [BAI_Kind.IMPROPER, "{} "+ f"{improper2kappa} {improper2angleATP}\n", imp_t2]
lower_compartment_unfolds1 = [BAI_Kind.IMPROPER, "{} "+ f"{improper2kappa} {improper2angleAPO}\n", imp_t2]

lower_compartment_folds2 = [BAI_Kind.IMPROPER, "{} " + f"{improper3kappa} {improper3angleATP}\n", imp_t3]
lower_compartment_unfolds2 = [BAI_Kind.IMPROPER, "{} " + f"{improper3kappa} {improper3angleAPO}\n", imp_t3]

# make sure the directory exists
states_path = path / "states"
states_path.mkdir(exist_ok=True)

def apply(function, file, list_of_args):
    for args in list_of_args:
        function(file, *args)

with open(states_path / "adp_bound", 'w') as adp_bound_file:
    options = [
       bridge_off,
       top_site_on,
       middle_site_off,
       lower_site_off,
       arms_open,
       lower_compartment_unfolds1,
       lower_compartment_unfolds2
    ]
    apply(gen.write_script_bai_coeffs, adp_bound_file, options)

with open(states_path / "apo", 'w') as apo_file:
    options = [
        bridge_off,
        top_site_off,
        middle_site_off,
        lower_site_on,
        arms_close,
        lower_compartment_unfolds1,
        lower_compartment_unfolds2
    ]
    apply(gen.write_script_bai_coeffs, apo_file, options)
    
    # gen.write_script_bai_coeffs(adp_bound_file, BAI_Kind.ANGLE, "{} harmonic " + f"{angle3kappa} {angle3angleAPO2}\n", angle_t3)   # Arms close MORE

with open(states_path / "atp_bound_1", 'w') as atp_bound_1_file:
    options = [
        bridge_soft_on,
        middle_site_soft_on
    ]
    apply(gen.write_script_bai_coeffs, atp_bound_1_file, options)

with open(states_path / "atp_bound_2", 'w') as atp_bound_2_file:
    options = [
        bridge_soft_off,
        middle_site_soft_off,
        bridge_on,
        top_site_on,
        middle_site_on,
        lower_site_on,
        arms_open,
        lower_compartment_folds1,
        lower_compartment_folds2
    ]
    apply(gen.write_script_bai_coeffs, atp_bound_2_file, options)


"""
plt.plot(   rHK[:,2],    rHK[:,1], '.')
plt.plot(  rATP[:,2],   rATP[:,1], '.')
plt.plot(rArmDL[:,2], rArmDL[:,1], '.')
plt.plot(rArmUL[:,2], rArmUL[:,1], '.')
plt.plot(rArmUR[:,2], rArmUR[:,1], '.')
plt.plot(rArmDR[:,2], rArmDR[:,1], '.')
#plt.plot(rSiteU[0,2], rSiteU[0,1], '.')
#plt.plot(rSiteD[0,2], rSiteD[0,1], '.')

plt.axis('scaled')
plt.show()
"""


#################################################################################
#                           Print to parameterfile                              #
#################################################################################


def get_variables_from_module(module):
    all_vars = dir(module)
    return list(filter(lambda name: not name.startswith("_"), all_vars))


with open(filepath_param, 'w') as parameterfile:
    parameterfile.write("# LAMMPS parameter file\n\n")

    params = get_variables_from_module(par)
    for key in params:
        parameterfile.write("variable %s equal %s\n\n"       %(key, getattr(par, key)))
