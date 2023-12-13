import math
import numpy as np
from generator import Generator, BAI, BAI_Type, BAI_Kind, AtomGroup, AtomType, PairWise
from sys import argv
from pathlib import Path
from dna_creator import get_dna_coordinates, get_dna_coordinates_twist
from smc_creator import SMC_Creator
from smc import SMC
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

rArmDL, rArmUL, rArmUR, rArmDR, rATP, rHK, rSiteU, rSiteM, rSiteD = \
    SMC_Creator(
        SMCspacing=SMCspacing,

        siteUhDist=siteUhDist,
        siteUvDist=siteUvDist,
        siteMhDist=siteMhDist,
        siteMvDist=siteMvDist,
        siteDhDist=siteDhDist,
        siteDvDist=siteDvDist,

        armLength=par.armLength,
        bridgeWidth=par.bridgeWidth,

        HKradius=par.HKradius,

        foldingAngleAPO=par.foldingAngleAPO
    ).get_smc()

nArmDL, nArmUL, nArmUR, nArmDR, nATP, nHK, nSiteU, nSiteM, nSiteD = \
    len(rArmDL), len(rArmUL), len(rArmUR), len(rArmDR), len(rATP), len(rHK), len(rSiteU), len(rSiteM), len(rSiteD)


#################################################################################
#                                     DNA                                       #
#################################################################################

fold_dna = True

if fold_dna:
    rDNA, nLowerDNA = get_dna_coordinates_twist(nDNA, DNAbondLength, 17)
else:
    rDNA, nLowerDNA = get_dna_coordinates(nDNA, DNAbondLength, 14, 10)

#################################################################################
#                               Shift DNA to SMC                                #
#################################################################################

if fold_dna:
    # make sure SMC contains DNA
    desired_y_pos = rSiteD[1][1] + 0.9 * par.cutoff6
    shift_y = desired_y_pos - rDNA[-1][1]
    desired_x_pos = rSiteD[1][0]
    shift_x = desired_x_pos - rDNA[-(nLowerDNA - 10)][0]
    shift = np.array([shift_x, shift_y, 0]).reshape(1, 3)
    rDNA += shift

    # get dna beads to freeze
    # closest to bottom
    distances = np.linalg.norm(rDNA - rSiteD[1], axis=1)
    closest_DNA_index_b = int(np.argmin(distances))
    # closest to middle
    distances = np.linalg.norm(rDNA - rSiteM[1], axis=1)
    closest_DNA_index_m = int(np.argmin(distances))
else:
    # make sure SMC touches the DNA at the lower site (siteD)
    desired_y_pos = rSiteD[1][1] - 0.9 * par.cutoff6
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
mSMC = mSMCtotal / ( nArmDL + nArmUL + nArmUR + nArmDR + nHK + nATP + nSiteU + nSiteM + nSiteD )

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


smc_1 = SMC(
    rArmDL=rArmDL,
    rArmUL=rArmUL,
    rArmUR=rArmUR,
    rArmDR=rArmDR,
    rATP=rATP,
    rHK=rHK,
    rSiteU=rSiteU,
    rSiteM=rSiteM,
    rSiteD=rSiteD,

    molArmDL=molArmDL,
    molArmUL=molArmUL,
    molArmUR=molArmUR,
    molArmDR=molArmDR,
    molHK=molHK,
    molATP=molATP,
    molSiteU=molSiteU,
    molSiteM=molSiteM,
    molSiteD=molSiteD,

    armHK_type=armHK_type,
    atp_type=atp_type,
    siteU_type=siteU_type,
    siteM_type=siteM_type,
    siteD_type=siteD_type,
    refSite_type=refSite_type,
)

smc_1_groups = smc_1.get_groups()

gen.atom_groups += [
    dna_group,
    *smc_1_groups
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

bonds = smc_1.get_bonds(bond_t2, bond_t3, bond_t4)

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

angles = smc_1.get_angles(angle_t2, angle_t3)
gen.bais += dna_angle_list + angles

# We impose zero improper angle
imp_t1 = BAI_Type(BAI_Kind.IMPROPER, "%s %s\n"   %( kAlignSite, 0 ) )
imp_t2 = BAI_Type(BAI_Kind.IMPROPER, "%s %s\n"   %( kFolding,   180 - par.foldingAngleAPO ) )
imp_t3 = BAI_Type(BAI_Kind.IMPROPER, "%s %s\n" %( kAsymmetry,  math.fabs(90 - par.foldingAngleAPO) ) )

gen.bais += smc_1.get_impropers(imp_t1, imp_t2, imp_t3)


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
    
    parameterfile.write(f"variable index1 equal {closest_DNA_index_b}\n")
    parameterfile.write(f"variable index2 equal {closest_DNA_index_m}\n")
