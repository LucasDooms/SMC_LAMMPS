import math
import numpy as np
from generator import AtomIdentifier, Generator, BAI, BAI_Type, BAI_Kind, AtomType, PairWise, MoleculeId
from sys import argv
from pathlib import Path
from typing import Any, List
from structures.dna import dna
from structures.smc.smc_creator import SMC_Creator
from structures.smc.smc import SMC
from runpy import run_path
import default_parameters


if len(argv) < 2:
    raise Exception("Provide a folder path")
path = Path(argv[1])

parameters = run_path((path / "parameters.py").as_posix())

class Parameters:

    def __getattr__(self, __name: str) -> Any:
        return parameters.get(__name , getattr(default_parameters, __name))

    def __setattr__(self, __name: str, __value: Any) -> None:
        parameters[__name] = __value

    def __dir__(self):
        return dir(default_parameters)

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

kBT = kB * T


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
boxWidth = 2 * DNAcontourLength


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
epsilonSiteDvsDNA = par.epsilon6


# Even More Parameters


# Relative bond fluctuations
bondFlDNA = 1e-2
bondFlSMC = 1e-2

# Maximum relative bond extension (units of rest length)
bondMax = 1.

# Spring constant obeying equilibrium relative bond fluctuations
kBondDNA = 3 * kBT / (DNAbondLength * bondFlDNA)**2
kBondSMC = 3 * kBT / (SMCspacing * bondFlSMC)**2
kBondAlign1 = 10 * kBT / SMCspacing**2
kBondAlign2 = 200 * kBT / SMCspacing**2


# Maximum bond length
maxLengthDNA = DNAbondLength * bondMax
maxLengthSMC = SMCspacing * bondMax

# DNA bending rigidity
kDNA = DNAstiff * kBT / DNAbondLength

# Angular trap constants
# kElbows: Bending of elbows (kinkable arms, hence soft)
# kArms:   Arms opening angle wrt ATP bridge (should be stiff)
kElbows = par.elbowsStiffness * kBT
kArms = par.armsStiffness * kBT

# Fixes site orientation (prevents free rotation, should be stiff)
kAlignSite = par.siteStiffness * kBT

# Folding stiffness of lower compartment (should be stiff)
kFolding = par.foldingStiffness * kBT

# Makes folding asymmetric (should be stiff)
kAsymmetry = par.asymmetryStiffness * kBT


#################################################################################
#                                 Start Setup                                   #
#################################################################################


dna.DnaConfiguration.set_parameters(par)
dnaConfigClass = dna.DnaConfiguration.str_to_config(par.dnaConfig)


#################################################################################
#                                 SMC complex                                   #
#################################################################################

smc_creator = SMC_Creator(
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
)

rArmDL, rArmUL, rArmUR, rArmDR, rATP, rHK, rSiteU, rSiteM, rSiteD = \
        smc_creator.get_smc(siteD_points_down=dnaConfigClass in {dna.ObstacleSafety})


#################################################################################
#                                     DNA                                       #
#################################################################################

def get_closest(array, position) -> int:
    """returns the index of the array that is closest to the given position"""
    distances = np.linalg.norm(array - position, axis=1)
    return int(np.argmin(distances))

# set DNA bonds, angles, and mass
molDNA = MoleculeId.get_next()
dna_bond = BAI_Type(BAI_Kind.BOND, "fene/expand %s %s %s %s %s\n" %(kBondDNA, maxLengthDNA, 0, 0, DNAbondLength))
dna_angle = BAI_Type(BAI_Kind.ANGLE, "cosine %s\n"        %  kDNA )
dna_type = AtomType(mDNA)

dna_parameters = dna.DnaParameters(
    nDNA=nDNA,
    DNAbondLength=DNAbondLength,
    mDNA=mDNA,
    dna_type=dna_type,
    molDNA=molDNA,
    dna_bond=dna_bond,
    dna_angle=dna_angle
)
dnaConfig = dna.get_dna_config(dnaConfigClass, dna_parameters, rSiteD, par)

#################################################################################
#                                Print to file                                  #
#################################################################################

# Divide total mass evenly among the segments
mSMC = smc_creator.get_mass_per_atom(mSMCtotal)


# get indices to bind top site to arms
indL = get_closest(rArmUL, rSiteU[-2])
indR = get_closest(rArmUR, rSiteU[-2])
# binds from the side of the arms to the shields of SiteU
bondMinArmUSide = np.linalg.norm(rSiteU[-2] - rArmUL[indL])
# binds from the top of the arms to the shields of SiteU
bondMinArmUTop = np.linalg.norm(rSiteU[-2] - rArmUL[-1])


# SET UP DATAFILE GENERATOR
gen = Generator()
gen.set_system_size(boxWidth)

# Molecule for each rigid body
molArmDL, molArmUL, molArmUR, molArmDR, molHK, molATP, molSiteU = [MoleculeId.get_next() for _ in range(7)]
molSiteM = molATP
molSiteD = molHK

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

dnaConfig.set_smc(smc_1)

smc_1_groups = smc_1.get_groups()

gen.atom_groups += [
    *dnaConfig.get_all_groups(),
    *smc_1_groups
]

# get post process stuff

# Pair coefficients
pair_inter = PairWise("PairIJ Coeffs # hybrid\n\n", "lj/cut {} {} {}\n", [0.0, 0.0, 0.0])

pair_inter.add_interaction(dna_type, dna_type, epsilonDNAvsDNA * kBT, sigmaDNAvsDNA, rcutDNAvsDNA)
pair_inter.add_interaction(dna_type, armHK_type, epsilonSMCvsDNA * kBT, sigmaSMCvsDNA, rcutSMCvsDNA)
pair_inter.add_interaction(dna_type, siteD_type, epsilonSiteDvsDNA * kBT, sigmaSiteDvsDNA, rcutSiteDvsDNA)
if isinstance(dnaConfig, (dna.Obstacle, dna.ObstacleSafety)):
    tether_type = dnaConfig.tether_group.type
    # tether
    pair_inter.add_interaction(tether_type, tether_type, epsilonDNAvsDNA * kBT, sigmaDNAvsDNA, rcutDNAvsDNA)
    pair_inter.add_interaction(tether_type, dna_type, epsilonDNAvsDNA * kBT, sigmaDNAvsDNA, rcutDNAvsDNA)
    pair_inter.add_interaction(tether_type, armHK_type, epsilonSMCvsDNA * kBT, sigmaSMCvsDNA, rcutSMCvsDNA)
    pair_inter.add_interaction(tether_type, siteD_type, epsilonSiteDvsDNA * kBT, sigmaSiteDvsDNA, rcutSiteDvsDNA)

# soft interactions
pair_soft_inter = PairWise("PairIJ Coeffs # hybrid\n\n", "soft {} {}\n", [0.0, 0.0])

gen.pair_interactions.append(pair_inter)
gen.pair_interactions.append(pair_soft_inter)

# Interactions that change for different phases of SMC
bridge_off = [None, "{} {} lj/cut 0 0 0\n", dna_type, atp_type]
bridge_on = [None, "{} {} " + f"lj/cut {epsilonSMCvsDNA * kBT} {par.sigma} {par.sigma * 2**(1/6)}\n", dna_type, atp_type]

bridge_soft_off = [None, "{} {} soft 0 0\n", dna_type, atp_type]
bridge_soft_on = [None, "{} {} soft " + f"{epsilonSMCvsDNA * kBT} {par.sigma * 2**(1/6)}\n", dna_type, atp_type]

top_site_off = [None, "{} {} lj/cut 0 0 0\n", dna_type, siteU_type]
top_site_on = [None, "{} {} " + f"lj/cut {par.epsilon4 * kBT} {par.sigma} {par.cutoff4}\n", dna_type, siteU_type]

if isinstance(dnaConfig, dna.ObstacleSafety):
    # always keep site on
    lower_site_off = [None, "{} {} " + f"lj/cut {par.epsilon6 * kBT} {par.sigma} {par.cutoff6}\n", dna_type, siteD_type]
else:
    lower_site_off = [None, "{} {} lj/cut 0 0 0\n", dna_type, siteD_type]
lower_site_on = [None, "{} {} " + f"lj/cut {par.epsilon6 * kBT} {par.sigma} {par.cutoff6}\n", dna_type, siteD_type]

middle_site_off = [None, "{} {} lj/cut 0 0 0\n", dna_type, siteM_type]
middle_site_on = [None, "{} {} " + f"lj/cut {par.epsilon5 * kBT} {par.sigma} {par.cutoff5}\n", dna_type, siteM_type]

middle_site_soft_off = [None, "{} {} soft 0 0\n", dna_type, siteM_type]
middle_site_soft_on = [None, "{} {} soft " + f"{par.epsilon5 * kBT} {par.sigma * 2**(1/6)}\n", dna_type, siteM_type]


# Every joint is kept in place through bonds
bond_t2 = BAI_Type(BAI_Kind.BOND, "fene/expand %s %s %s %s %s\n" %(kBondSMC, maxLengthSMC, 0, 0, 0))
bond_t3 = BAI_Type(BAI_Kind.BOND, "harmonic %s %s\n"             %(kBondAlign1, bondMinArmUSide))
bond_t4 = BAI_Type(BAI_Kind.BOND, "harmonic %s %s\n"             %(kBondAlign2, bondMinArmUTop))

bonds = smc_1.get_bonds(bond_t2, bond_t3, bond_t4, indL, indR)
gen.bais += bonds
if isinstance(dnaConfig, (dna.Obstacle, dna.ObstacleSafety)):
    tether_to_dna_bond = BAI(dna_bond, (dnaConfig.tether_group, -1), (dnaConfig.dna_groups[0], dnaConfig.dna_tether_id))
    gen.bais += [tether_to_dna_bond]

angle_t2 = BAI_Type(BAI_Kind.ANGLE, "harmonic %s %s\n" %( kElbows, 180 ) )
angle_t3 = BAI_Type(BAI_Kind.ANGLE, "harmonic %s %s\n" %( kArms,  np.rad2deg( math.acos( par.bridgeWidth / par.armLength ) ) ) )

angles = smc_1.get_angles(angle_t2, angle_t3)
gen.bais += angles

# Angle interactions that change for different phases of SMC
# angle3angleAPO1 = np.rad2deg(np.arccos(par.bridgeWidth / par.armLength))
# angle3angleAPO1 = np.rad2deg(np.arccos(2 * par.bridgeWidth / par.armLength))
arms_close = [BAI_Kind.ANGLE, "{} harmonic " + f"{kArms} {np.rad2deg(np.arccos(par.bridgeWidth / par.armLength))}\n", angle_t3]
arms_open = [BAI_Kind.ANGLE, "{} harmonic " + f"{kArms} {par.armsAngleATP}\n", angle_t3]

# We impose zero improper angle
imp_t1 = BAI_Type(BAI_Kind.IMPROPER, "%s %s\n" %( kAlignSite, 0 ) )
imp_t2 = BAI_Type(BAI_Kind.IMPROPER, "%s %s\n" %( kFolding, 180 - par.foldingAngleAPO ) )
imp_t3 = BAI_Type(BAI_Kind.IMPROPER, "%s %s\n" %( kAsymmetry, abs(90 - par.foldingAngleAPO) ) )

gen.bais += smc_1.get_impropers(imp_t1, imp_t2, imp_t3)


# Improper interactions that change for different phases of SMC
lower_compartment_folds1 = [BAI_Kind.IMPROPER, "{} " + f"{kFolding} {180 - par.foldingAngleATP}\n", imp_t2]
lower_compartment_unfolds1 = [BAI_Kind.IMPROPER, "{} " + f"{kFolding} {180 - par.foldingAngleAPO}\n", imp_t2]

lower_compartment_folds2 = [BAI_Kind.IMPROPER, "{} " + f"{kAsymmetry} {abs(90 - par.foldingAngleATP)}\n", imp_t3]
lower_compartment_unfolds2 = [BAI_Kind.IMPROPER, "{} " + f"{kAsymmetry} {abs(90 - par.foldingAngleAPO)}\n", imp_t3]


# Override molecule ids to form rigid safety-belt bond
if isinstance(dnaConfig, dna.ObstacleSafety):
    safety_index = dnaConfig.dna_safety_belt_index
    gen.molecule_override[(dnaConfig.dna_groups[0], safety_index)] = molSiteD
    # add neighbors to prevent rotation
    gen.molecule_override[(dnaConfig.dna_groups[0], safety_index - 1)] = molSiteD
    gen.molecule_override[(dnaConfig.dna_groups[0], safety_index + 1)] = molSiteD

# Create datafile
with open(filepath_data, 'w') as datafile:
    gen.write(datafile)


#################################################################################
#                                Phases of SMC                                  #
#################################################################################

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


#################################################################################
#                           Print to post processing                            #
#################################################################################


with open(path / "post_processing_parameters.py", 'w') as file:
    file.write(
        "# use to form plane of SMC arms\n"
        "top_bead_id = {}\n"
        "left_bead_id = {}\n"
        "right_bead_id = {}\n"
        "middle_left_bead_id = {}\n"
        "middle_right_bead_id = {}\n".format(
            gen.get_atom_index((smc_1.armUL_group, -1)),
            gen.get_atom_index((smc_1.armUL_group, 0)),
            gen.get_atom_index((smc_1.armUR_group, -1)),
            gen.get_atom_index((smc_1.atp_group, 0)),
            gen.get_atom_index((smc_1.atp_group, -1))
        )
    )
    file.write("\n")
    dna_indices_list = []
    for dna_grp in dnaConfig.dna_groups:
        if isinstance(dnaConfig, (dna.ObstacleSafety, dna.Line)):
            dna_indices_list.append(
                ( # take all DNA
                    gen.get_atom_index((dna_grp, 0)),
                    gen.get_atom_index((dna_grp, -1))
                )
            )
        elif isinstance(dnaConfig, dna.Obstacle):
            dna_indices_list.append(
                ( # take up to index where SMC is
                    gen.get_atom_index((dna_grp, 0)),
                    gen.get_atom_index((dna_grp, dnaConfig.dna_start_index))
                )
            )
        else:
            dna_indices_list.append(
                (
                    gen.get_atom_index((dna_grp, 0)), # min = start (starts at upper DNA, which we want)
                    gen.get_atom_index((dna_grp, len(dna_grp.positions) // 2)) # max = half way point (so that lower DNA is not included)
                )
            )
    file.write(
        "# list of (min, max) of DNA indices for separate pieces to analyze\n"
        "dna_indices_list = {}\n".format(dna_indices_list)
    )
    file.write("\n")
    kleisin_ids_list = [
        gen.get_atom_index((smc_1.hk_group, i))
         for i in range(len(smc_1.hk_group.positions))
    ]
    file.write(
        "# use to form plane of SMC kleisin\n"
        "kleisin_ids = {}\n".format(kleisin_ids_list)
    )
    file.write("\n")
    file.write(
        "dna_spacing = {}\n".format(maxLengthDNA)
    )


#################################################################################
#                           Print to parameterfile                              #
#################################################################################

def atomIds_to_LAMMPS_ids(lst: List[AtomIdentifier]) -> List[int]:
    return [gen.get_atom_index(atomId) for atomId in lst]

def get_variables_from_module(module):
    all_vars = dir(module)
    return list(filter(lambda name: not name.startswith("_"), all_vars))

def list_to_space_str(lst) -> str:
    """turn list into space separated string
    example: [1, 2, 6] -> 1 2 6"""
    return " ".join(map(str, lst))

def prepend_or_empty(string: str, prepend: str) -> str:
    """prepend something if the string is non-empty
       otherwise replace it with the string "empty"."""
    if string:
        return prepend + string
    return "empty"

def get_string_def(name: str, value: str) -> str:
    """define a LAMMPS string"""
    return f'variable {name} string "{value}"\n'

def get_universe_def(name: str, values: List[str]) -> str:
    """define a LAMMPS universe"""
    values = ['"' + value + '"' for value in values]
    return f'variable {name} universe {list_to_space_str(values)}\n'


ppp = dnaConfig.get_post_process_parameters()
with open(filepath_param, 'w') as parameterfile:
    parameterfile.write("# LAMMPS parameter file\n\n")

    # change seed if arg 2 provided
    if len(argv) > 2:
        seed_overwrite = int(argv[2])
        par.seed = seed_overwrite
    params = get_variables_from_module(par)
    for key in params:
        parameterfile.write("variable %s equal %s\n\n"       %(key, getattr(par, key)))

    # write molecule ids
    # NOTE: indices are allowed to be the same, LAMMPS will ignore duplicates
    parameterfile.write(
        get_string_def("DNA_mols",
            list_to_space_str(
                [grp.molecule_index for grp in dnaConfig.get_all_groups()]
            )
        )
    )
    parameterfile.write(
        get_string_def("SMC_mols",
            list_to_space_str(
                [molArmDL, molArmUL, molArmUR, molArmDR, molHK, molATP, molSiteU, molSiteM, molSiteD]
            )
        )
    )

    parameterfile.write("\n")

    # turn into LAMMPS indices
    end_points_LAMMPS = atomIds_to_LAMMPS_ids(ppp.end_points)
    parameterfile.write(
        get_string_def("dna_end_points",
            prepend_or_empty(list_to_space_str(end_points_LAMMPS), "id ")
        )
    )

    # turn into LAMMPS indices
    freeze_indices_LAMMPS = atomIds_to_LAMMPS_ids(ppp.freeze_indices)
    parameterfile.write(
        get_string_def("indices",
            prepend_or_empty(list_to_space_str(freeze_indices_LAMMPS), "id ")
        )
    )
    
    if isinstance(dnaConfig, (dna.Obstacle, dna.ObstacleSafety)):
        parameterfile.write(f"variable wall_y equal {dnaConfig.tether_group.positions[0][1]}\n")

        excluded = [gen.get_atom_index((dnaConfig.tether_group, 0)), gen.get_atom_index((dnaConfig.tether_group, 1))]
        parameterfile.write(
            get_string_def("excluded",
                prepend_or_empty(list_to_space_str(excluded), "id ")
            )
        )

    # forces
    stretching_forces_array_LAMMPS = {key: atomIds_to_LAMMPS_ids(val) for key, val in ppp.stretching_forces_array.items()}
    if stretching_forces_array_LAMMPS:
        parameterfile.write(f"variable stretching_forces_len equal {len(stretching_forces_array_LAMMPS)}\n")
        sf_ids = [prepend_or_empty(list_to_space_str(lst), "id ") for lst in stretching_forces_array_LAMMPS.values()]
        parameterfile.write(
            get_universe_def(
                "stretching_forces_groups",
                sf_ids
            )
        )
        sf_forces = [list_to_space_str(tup) for tup in stretching_forces_array_LAMMPS.keys()]
        parameterfile.write(
            get_universe_def(
                "stretching_forces",
                sf_forces
            )
        )
