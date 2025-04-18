# Copyright (c) 2021 Stefanos Nomidis
# Copyright (c) 2022 Arwin Goossens
# Copyright (c) 2024 Lucas Dooms

import math
import numpy as np
from numpy.random import default_rng
from generator import AtomIdentifier, Generator, BAI_Type, BAI_Kind, AtomType, PairWise, MoleculeId
from sys import argv, maxsize
from pathlib import Path
from typing import Any, List, Tuple
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
ssDNAstiff = 5.

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

interaction_parameters = dna.InteractionParameters(
    sigmaDNAvsDNA=sigmaDNAvsDNA,
    epsilonDNAvsDNA=epsilonDNAvsDNA,
    rcutDNAvsDNA=rcutDNAvsDNA,
    sigmaSMCvsDNA=sigmaSMCvsDNA,
    epsilonSMCvsDNA=epsilonSMCvsDNA,
    rcutSMCvsDNA=rcutSMCvsDNA,
    sigmaSiteDvsDNA=sigmaSiteDvsDNA,
    rcutSiteDvsDNA=rcutSiteDvsDNA,
    epsilonSiteDvsDNA=epsilonSiteDvsDNA,
)

# Even More Parameters


# Relative bond fluctuations
bondFlDNA = 1e-2
bondFlSMC = 1e-2
# bondFlHinge = 0.5 # large fluctuations to allow tether passing
bondFlHinge = 3e-2 # small fluctuations

# Maximum relative bond extension (units of rest length)
bondMax = 1.

# Spring constant obeying equilibrium relative bond fluctuations
kBondDNA = 3 * kBT / (DNAbondLength * bondFlDNA)**2
kBondSMC = 3 * kBT / (SMCspacing * bondFlSMC)**2
kBondHinge = 3 * kBT / (SMCspacing * bondFlHinge)**2
kBondAlign1 = 10 * kBT / SMCspacing**2
kBondAlign2 = 200 * kBT / SMCspacing**2


# Maximum bond length
maxLengthDNA = DNAbondLength * bondMax
maxLengthSMC = SMCspacing * bondMax

# DNA bending rigidity
kDNA = DNAstiff * kBT / DNAbondLength
kssDNA = ssDNAstiff * kBT / DNAbondLength


#################################################################################
#                                 Start Setup                                   #
#################################################################################


dna.DnaConfiguration.set_parameters(par, interaction_parameters)
dnaConfigClass = dna.DnaConfiguration.str_to_config(par.dnaConfig)


#################################################################################
#                                 SMC complex                                   #
#################################################################################


smc_creator = SMC_Creator(
    SMCspacing=SMCspacing,

    siteUvDist=4.0,
    siteUhDist=2.0,
    siteMvDist=1.0,
    siteMhDist=2.0,
    siteDvDist=0.5,
    siteDhDist=2.0,

    armLength=par.armLength,
    bridgeWidth=par.bridgeWidth,
    hingeRadius=par.hingeRadius,
    # SMCspacing half of the minimal required spacing of ssDNA
    # so between 2*SMCspacing and 4*SMCspacing should
    # allow ssDNA passage but not dsDNA
    hinge_opening=2.2 * SMCspacing,

    HKradius=par.HKradius,

    foldingAngleAPO=par.foldingAngleAPO
)

rot_vec = np.array([0.0, 0.0, -np.deg2rad(42)]) if dnaConfigClass is dna.AdvancedObstacleSafety else None
rArmDL, rArmUL, rArmUR, rArmDR, rATP, rHK, rSiteU, rSiteM, rSiteD, rHinge = \
        smc_creator.get_smc(
            siteD_points_down=False,
            #dnaConfigClass in {dna.ObstacleSafety, dna.AdvancedObstacleSafety},
            extra_rotation=rot_vec
        )


#################################################################################
#                                     DNA                                       #
#################################################################################

# set DNA bonds, angles, and mass
molDNA = MoleculeId.get_next()
dna_bond = BAI_Type(BAI_Kind.BOND, "fene/expand %s %s %s %s %s\n" %(kBondDNA, maxLengthDNA, 0, 0, DNAbondLength))
dna_angle = BAI_Type(BAI_Kind.ANGLE, "cosine %s\n"        %  kDNA )
ssdna_angle = BAI_Type(BAI_Kind.ANGLE, "cosine %s\n"        %  kssDNA )
dna_type = AtomType(mDNA)

dna_parameters = dna.DnaParameters(
    nDNA=nDNA,
    DNAbondLength=DNAbondLength,
    mDNA=mDNA,
    type=dna_type,
    molDNA=molDNA,
    bond=dna_bond,
    angle=dna_angle,
    ssangle=ssdna_angle,
)
dnaConfig = dnaConfigClass.get_dna_config(dna_parameters, rSiteD, par)

#################################################################################
#                                Print to file                                  #
#################################################################################

# Divide total mass evenly among the segments
mSMC = smc_creator.get_mass_per_atom(mSMCtotal)


# SET UP DATAFILE GENERATOR
gen = Generator()
gen.set_system_size(boxWidth)

armHK_type = AtomType(mSMC)
hinge_type = AtomType(mSMC)
atp_type = AtomType(mSMC)
siteU_type = AtomType(mSMC)
siteM_type = AtomType(mSMC)
siteD_type = AtomType(mSMC)
refSite_type = AtomType(mSMC)


smc_1 = SMC(
    use_rigid_hinge=par.rigidHinge,

    rArmDL=rArmDL,
    rArmUL=rArmUL,
    rArmUR=rArmUR,
    rArmDR=rArmDR,
    rATP=rATP,
    rHK=rHK,
    rSiteU=rSiteU,
    rSiteM=rSiteM,
    rSiteD=rSiteD,
    rHinge=rHinge,

    armHK_type=armHK_type,
    hinge_type=hinge_type,
    atp_type=atp_type,
    siteU_type=siteU_type,
    siteM_type=siteM_type,
    siteD_type=siteD_type,
    refSite_type=refSite_type,

    k_bond = kBondSMC,
    k_hinge = kBondHinge,
    max_bond_length = maxLengthSMC,

    k_elbow = par.elbowsStiffness * kBT,
    k_arm = par.armsStiffness * kBT,

    k_align_site = par.siteStiffness * kBT,
    k_fold = par.foldingStiffness * kBT,
    k_asymmetry = par.asymmetryStiffness * kBT,

    bridge_width = par.bridgeWidth,
    arm_length = par.armLength,
    hinge_radius = par.hingeRadius,
    arms_angle_ATP = par.armsAngleATP,
    folding_angle_ATP = par.foldingAngleATP,
    folding_angle_APO = par.foldingAngleAPO,
)

dnaConfig.set_smc(smc_1)

if hasattr(dnaConfig, 'tether') and par.addRNAPolymerase:
    molBead = MoleculeId.get_next()
    bead_type = AtomType(10.0 * mDNA)
    bead_size = 3 # half of 6 (since it binds to DNA of two sides) -> ~ 6 * 1.7 nm ~ 10 nm
    if par.RNAPolymeraseType == 0:
        bead_bond = BAI_Type(BAI_Kind.BOND, "harmonic %s %s\n" %(kBondDNA, bead_size * DNAbondLength))
    elif par.RNAPolymeraseType == 1:
        bead_bond = None
    else:
        raise ValueError(f"unknown RNAPolymeraseType, {par.RNAPolymeraseType}")
    dna_id = dnaConfig.tether.dna_tether_id
    dnaConfig.add_bead_to_dna(bead_type, molBead, dna_id, bead_bond, bead_size)

    if bead_bond is None:
        gen.molecule_override[dna_id] = molBead

gen.atom_groups += [
    *dnaConfig.get_all_groups(),
    *smc_1.get_groups(),
]


# Pair coefficients
pair_inter = PairWise("PairIJ Coeffs # hybrid\n\n", "lj/cut {} {} {}\n", [0.0, 0.0, 0.0])
# prevent hinges from overlapping
pair_inter.add_interaction(hinge_type, hinge_type, epsilonSMCvsDNA * kBT, sigmaSMCvsDNA, rcutSMCvsDNA)
# prevent upper site from overlapping with arms
pair_inter.add_interaction(armHK_type, siteU_type, epsilonSMCvsDNA * kBT, sigmaSMCvsDNA, rcutSMCvsDNA)

dnaConfig.add_interactions(pair_inter)

# soft interactions
pair_soft_inter = PairWise("PairIJ Coeffs # hybrid\n\n", "soft {} {}\n", [0.0, 0.0])

gen.pair_interactions.append(pair_inter)
gen.pair_interactions.append(pair_soft_inter)

# Interactions that change for different phases of SMC
bridge_off = Generator.DynamicCoeffs(None, "{} {} lj/cut 0 0 0\n", [dna_type, atp_type])
bridge_on = Generator.DynamicCoeffs(None, "{} {} " + f"lj/cut {epsilonSMCvsDNA * kBT} {par.sigma} {par.sigma * 2**(1/6)}\n", [dna_type, atp_type])

bridge_soft_off = Generator.DynamicCoeffs(None, "{} {} soft 0 0\n", [dna_type, atp_type])
bridge_soft_on = Generator.DynamicCoeffs(None, "{} {} soft " + f"{epsilonSMCvsDNA * kBT} {par.sigma * 2**(1/6)}\n", [dna_type, atp_type])

hinge_attraction_off = Generator.DynamicCoeffs(None, "{} {} lj/cut 0 0 0\n", [dna_type, siteU_type])
hinge_attraction_on = Generator.DynamicCoeffs(None, "{} {} " + f"lj/cut {par.epsilon4 * kBT} {par.sigma} {par.cutoff4}\n", [dna_type, siteU_type])

if False: #isinstance(dnaConfig, (dna.ObstacleSafety, dna.AdvancedObstacleSafety))
    # always keep site on
    lower_site_off = Generator.DynamicCoeffs(None, "{} {} " + f"lj/cut {par.epsilon6 * kBT} {par.sigma} {par.cutoff6}\n", [dna_type, siteD_type])
else:
    lower_site_off = Generator.DynamicCoeffs(None, "{} {} lj/cut 0 0 0\n", [dna_type, siteD_type])
lower_site_on = Generator.DynamicCoeffs(None, "{} {} " + f"lj/cut {par.epsilon6 * kBT} {par.sigma} {par.cutoff6}\n", [dna_type, siteD_type])

middle_site_off = Generator.DynamicCoeffs(None, "{} {} lj/cut 0 0 0\n", [dna_type, siteM_type])
middle_site_on = Generator.DynamicCoeffs(None, "{} {} " + f"lj/cut {par.epsilon5 * kBT} {par.sigma} {par.cutoff5}\n", [dna_type, siteM_type])

middle_site_soft_off = Generator.DynamicCoeffs(None, "{} {} soft 0 0\n", [dna_type, siteM_type])
middle_site_soft_on = Generator.DynamicCoeffs(None, "{} {} soft " + f"{par.epsilon5 * kBT} {par.sigma * 2**(1/6)}\n", [dna_type, siteM_type])

gen.bais += [
    *smc_1.get_bonds(smc_creator.hinge_opening),
    *dnaConfig.get_bonds()
]

gen.bais += smc_1.get_angles()

gen.bais += smc_1.get_impropers()

# Override molecule ids to form rigid safety-belt bond
if isinstance(dnaConfig, (dna.ObstacleSafety, dna.AdvancedObstacleSafety)): #TODO
    safety_index = dnaConfig.dna_safety_belt_index
    gen.molecule_override[(dnaConfig.dna_groups[0], safety_index)] = smc_1.mol_lower_site
    # add neighbors to prevent rotation
    # gen.molecule_override[(dnaConfig.dna_groups[0], safety_index - 1)] = smc_1.mol_lower_site
    # gen.molecule_override[(dnaConfig.dna_groups[0], safety_index + 1)] = smc_1.mol_lower_site

with open(path / 'datafile_coeffs', 'w', encoding='utf-8') as datafile:
    # gen.write_full(datafile)
    gen.write_coeffs(datafile)

with open(path / 'datafile_positions', 'w', encoding='utf-8') as datafile:
    gen.write_positions_and_bonds(datafile)


#################################################################################
#                                Phases of SMC                                  #
#################################################################################

# make sure the directory exists
states_path = path / "states"
states_path.mkdir(exist_ok=True)

def create_phase(phase_path: Path, options: List[Generator.DynamicCoeffs]):
    def apply(function, file, list_of_args: List[Any]):
        for args in list_of_args:
            function(file, args)

    with open(phase_path, 'w', encoding='utf-8') as phase_file:
        apply(gen.write_script_bai_coeffs, phase_file, options)

create_phase(
    states_path / "adp_bound",
    [
        bridge_off,
        hinge_attraction_on,
        middle_site_off,
        lower_site_off,
        smc_1.arms_open,
        smc_1.kleisin_unfolds1,
        smc_1.kleisin_unfolds2,
    ]
)

create_phase(
    states_path / "apo",
    [
        bridge_off,
        hinge_attraction_off,
        middle_site_off,
        lower_site_on,
        smc_1.arms_close,
        smc_1.kleisin_unfolds1,
        smc_1.kleisin_unfolds2,
    ]
)
# gen.write_script_bai_coeffs(adp_bound_file, BAI_Kind.ANGLE, "{} harmonic " + f"{angle3kappa} {angle3angleAPO2}\n", angle_t3)   # Arms close MORE

create_phase(
    states_path / "atp_bound_1",
    [
        bridge_soft_on,
        middle_site_soft_on,
    ]
)

create_phase(
    states_path / "atp_bound_2",
    [
        bridge_soft_off,
        middle_site_soft_off,
        bridge_on,
        hinge_attraction_on,
        middle_site_on,
        lower_site_on,
        smc_1.arms_open,
        smc_1.kleisin_folds1,
        smc_1.kleisin_folds2,
    ]
)


#################################################################################
#                           Print to post processing                            #
#################################################################################

ppp = dnaConfig.get_post_process_parameters()

with open(path / "post_processing_parameters.py", 'w', encoding='utf-8') as file:
    file.write(
        "# use to form plane of SMC arms\n"
        "top_bead_id = {}\n"
        "left_bead_id = {}\n"
        "right_bead_id = {}\n"
        "middle_left_bead_id = {}\n"
        "middle_right_bead_id = {}\n".format(
            gen.get_atom_index((smc_1.arm_ul_grp, -1)),
            gen.get_atom_index((smc_1.arm_ul_grp, 0)),
            gen.get_atom_index((smc_1.arm_ur_grp, -1)),
            gen.get_atom_index((smc_1.atp_grp, 0)),
            gen.get_atom_index((smc_1.atp_grp, -1))
        )
    )
    file.write("\n")
    dna_indices_list = [
        (gen.get_atom_index(atomId1), gen.get_atom_index(atomId2))
        for (atomId1, atomId2) in ppp.dna_indices_list
    ]
    file.write(
        "# list of (min, max) of DNA indices for separate pieces to analyze\n"
        "dna_indices_list = {}\n".format(dna_indices_list)
    )
    file.write("\n")
    kleisin_ids_list = [
        gen.get_atom_index((smc_1.hk_grp, i))
         for i in range(len(smc_1.hk_grp.positions))
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

def get_index_def(name: str, values: List[str]) -> str:
    """define a LAMMPS universe"""
    return f'variable {name} index {list_to_space_str(values)}\n'

def get_times(apo: int, atp1: int, atp2: int, adp: int, rng_gen: np.random.Generator):
    # get run times for each SMC state
    # APO -> ATP1 -> ATP2 -> ADP -> ...

    def mult(x):
        # use 1.0 to get (0, 1] lower exclusive
        return -x * np.log(1.0 - rng_gen.uniform())

    return [math.ceil(mult(x)) for x in (apo, atp1, atp2, adp)]

def get_times_with_max_steps(rng_gen: np.random.Generator):
    run_steps = []

    def none_to_max(x):
        if x is None:
            return maxsize # very large number!
        return x

    cycles_left = none_to_max(par.cycles)
    max_steps = none_to_max(par.max_steps)

    cum_steps = 0
    while True: # use do while loop since run_steps should not be empty
        new_times = get_times(par.stepsAPO, 10000, par.stepsATP, par.stepsADP, rng_gen)
        run_steps += new_times

        cum_steps += sum(new_times)
        cycles_left -= 1

        if cycles_left <= 0 or cum_steps >= max_steps:
            break

    return run_steps


with open(path / 'parameterfile', 'w', encoding='utf-8') as parameterfile:
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
                smc_1.get_molecule_ids() \
                        + list(filter(lambda xyz: xyz is not None, [molBead if 'molBead' in globals() else None]))
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

    if isinstance(dnaConfig, (dna.Obstacle, dna.ObstacleSafety, dna.AdvancedObstacleSafety)) and isinstance(dnaConfig.tether.obstacle, dna.Tether.Wall):
        parameterfile.write(f"variable wall_y equal {dnaConfig.tether.group.positions[0][1]}\n")

        excluded = [gen.get_atom_index((dnaConfig.tether.group, 0)), gen.get_atom_index((dnaConfig.tether.group, 1))]
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

    # obstacle, if particle
    if hasattr(dnaConfig, "tether") and isinstance(dnaConfig.tether.obstacle, dna.Tether.Gold):
        obstacle_lammps_id = gen.get_atom_index((dnaConfig.tether.obstacle.group, 0))
        parameterfile.write(
            "variable obstacle_id equal {}\n".format(obstacle_lammps_id)
        )

    parameterfile.write("\n")

    # get run times for each SMC state
    # APO -> ATP1 -> ATP2 -> ADP -> ...
    rng = default_rng(par.seed)
    runtimes = get_times_with_max_steps(rng)

    parameterfile.write(
        get_index_def("runtimes", [str(x) for x in runtimes])
    )
