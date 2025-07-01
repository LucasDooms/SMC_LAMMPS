# Copyright (c) 2021 Stefanos Nomidis
# Copyright (c) 2022 Arwin Goossens
# Copyright (c) 2024 Lucas Dooms

import math
from collections.abc import Sequence
from pathlib import Path
from runpy import run_path
from sys import argv, maxsize
from typing import Any, List

import numpy as np
from numpy.random import default_rng

from smc_lammps.generate.default_parameters import Parameters
from smc_lammps.generate.generator import (
    AtomIdentifier,
    AtomType,
    BAI_Kind,
    BAI_Type,
    Generator,
    MoleculeId,
    PairWise,
)
from smc_lammps.generate.structures.dna import dna
from smc_lammps.generate.structures.smc.smc import SMC
from smc_lammps.generate.structures.smc.smc_creator import SMC_Creator
from smc_lammps.generate.util import create_phase, get_closest

if len(argv) < 2:
    raise ValueError("Provide a folder path")
path = Path(argv[1])

par = run_path((path / "parameters.py").as_posix())["p"]

assert isinstance(par, Parameters)

nDNA = par.N
bases_per_bead = par.n


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
basepair_mass = 2 * 3.1575 * 5.24e-4

# Effective bead mass (ag)
DNA_bead_mass = bases_per_bead * basepair_mass


#######
# SMC #
#######

# Total mass of SMC protein (ag)
SMC_total_mass = 0.25


#################################### Lengths ####################################


################
# Interactions #
################

# DNA-DNA repulsion radius (nm)
radius_DNA_DNA = 3.5


#######
# DNA #
#######

# Bending stiffness (nm)
DNA_persistence_length = 50.0
ssDNA_persistence_length = 1.0

# Base pair step (nm)
basepair_size = 0.34

# Effective bond length = DNA bead size (nm)
DNA_bond_length = bases_per_bead * basepair_size

# Total length of DNA (nm)
DNA_total_length = DNA_bond_length * nDNA


#######
# SMC #
#######

# Desirable SMC spacing (radius of 1 SMC bead is R = intRadSMCvsDNA)
# Equal to R:   Minimum diameter = sqrt(3)    = 1.73 R
# Equal to R/2: Minimum diameter = sqrt(15)/2 = 1.94 R
SMC_spacing = par.sigma_SMC_DNA / 2


##################
# Simulation box #
##################


# Width of cubic simulation box (nm)
box_width = 2 * DNA_total_length


################################## Interactions #################################


###########
# DNA-DNA #
###########

sigma_DNA_DNA = radius_DNA_DNA
epsilon_DNA_DNA = par.epsilon3
rcut_DNA_DNA = sigma_DNA_DNA * 2 ** (1 / 6)


###########
# SMC-DNA #
###########

sigma_SMC_DNA = par.sigma_SMC_DNA
epsilon_SMC_DNA = par.epsilon3
rcut_SMC_DNA = sigma_SMC_DNA * 2 ** (1 / 6)


#############
# Sites-DNA #
#############

# Sigma of LJ attraction (same as those of the repulsive SMC sites)
sigma_upper_site_DNA = sigma_SMC_DNA

# Cutoff distance of LJ attraction
rcut_upper_site_DNA = par.cutoff6

# Epsilon parameter of LJ attraction
epsilon_upper_site_DNA = par.epsilon6

interaction_parameters = dna.InteractionParameters(
    sigma_DNA_DNA=sigma_DNA_DNA,
    epsilon_DNA_DNA=epsilon_DNA_DNA,
    rcut_DNA_DNA=rcut_DNA_DNA,
    sigma_SMC_DNA=sigma_SMC_DNA,
    epsilon_SMC_DNA=epsilon_SMC_DNA,
    rcut_SMC_DNA=rcut_SMC_DNA,
    sigma_upper_site_DNA=sigma_upper_site_DNA,
    rcut_lower_site_DNA=rcut_upper_site_DNA,
    epsilon_upper_site_DNA=epsilon_upper_site_DNA,
)

# Even More Parameters


# Relative bond fluctuations
bond_fluctuation_DNA = 1e-2
bond_fluctuation_SMC = 1e-2
# bond_fluctuation_hinge = 0.5 # large fluctuations to allow tether passing
bond_fluctuation_hinge = 3e-2  # small fluctuations

# Maximum relative bond extension (units of rest length)
bond_max_extension = 1.0

# Spring constant obeying equilibrium relative bond fluctuations
k_bond_DNA = 3 * kBT / (DNA_bond_length * bond_fluctuation_DNA) ** 2
k_bond_SMC = 3 * kBT / (SMC_spacing * bond_fluctuation_SMC) ** 2
if par.use_toroidal_hinge:
    k_bond_hinge = 3 * kBT / (SMC_spacing * bond_fluctuation_hinge) ** 2
else:
    k_bond_hinge = 10 * kBT / SMC_spacing**2


# Maximum bond length
max_bond_length_DNA = DNA_bond_length * bond_max_extension
max_bond_length_SMC = SMC_spacing * bond_max_extension

# DNA bending rigidity
k_angle_DNA = DNA_persistence_length * kBT / DNA_bond_length
k_angle_ssDNA = ssDNA_persistence_length * kBT / DNA_bond_length


#################################################################################
#                                 Start Setup                                   #
#################################################################################


dna.DnaConfiguration.set_parameters(par, interaction_parameters)
dna_config_class = dna.DnaConfiguration.str_to_config(par.dna_config)


#################################################################################
#                                 SMC complex                                   #
#################################################################################


smc_creator = SMC_Creator(
    SMC_spacing=SMC_spacing,
    #
    upper_site_v=4.0,
    upper_site_h=2.0,
    middle_site_v=1.0,
    middle_site_h=2.0,
    lower_site_v=0.5,
    lower_site_h=2.0,
    #
    arm_length=par.arm_length,
    bridge_width=par.bridge_width,
    use_toroidal_hinge=par.use_toroidal_hinge,
    hinge_radius=par.hinge_radius,
    # SMCspacing half of the minimal required spacing of ssDNA
    # so between 2*SMCspacing and 4*SMCspacing should
    # allow ssDNA passage but not dsDNA
    hinge_opening=2.2 * SMC_spacing,
    #
    kleisin_radius=par.kleisin_radius,
    folding_angle_APO=par.folding_angle_APO,
)

rot_vec = (
    np.array([0.0, 0.0, -np.deg2rad(42)])
    if dna_config_class is dna.AdvancedObstacleSafety
    else None
)
smc_positions = smc_creator.get_smc(
    lower_site_points_down=False,
    # dnaConfigClass in {dna.ObstacleSafety, dna.AdvancedObstacleSafety},
    extra_rotation=rot_vec,
)


#################################################################################
#                                     DNA                                       #
#################################################################################

# set DNA bonds, angles, and mass
mol_DNA = MoleculeId.get_next()
dna_bond = BAI_Type(
    BAI_Kind.BOND,
    "fene/expand",
    f"{k_bond_DNA} {max_bond_length_DNA} {0.0} {0.0} {DNA_bond_length}\n",
)
dna_angle = BAI_Type(BAI_Kind.ANGLE, "cosine", f"{k_angle_DNA}\n")
ssdna_angle = BAI_Type(BAI_Kind.ANGLE, "cosine", f"{k_angle_ssDNA}\n")
dna_type = AtomType(DNA_bead_mass)

dna_parameters = dna.DnaParameters(
    nDNA=nDNA,
    DNA_bond_length=DNA_bond_length,
    DNA_mass=DNA_bead_mass,
    type=dna_type,
    mol_DNA=mol_DNA,
    bond=dna_bond,
    angle=dna_angle,
    ssangle=ssdna_angle,
)
dna_config = dna_config_class.get_dna_config(dna_parameters, smc_positions.r_lower_site, par)

#################################################################################
#                                Print to file                                  #
#################################################################################

# Divide total mass evenly among the segments
mSMC = smc_creator.get_mass_per_atom(SMC_total_mass)


# SET UP DATAFILE GENERATOR
gen = Generator()
gen.set_system_size(box_width)
gen.use_charges = par.use_charges
if gen.use_charges:
    # prevents inf/nan in coul calculations
    shift_rng = default_rng(par.seed)
    gen.random_shift = lambda: shift_rng.normal(0, 1e-6 * DNA_bond_length, (3,))

smc_1 = SMC(
    use_rigid_hinge=par.rigid_hinge,
    pos=smc_positions,
    #
    t_arms_heads_kleisin=AtomType(mSMC),
    t_hinge=AtomType(mSMC, unused=not par.use_toroidal_hinge),
    t_atp=AtomType(mSMC),
    t_upper_site=AtomType(mSMC),
    t_middle_site=AtomType(mSMC),
    t_lower_site=AtomType(mSMC),
    t_ref_site=AtomType(mSMC),
    #
    k_bond=k_bond_SMC,
    k_hinge=k_bond_hinge,
    max_bond_length=max_bond_length_SMC,
    #
    k_elbow=par.elbows_stiffness * kBT,
    k_arm=par.arms_stiffness * kBT,
    #
    k_align_site=par.site_stiffness * kBT,
    k_fold=par.folding_stiffness * kBT,
    k_asymmetry=par.asymmetry_stiffness * kBT,
    #
    bridge_width=par.bridge_width,
    arm_length=par.arm_length,
    _hinge_radius=par.hinge_radius,
    arms_angle_ATP=par.arms_angle_ATP,
    folding_angle_ATP=par.folding_angle_ATP,
    folding_angle_APO=par.folding_angle_APO,
)

dna_config.set_smc(smc_1)

extra_mols_smc: List[int] = []
extra_mols_dna: List[int] = []

if par.add_RNA_polymerase:
    mol_bead = MoleculeId.get_next()
    bead_type = AtomType(10.0 * DNA_bead_mass)
    bead_size = par.RNA_polymerase_size

    if par.RNA_polymerase_type == 0:
        bead_bond = BAI_Type(BAI_Kind.BOND, "harmonic", f"{k_bond_DNA} {bead_size}\n")
        bead_angle = dna_angle
        extra_mols_dna.append(mol_bead)
    elif par.RNA_polymerase_type == 1:
        bead_bond = None
        bead_angle = None
        extra_mols_smc.append(mol_bead)
    else:
        raise ValueError(f"unknown RNA_polymerase_type, {par.RNA_polymerase_type}")

    if hasattr(dna_config, "tether"):
        dna_id = dna_config.tether.dna_tether_id
    else:
        dna_id = (
            dna_config.dna_groups[0],
            int(len(dna_config.dna_groups[0].positions) // 2),
        )
    dna_config.add_bead_to_dna(bead_type, mol_bead, dna_id, bead_bond, bead_angle, bead_size)

    if bead_bond is None:
        gen.molecule_override[dna_id] = mol_bead

if par.spaced_beads_interval is not None:
    spaced_bead_type = AtomType(DNA_bead_mass)

    # get spacing
    start_id = par.spaced_beads_interval
    stop_id = get_closest(dna_config.dna_groups[0].positions, smc_positions.r_lower_site[1])
    spaced_bead_ids = list(range(start_id, stop_id, par.spaced_beads_interval))
    spaced_bead_ids += list(
        range(
            stop_id + par.spaced_beads_interval,
            len(dna_config.dna_groups[0].positions),
            par.spaced_beads_interval,
        )
    )

    for dna_id in spaced_bead_ids:
        mol_spaced_bead = MoleculeId.get_next()
        extra_mols_smc.append(mol_spaced_bead)
        dna_id = (dna_config.dna_groups[0], dna_id)
        dna_config.add_bead_to_dna(
            spaced_bead_type,
            mol_spaced_bead,
            dna_id,
            None,
            None,
            par.spaced_beads_size,
        )
        gen.molecule_override[dna_id] = mol_spaced_bead

if par.add_stopper_bead:
    mol_stopper = MoleculeId.get_next()
    extra_mols_smc.append(mol_stopper)
    stopper_type = AtomType(0.01 * DNA_bead_mass)
    stopper_size = 25.0

    stopper_ids = dna_config.get_stopper_ids()
    for dna_id in stopper_ids:
        dna_config.add_bead_to_dna(stopper_type, mol_stopper, dna_id, None, None, stopper_size)
        gen.molecule_override[dna_id] = mol_stopper


gen.add_atom_groups(
    *dna_config.get_all_groups(),
    *smc_1.get_groups(),
)


# Pair coefficients
pair_inter = PairWise("PairIJ Coeffs # hybrid\n\n", "lj/cut {} {} {}\n", [0.0, 0.0, 0.0])

dna_config.add_interactions(pair_inter)
smc_1.add_repel_interactions(pair_inter, epsilon_SMC_DNA * kBT, sigma_SMC_DNA, rcut_SMC_DNA)

# soft interactions
pair_soft_inter = PairWise("PairIJ Coeffs # hybrid\n\n", "soft {} {}\n", [0.0, 0.0])

gen.pair_interactions.append(pair_inter)
gen.pair_interactions.append(pair_soft_inter)
if gen.use_charges:
    gen.pair_interactions.append(PairWise("PairIJ Coeffs # hybrid\n\n", "coul/debye {}\n", [""]))

# Interactions that change for different phases of SMC
bridge_off = Generator.DynamicCoeffs(None, "lj/cut 0 0 0\n", [dna_type, smc_1.t_atp])
bridge_on = Generator.DynamicCoeffs(
    None,
    f"lj/cut {epsilon_SMC_DNA * kBT} {par.sigma} {par.sigma * 2 ** (1 / 6)}\n",
    [dna_type, smc_1.t_atp],
)

bridge_soft_off = Generator.DynamicCoeffs(None, "soft 0 0\n", [dna_type, smc_1.t_atp])
bridge_soft_on = Generator.DynamicCoeffs(
    None,
    f"soft {epsilon_SMC_DNA * kBT} {par.sigma * 2 ** (1 / 6)}\n",
    [dna_type, smc_1.t_atp],
)

hinge_attraction_off = Generator.DynamicCoeffs(
    None, "lj/cut 0 0 0\n", [dna_type, smc_1.t_upper_site]
)
hinge_attraction_on = Generator.DynamicCoeffs(
    None,
    f"lj/cut {par.epsilon4 * kBT} {par.sigma} {par.cutoff4}\n",
    [dna_type, smc_1.t_upper_site],
)

if False:  # isinstance(dnaConfig, (dna.ObstacleSafety, dna.AdvancedObstacleSafety))
    # always keep site on
    lower_site_off = Generator.DynamicCoeffs(
        None,
        f"lj/cut {par.epsilon6 * kBT} {par.sigma} {par.cutoff6}\n",
        [dna_type, smc_1.t_lower_site],
    )
else:
    lower_site_off = Generator.DynamicCoeffs(None, "lj/cut 0 0 0\n", [dna_type, smc_1.t_lower_site])
lower_site_on = Generator.DynamicCoeffs(
    None,
    f"lj/cut {par.epsilon6 * kBT} {par.sigma} {par.cutoff6}\n",
    [dna_type, smc_1.t_lower_site],
)

middle_site_off = Generator.DynamicCoeffs(None, "lj/cut 0 0 0\n", [dna_type, smc_1.t_middle_site])
middle_site_on = Generator.DynamicCoeffs(
    None,
    f"lj/cut {par.epsilon5 * kBT} {par.sigma} {par.cutoff5}\n",
    [dna_type, smc_1.t_middle_site],
)

middle_site_soft_off = Generator.DynamicCoeffs(None, "soft 0 0\n", [dna_type, smc_1.t_middle_site])
middle_site_soft_on = Generator.DynamicCoeffs(
    None,
    "soft " + f"{par.epsilon5 * kBT} {par.sigma * 2 ** (1 / 6)}\n",
    [dna_type, smc_1.t_middle_site],
)

gen.bais += [*smc_1.get_bonds(smc_creator.hinge_opening), *dna_config.get_bonds()]

gen.bais += smc_1.get_angles()

gen.bais += smc_1.get_impropers()

# Override molecule ids to form rigid safety-belt bond
if isinstance(dna_config, (dna.ObstacleSafety, dna.AdvancedObstacleSafety, dna.Safety)):  # TODO
    safety_index = dna_config.dna_safety_belt_index
    gen.molecule_override[(dna_config.dna_groups[0], safety_index)] = smc_1.mol_lower_site
    # add neighbors to prevent rotation
    # gen.molecule_override[(dnaConfig.dna_groups[0], safety_index - 1)] = smc_1.mol_lower_site
    # gen.molecule_override[(dnaConfig.dna_groups[0], safety_index + 1)] = smc_1.mol_lower_site

with open(path / "datafile_coeffs", "w", encoding="utf-8") as datafile:
    gen.write_coeffs(datafile)

with open(path / "datafile_positions", "w", encoding="utf-8") as datafile:
    gen.write_positions_and_bonds(datafile)

with open(path / "styles", "w", encoding="utf-8") as stylesfile:
    stylesfile.write(gen.get_atom_style_command())
    stylesfile.write(gen.get_BAI_styles_command())
    pair_style = "pair_style hybrid/overlay lj/cut $(3.5) soft $(3.5)"
    if gen.use_charges:
        pair_style += " coul/debye $(1.0/5.0) $(7.5)"
    stylesfile.write(pair_style)

# VMD visualization of initial configuration
with open(path / "vmd.tcl", "w", encoding="utf-8") as vmdfile:
    vmdfile.write(f"topo readlammpsdata {path / 'datafile_positions'}")

#################################################################################
#                                Phases of SMC                                  #
#################################################################################

# make sure the directory exists
states_path = path / "states"
states_path.mkdir(exist_ok=True)

create_phase(
    gen,
    states_path / "adp_bound",
    [
        bridge_off,
        hinge_attraction_on,
        middle_site_off,
        lower_site_off,
        smc_1.arms_open,
        smc_1.kleisin_unfolds1,
        smc_1.kleisin_unfolds2,
    ],
)

create_phase(
    gen,
    states_path / "apo",
    [
        bridge_off,
        hinge_attraction_off,
        middle_site_off,
        lower_site_on,
        smc_1.arms_close,
        smc_1.kleisin_unfolds1,
        smc_1.kleisin_unfolds2,
    ],
)
# gen.write_script_bai_coeffs(adp_bound_file, BAI_Kind.ANGLE, "{} harmonic " + f"{angle3kappa} {angle3angleAPO2}\n", angle_t3)   # Arms close MORE

create_phase(
    gen,
    states_path / "atp_bound_1",
    [
        bridge_soft_on,
        middle_site_soft_on,
    ],
)

create_phase(
    gen,
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
    ],
)


#################################################################################
#                           Print to post processing                            #
#################################################################################

ppp = dna_config.get_post_process_parameters()

with open(path / "post_processing_parameters.py", "w", encoding="utf-8") as file:
    file.write(
        "# use to form plane of SMC arms\n"
        f"top_left_bead_id = {gen.get_atom_index((smc_1.arm_ul_grp, -1))}\n"
        f"top_right_bead_id = {gen.get_atom_index((smc_1.arm_ur_grp, 0))}\n"
        f"left_bead_id = {gen.get_atom_index((smc_1.arm_dl_grp, -1))}\n"
        f"right_bead_id = {gen.get_atom_index((smc_1.arm_dr_grp, 0))}\n"
        f"middle_left_bead_id = {gen.get_atom_index((smc_1.atp_grp, 0))}\n"
        f"middle_right_bead_id = {gen.get_atom_index((smc_1.atp_grp, -1))}\n"
    )
    file.write("\n")
    dna_indices_list = [
        (gen.get_atom_index(atomId1), gen.get_atom_index(atomId2))
        for (atomId1, atomId2) in ppp.dna_indices_list
    ]
    file.write(
        "# list of (min, max) of DNA indices for separate pieces to analyze\n"
        f"dna_indices_list = {dna_indices_list}\n"
    )
    file.write("\n")
    kleisin_ids_list = [
        gen.get_atom_index((smc_1.hk_grp, i)) for i in range(len(smc_1.hk_grp.positions))
    ]
    file.write(f"# use to form plane of SMC kleisin\nkleisin_ids = {kleisin_ids_list}\n")
    file.write("\n")
    file.write(f"dna_spacing = {max_bond_length_DNA}\n")
    file.write("\n")
    file.write(f"DNA_types = {list(set(grp.type.index for grp in dna_config.dna_groups))}\n")
    file.write(f"SMC_types = {list(set(grp.type.index for grp in smc_1.get_groups()))}\n")


#################################################################################
#                           Print to parameterfile                              #
#################################################################################


def atomIds_to_LAMMPS_ids(lst: Sequence[AtomIdentifier]) -> List[int]:
    return [gen.get_atom_index(atomId) for atomId in lst]


def get_variables_for_lammps() -> List[str]:
    """returns variable names that are needed in LAMMPS script"""
    return [
        "T",
        "gamma",
        "seed",
        "output_steps",
        "epsilon3",
        "sigma",
        "timestep",
        "smc_force",
    ]


def list_to_space_str(lst: Sequence[Any], surround="") -> str:
    """turn list into space separated string
    example: [1, 2, 6] -> 1 2 6"""
    return " ".join([surround + str(val) + surround for val in lst])


def prepend_or_empty(string: str, prepend: str) -> str:
    """prepend something if the string is non-empty
    otherwise replace it with the string "empty"."""
    if string:
        return prepend + string
    return "empty"


def get_string_def(name: str, value: str) -> str:
    """define a LAMMPS string"""
    return f'variable {name} string "{value}"\n'


def get_universe_def(name: str, values: Sequence[Any]) -> str:
    """define a LAMMPS universe"""
    return f"""variable {name} universe {list_to_space_str(values, surround='"')}\n"""


def get_index_def(name: str, values: Sequence[Any]) -> str:
    """define a LAMMPS universe"""
    return f"variable {name} index {list_to_space_str(values)}\n"


def get_times(apo: int, atp1: int, atp2: int, adp: int, rng_gen: np.random.Generator) -> List[int]:
    # get run times for each SMC state
    # APO -> ATP1 -> ATP2 -> ADP -> ...

    def mult(x):
        # use 1.0 to get (0, 1] lower exclusive
        return -x * np.log(1.0 - rng_gen.uniform())

    return [math.ceil(mult(x)) for x in (apo, atp1, atp2, adp)]


def get_times_with_max_steps(parameters: Parameters, rng_gen: np.random.Generator) -> List[int]:
    run_steps = []

    def none_to_max(x):
        if x is None:
            return maxsize  # very large number!
        return x

    cycles_left = none_to_max(parameters.cycles)
    max_steps = none_to_max(parameters.max_steps)

    cum_steps = 0
    while True:  # use do while loop since run_steps should not be empty
        new_times = get_times(
            parameters.steps_APO,
            10000,
            parameters.steps_ATP,
            parameters.steps_ADP,
            rng_gen,
        )
        run_steps += new_times

        cum_steps += sum(new_times)
        cycles_left -= 1

        if cycles_left <= 0 or cum_steps >= max_steps:
            break

    return run_steps


with open(path / "parameterfile", "w", encoding="utf-8") as parameterfile:
    parameterfile.write("# LAMMPS parameter file\n\n")

    # change seed if arg 2 provided
    if len(argv) > 2:
        seed_overwrite = int(argv[2])
        par.seed = seed_overwrite
    params = get_variables_for_lammps()
    for key in params:
        parameterfile.write(f"variable {key} equal {getattr(par, key)}\n\n")

    # write molecule ids
    # NOTE: indices are allowed to be the same, LAMMPS will ignore duplicates
    parameterfile.write(
        get_string_def(
            "DNA_mols",
            list_to_space_str(
                list(
                    set(grp.molecule_index for grp in dna_config.get_all_groups())
                    - set(
                        grp.molecule_index for grp in dna_config.beads
                    )  # do not include RNA beads
                )
                + extra_mols_dna
            ),
        )
    )
    parameterfile.write(
        get_string_def(
            "SMC_mols",
            list_to_space_str(smc_1.get_molecule_ids() + extra_mols_smc),
        )
    )

    parameterfile.write("\n")

    # turn into LAMMPS indices
    end_points_LAMMPS = atomIds_to_LAMMPS_ids(ppp.end_points)
    parameterfile.write(
        get_string_def(
            "dna_end_points",
            prepend_or_empty(list_to_space_str(end_points_LAMMPS), "id "),
        )
    )

    # turn into LAMMPS indices
    freeze_indices_LAMMPS = atomIds_to_LAMMPS_ids(ppp.freeze_indices)
    parameterfile.write(
        get_string_def("indices", prepend_or_empty(list_to_space_str(freeze_indices_LAMMPS), "id "))
    )

    if isinstance(
        dna_config, (dna.Obstacle, dna.ObstacleSafety, dna.AdvancedObstacleSafety)
    ) and isinstance(dna_config.tether.obstacle, dna.Tether.Wall):
        parameterfile.write(f"variable wall_y equal {dna_config.tether.group.positions[0][1]}\n")

        excluded = [
            gen.get_atom_index((dna_config.tether.group, 0)),
            gen.get_atom_index((dna_config.tether.group, 1)),
        ]
        parameterfile.write(
            get_string_def("excluded", prepend_or_empty(list_to_space_str(excluded), "id "))
        )

    # forces
    stretching_forces_array_LAMMPS = {
        key: atomIds_to_LAMMPS_ids(val) for key, val in ppp.stretching_forces_array.items()
    }
    if stretching_forces_array_LAMMPS:
        parameterfile.write(
            f"variable stretching_forces_len equal {len(stretching_forces_array_LAMMPS)}\n"
        )
        sf_ids = [
            prepend_or_empty(list_to_space_str(lst), "id ")
            for lst in stretching_forces_array_LAMMPS.values()
        ]
        parameterfile.write(get_universe_def("stretching_forces_groups", sf_ids))
        sf_forces = [list_to_space_str(tup) for tup in stretching_forces_array_LAMMPS.keys()]
        parameterfile.write(get_universe_def("stretching_forces", sf_forces))

    # obstacle, if particle
    if hasattr(dna_config, "tether") and isinstance(dna_config.tether.obstacle, dna.Tether.Gold):
        obstacle_lammps_id = gen.get_atom_index((dna_config.tether.obstacle.group, 0))
        parameterfile.write(f"variable obstacle_id equal {obstacle_lammps_id}\n")

    parameterfile.write("\n")

    # get run times for each SMC state
    # APO -> ATP1 -> ATP2 -> ADP -> ...
    rng = default_rng(par.seed)
    runtimes = get_times_with_max_steps(par, rng)

    parameterfile.write(get_index_def("runtimes", runtimes))
