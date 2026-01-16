# Copyright (c) 2021 Stefanos Nomidis
# Copyright (c) 2022 Arwin Goossens
# Copyright (c) 2024-2026 Lucas Dooms

import math
from pathlib import Path
from sys import argv

import numpy as np
from numpy.random import default_rng

from smc_lammps.console import warn
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
from smc_lammps.generate.lammps.parameterfile import (
    get_def_dynamically,
    get_index_def,
    get_string_def,
    get_universe_def,
    list_to_space_str,
    prepend_or_empty,
)
from smc_lammps.generate.lammps.runtimes import get_times_with_max_steps
from smc_lammps.generate.lammps.util import atomIds_to_LAMMPS_ids
from smc_lammps.generate.structures.dna import dna
from smc_lammps.generate.structures.smc.smc import SMC
from smc_lammps.generate.structures.smc.smc_creator import SMC_Creator
from smc_lammps.generate.util import create_phase_wrapper, get_closest, load_parameters


def parse_inputs(argv: list[str]) -> tuple[Path, Parameters]:
    if len(argv) < 2:
        raise ValueError("Provide a folder path")

    path = Path(argv[1])
    if not path.exists():
        raise ValueError(f"Path '{path}' not found!")

    par = load_parameters(path)

    # change seed if arg 2 provided
    if len(argv) > 2:
        try:
            seed_overwrite = int(argv[2])
        except Exception as e:
            raise ValueError(f"Invalid seed received!\n{e}")
        else:
            par.seed = seed_overwrite

    return path, par


path, par = parse_inputs(argv)

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

# fmt: off
basepair_mass = (
    2         # pair = two bases
    * 315.75  # average mass of base in Da (Dalton)
    * 1.66054 * 1e-6 # ag / Da (attograms per Dalton)
)
# fmt: on

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
radius_DNA_DNA = 2.0


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


interaction_parameters = dna.InteractionParameters(
    sigma_DNA_DNA=sigma_DNA_DNA,
    epsilon_DNA_DNA=epsilon_DNA_DNA,
    rcut_DNA_DNA=rcut_DNA_DNA,
    k_bond_DNA_DNA=k_bond_DNA,
    sigma_SMC_DNA=sigma_SMC_DNA,
    epsilon_SMC_DNA=epsilon_SMC_DNA,
    rcut_SMC_DNA=rcut_SMC_DNA,
    sigma_upper_site_DNA=sigma_upper_site_DNA,
    rcut_lower_site_DNA=rcut_upper_site_DNA,
    epsilon_upper_site_DNA=epsilon_upper_site_DNA,
)

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
    add_side_site=par.add_side_site,
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
stiff_dna_angle = BAI_Type(
    BAI_Kind.ANGLE, "cosine", f"{par.spaced_beads_custom_stiffness * k_angle_DNA}\n"
)
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
gen.use_charges = par.use_charges
if gen.use_charges:
    # prevents inf/nan in coul calculations
    shift_rng = default_rng(par.seed)
    gen.random_shift = lambda: shift_rng.normal(0, 1e-6 * DNA_bond_length, (3,))

smc_1 = SMC(
    use_rigid_hinge=par.rigid_hinge,
    pos=smc_positions,
    #
    t_arms_heads=AtomType(mSMC),
    t_kleisin=AtomType(mSMC),
    t_shield=AtomType(mSMC, unused=not par.add_side_site),  # currently only used for side site
    t_hinge=AtomType(mSMC, unused=not par.use_toroidal_hinge),
    t_atp=AtomType(mSMC),
    t_upper_site=AtomType(mSMC),
    t_middle_site=AtomType(mSMC),
    t_lower_site=AtomType(mSMC),
    t_ref_site=AtomType(mSMC),
    t_side_site=AtomType(mSMC, unused=not par.add_side_site),
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
    elbow_attraction=par.elbow_attraction,
    elbow_spacing=par.elbow_spacing,
)

dna_config.set_smc(smc_1)

# lock the SMC position relative to the DNA polymer
closest_id = get_closest(dna_config.dna_strands[0].full_list(), smc_positions.r_lower_site[1])
dna_config.dna_strands[0].add_tagged_atom_groups(
    dna_config.dna_strands[0].get_id_from_list_index(closest_id), *smc_1.get_groups()
)

extra_mols_smc: list[int] = []
extra_mols_dna: list[int] = []

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

    if isinstance(dna_config.tether, dna.Tether):
        st_dna_id = dna_config.tether.dna_tether_id
    else:
        st_dna_id = (0, int(0.5 * dna_config.dna_strands[0].full_list_length()))
    dna_config.add_bead_to_dna(
        bead_type, mol_bead, st_dna_id[0], st_dna_id[1], bead_bond, bead_angle, bead_size
    )

spaced_beads: list[AtomIdentifier] = []
if par.spaced_beads_interval is not None:
    # get mass based on size
    # (0.22 = arbitrary factor chosen to approximate nucleosome mass at 5.5 nm radius)
    spaced_bead_type = AtomType(
        0.22 * DNA_bead_mass * (2.0 * par.spaced_beads_size / DNA_bond_length) ** 3
    )

    # get spacing
    start_id = par.spaced_beads_interval
    stop_id = get_closest(dna_config.dna_strands[0].full_list(), smc_positions.r_lower_site[1])
    clearance = math.ceil(par.spaced_beads_smc_clearance / DNA_bond_length)
    spaced_bead_ids = list(range(start_id, stop_id - clearance, par.spaced_beads_interval))

    if par.spaced_beads_full_dna:
        spaced_bead_ids += list(
            range(
                stop_id + clearance,
                dna_config.dna_strands[0].full_list_length(),
                par.spaced_beads_interval,
            )
        )

    for st_dna_id in spaced_bead_ids:
        mol_spaced_bead = MoleculeId.get_next()
        if par.spaced_beads_type == 0:
            extra_mols_dna.append(mol_spaced_bead)
            # use the same bond strength for now
            k_bond_spaced_bead = k_bond_DNA
            bead_dna_bond = BAI_Type(
                BAI_Kind.BOND,
                "fene/expand",
                f"{k_bond_spaced_bead} {par.spaced_beads_size} {0.0} {0.0} {par.spaced_beads_size}\n",
            )

            k_angle_spaced_bead = DNA_persistence_length * kBT / par.spaced_beads_size
            spaced_bead_angle = BAI_Type(BAI_Kind.ANGLE, "cosine", f"{k_angle_spaced_bead}\n")
            spaced_beads.append(
                dna_config.add_bead_to_dna(
                    spaced_bead_type,
                    mol_spaced_bead,
                    0,
                    st_dna_id,
                    bead_dna_bond,
                    spaced_bead_angle,
                    par.spaced_beads_size,
                )
            )
        elif par.spaced_beads_type == 1:
            extra_mols_smc.append(mol_spaced_bead)
            spaced_beads.append(
                dna_config.add_bead_to_dna(
                    spaced_bead_type,
                    mol_spaced_bead,
                    0,
                    st_dna_id,
                    None,
                    None,
                    par.spaced_beads_size,
                )
            )
        else:
            raise ValueError(f"unknown spaced_beads_type, {par.spaced_beads_type}")

        if par.spaced_beads_custom_stiffness != 1.0:
            offset = int(par.spaced_beads_size / DNA_bond_length)
            try:
                dna_config.change_dna_stiffness(
                    0, st_dna_id - offset, st_dna_id + offset + 1, dna_bond, stiff_dna_angle
                )
            except ValueError as e:
                print(e)
                raise ValueError("Overlapping stiffness ranges not supported yet.")

if par.add_stopper_bead:
    mol_stopper = MoleculeId.get_next()
    extra_mols_smc.append(mol_stopper)
    stopper_type = AtomType(0.01 * DNA_bead_mass)
    stopper_size = 25.0

    stopper_ids = dna_config.get_stopper_ids()
    for st_dna_id in stopper_ids:
        dna_config.add_bead_to_dna(
            stopper_type, mol_stopper, st_dna_id[0], st_dna_id[1], None, None, stopper_size
        )


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
bridge_off = Generator.DynamicCoeffs.create_from_pairwise(
    pair_inter, dna_type, smc_1.t_atp, [0, 0, 0]
)
bridge_on = Generator.DynamicCoeffs.create_from_pairwise(
    pair_inter,
    dna_type,
    smc_1.t_atp,
    [epsilon_SMC_DNA * kBT, par.sigma, par.sigma * 2 ** (1 / 6)],
)

bridge_soft_off = Generator.DynamicCoeffs.create_from_pairwise(
    pair_soft_inter, dna_type, smc_1.t_atp, [0, 0]
)
bridge_soft_on = Generator.DynamicCoeffs.create_from_pairwise(
    pair_soft_inter, dna_type, smc_1.t_atp, [epsilon_SMC_DNA * kBT, par.sigma * 2 ** (1 / 6)]
)


hinge_attraction_off = Generator.DynamicCoeffs.create_from_pairwise(
    pair_inter, dna_type, smc_1.t_upper_site, [0, 0, 0]
)

hinge_attraction_on = Generator.DynamicCoeffs.create_from_pairwise(
    pair_inter, dna_type, smc_1.t_upper_site, [par.epsilon4 * kBT, par.sigma, par.cutoff4]
)

lower_site_off = Generator.DynamicCoeffs.create_from_pairwise(
    pair_inter, dna_type, smc_1.t_lower_site, [0, 0, 0]
)
lower_site_on = Generator.DynamicCoeffs.create_from_pairwise(
    pair_inter,
    dna_type,
    smc_1.t_lower_site,
    None,
)

middle_site_off = Generator.DynamicCoeffs.create_from_pairwise(
    pair_inter, dna_type, smc_1.t_middle_site, [0, 0, 0]
)
middle_site_on = Generator.DynamicCoeffs.create_from_pairwise(
    pair_inter,
    dna_type,
    smc_1.t_middle_site,
    [par.epsilon5 * kBT, par.sigma, par.cutoff5],
)

middle_site_soft_off = Generator.DynamicCoeffs.create_from_pairwise(
    pair_soft_inter, dna_type, smc_1.t_middle_site, [0, 0]
)
middle_site_soft_on = Generator.DynamicCoeffs.create_from_pairwise(
    pair_soft_inter,
    dna_type,
    smc_1.t_middle_site,
    [par.epsilon5 * kBT, par.sigma * 2 ** (1 / 6)],
)

side_site_off = None
side_site_on = None
if par.add_side_site:
    side_site_off = Generator.DynamicCoeffs.create_from_pairwise(
        pair_inter, dna_type, smc_1.t_side_site, [0, 0, 0]
    )

    side_site_on = Generator.DynamicCoeffs.create_from_pairwise(
        pair_inter, dna_type, smc_1.t_side_site, None
    )


gen.bais += [*smc_1.get_bonds(smc_creator.hinge_opening), *dna_config.get_bonds()]

gen.bais += [*smc_1.get_angles(), *dna_config.get_angles()]

gen.bais += smc_1.get_impropers()

# get overrides for DNA
# check for duplicates
original = [(x[0], x[1]) for x in dna_config.molecule_overrides]
seen = set()
dupes = {x for x in original if x in seen or seen.add(x)}
if dupes:
    warn(
        "Conflicting molecule overrides where found!\n"
        "This will likely cause LAMMPS to crash.\n"
        f"\tduplicates: {dupes}\n"
        f"\tall overrides: {dna_config.molecule_overrides}"
    )
for s_id, g_id, mol_id in dna_config.molecule_overrides:
    gen.molecule_override[dna_config.dna_strands[s_id].get_id_from_list_index(g_id)] = mol_id

# Override molecule ids to form rigid safety-belt bond
if isinstance(dna_config, (dna.ObstacleSafety, dna.AdvancedObstacleSafety, dna.Safety)):  # TODO
    safety_index = dna_config.dna_safety_belt_index
    gen.molecule_override[dna_config.dna_strands[0].get_id_from_list_index(safety_index)] = (
        smc_1.mol_lower_site
    )
    # add neighbors to prevent rotation
    # gen.molecule_override[dna_config.dna_strands[0].get_id_from_list_index(safety_index - 1)] = (
    #     smc_1.mol_lower_site
    # )
    # gen.molecule_override[dna_config.dna_strands[0].get_id_from_list_index(safety_index + 1)] = (
    #     smc_1.mol_lower_site
    # )

# get center of dna
all_dna_atoms = []
for strand in dna_config.dna_strands:
    all_dna_atoms += [grp.positions for grp in strand.atom_groups]

positions = np.concatenate(all_dna_atoms, axis=0)
center = np.average(positions, axis=0)

gen.move_all_atoms(-center)

# compare to origin (which is now the center) to find furthest distance
max_distance = np.max(np.abs(positions))
gen.set_system_size(2 * max_distance)

lammps_path = path / "lammps"
lammps_path.mkdir(exist_ok=True)

with open(lammps_path / "datafile_coeffs", "w", encoding="utf-8") as datafile:
    gen.write_coeffs(datafile)

with open(lammps_path / "datafile_positions", "w", encoding="utf-8") as datafile:
    gen.write_positions_and_bonds(datafile)

with open(lammps_path / "styles", "w", encoding="utf-8") as stylesfile:
    stylesfile.write(gen.get_atom_style_command())
    stylesfile.write(gen.get_BAI_styles_command())
    pair_style = "pair_style hybrid/overlay lj/cut $(3.5) soft $(3.5)"
    if gen.use_charges:
        pair_style += " coul/debye $(1.0/5.0) $(7.5)"
    stylesfile.write(pair_style)

#################################################################################
#                                Phases of SMC                                  #
#################################################################################

# make sure the directory exists
states_path = lammps_path / "states"
states_path.mkdir(exist_ok=True)

# if par.site_cycle_period is not zero, the lower site is handled elsewhere
site_cond = par.site_cycle_period == 0 or par.add_side_site
use_lower_site_off = [lower_site_off] if site_cond else []
use_lower_site_on = [lower_site_on] if site_cond else []

if par.site_cycle_period > 0:
    create_phase_wrapper(
        states_path / "cycle_site_on",
        [
            *([lower_site_on] if not site_cond else []),
            side_site_on,
        ],
    )
    create_phase_wrapper(
        states_path / "cycle_site_off",
        [
            *([lower_site_off] if not site_cond else []),
            side_site_off,
        ],
    )

create_phase_wrapper(
    states_path / "adp_bound",
    [
        bridge_off,
        hinge_attraction_on,
        middle_site_off,
        *use_lower_site_off,
        smc_1.elbows_off,
        smc_1.arms_open,
        smc_1.kleisin_unfolds1,
        smc_1.kleisin_unfolds2,
    ],
)

create_phase_wrapper(
    states_path / "apo",
    [
        bridge_off,
        hinge_attraction_off,
        middle_site_off,
        *use_lower_site_on,
        smc_1.elbows_on,
        smc_1.arms_close,
        smc_1.kleisin_unfolds1,
        smc_1.kleisin_unfolds2,
    ],
)

create_phase_wrapper(
    states_path / "atp_bound_1",
    [
        bridge_soft_on,
        middle_site_soft_on,
    ],
)

create_phase_wrapper(
    states_path / "atp_bound_2",
    [
        bridge_soft_off,
        middle_site_soft_off,
        bridge_on,
        hinge_attraction_on,
        middle_site_on,
        *use_lower_site_on,
        smc_1.elbows_off,
        smc_1.arms_open,
        smc_1.kleisin_folds1,
        smc_1.kleisin_folds2,
    ],
)


# get run times for each SMC state
# APO -> ATP1 -> ATP2 -> ADP -> ...
rng = default_rng(par.seed)
if par.non_random_steps:
    warn("Parameter `non_random_steps` is enabled, this should only be used for testing!")
    runtimes = get_times_with_max_steps(par, None)
else:
    runtimes = get_times_with_max_steps(par, rng)

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

    def do_map(lst):
        return map(lambda tup: (gen.get_atom_index(tup[0]), gen.get_atom_index(tup[1])), lst)

    dna_indices_list = {key: do_map(lst) for key, lst in ppp.dna_indices_list.items()}
    dna_indices_list = [dna_config.strand_concat(list(lst)) for lst in dna_indices_list.values()]
    dna_indices_list = [t for x in dna_indices_list for t in x]
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
    file.write(f"DNA_types = {list(set(grp.type.index for grp in dna_config.all_dna_groups))}\n")
    file.write(f"SMC_types = {list(set(grp.type.index for grp in smc_1.get_groups()))}\n")
    file.write("\n")
    file.write(f"spaced_bead_indices = {atomIds_to_LAMMPS_ids(gen, spaced_beads)}\n")
    file.write("\n")
    file.write(f"runtimes = {runtimes}\n")


#################################################################################
#                           Print to parameterfile                              #
#################################################################################


def get_variables_for_lammps() -> list[str]:
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
        "site_cycle_period",
        "site_toggle_delay",
        "site_cycle_when",
    ]


with open(lammps_path / "parameterfile", "w", encoding="utf-8") as parameterfile:
    parameterfile.write("# LAMMPS parameter file\n\n")

    params = get_variables_for_lammps()
    for key in params:
        parameterfile.write(get_def_dynamically(key, getattr(par, key)))

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
    end_points_LAMMPS = atomIds_to_LAMMPS_ids(gen, ppp.end_points)
    parameterfile.write(
        get_string_def(
            "dna_end_points",
            prepend_or_empty(list_to_space_str(end_points_LAMMPS), "id "),
        )
    )

    # turn into LAMMPS indices
    freeze_indices_LAMMPS = atomIds_to_LAMMPS_ids(gen, ppp.freeze_indices)
    parameterfile.write(
        get_string_def("indices", prepend_or_empty(list_to_space_str(freeze_indices_LAMMPS), "id "))
    )

    if (
        isinstance(dna_config, (dna.Obstacle, dna.ObstacleSafety, dna.AdvancedObstacleSafety))
        and dna_config.tether is not None
        and isinstance(dna_config.tether.obstacle, dna.Tether.Wall)
    ):
        parameterfile.write(
            f"variable wall_y equal {dna_config.tether.polymer.atom_groups[0].positions[0][1]}\n"
        )

        excluded = []
        for group in dna_config.tether.polymer.atom_groups:
            excluded += [
                gen.get_atom_index((group, 0)),
                gen.get_atom_index((group, 1)),
            ]
        parameterfile.write(
            get_string_def("excluded", prepend_or_empty(list_to_space_str(excluded), "id "))
        )

    # forces
    stretching_forces_array_LAMMPS = {
        key: atomIds_to_LAMMPS_ids(gen, val) for key, val in ppp.stretching_forces_array.items()
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
    if dna_config.tether is not None and isinstance(dna_config.tether.obstacle, dna.Tether.Gold):
        obstacle_lammps_id = gen.get_atom_index((dna_config.tether.obstacle.group, 0))
        parameterfile.write(f"variable obstacle_id equal {obstacle_lammps_id}\n")

    parameterfile.write("\n")

    parameterfile.write(get_index_def("runtimes", runtimes))
