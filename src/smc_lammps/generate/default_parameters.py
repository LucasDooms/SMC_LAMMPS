# Copyright (c) 2024-2026 Lucas Dooms

from dataclasses import dataclass
from typing import Any


@dataclass
class Parameters:
    """
    Class that stores all simulation parameters defined by the user.
    """

    ################ General parameters ################

    T: float = 300.0
    "Simulation temperature (K)."

    kB: float = 0.01380649
    "Boltzmann's constant (pN nm / K)."

    gamma: float = 0.5
    "Inverse of friction coefficient (ns)."

    output_steps: int = 10000
    "Printing period (number of time steps)."

    timestep: float = 2e-4
    "Simulation timestep (ns)."

    seed: int = 123
    "Random number seed. Used in the LAMMPS ``fix langevin`` command and initial configuration generation."

    N: int = 501
    "Number of DNA beads. See also :py:attr:`n` for bead size."

    n: int = 5
    "Number of base pairs per DNA bead."

    force: float | None = 0.800
    """Stretching forces (pN) (set to ``None`` for fixed end points)."""

    ##################### SMC cycle #######################

    cycles: int | None = 2
    """Number of SMC cycles (if set to None, will find approximate value using :py:attr:`max_steps`).
    Note: cycles are stochastic, so time per cycle is variable, see also :py:attr:`non_random_steps`."""

    max_steps: int | None = None
    """Max time steps for a run (None -> no maximum, will complete every cycle).
    Note: this is not a hard limit, some extra steps may be performed to complete a cycle."""

    steps_APO: int = 2000000
    "Average number of steps spent in APO state (waiting for ATP binding)."

    steps_ATP: int = 8000000
    "Average number of steps spent in ATP state (waiting for ATP hydrolysis)."

    steps_ADP: int = 2000000
    "Average number of steps spent in ADP state (waiting for return to APO)."

    non_random_steps: bool = False
    "Disables the exponential sampling for :py:attr:`steps_APO`, :py:attr:`steps_ATP`, and :py:attr:`steps_ADP`."

    ##################### DNA #######################

    dna_config: str = "folded"
    "Initial DNA configuration to generate."

    add_stopper_bead: bool = False
    "Add a bead that prevents the SMC from slipping off of the wrong end of the DNA."

    add_RNA_polymerase: bool = False
    "Add a bead at the DNA-tether site, see also :py:attr:`RNA_polymerase_size`."

    RNA_polymerase_size: float = 5.0
    "Radius of RNA polymerase (nm)."

    RNA_polymerase_type: int = 1
    "TODO"

    spaced_beads_interval: int | None = None
    "Number of DNA beads to leave between small obstacles."

    spaced_beads_size: float = 5.0
    "Radius of beads along DNA (nm)."

    spaced_beads_full_dna: bool = False
    "Whether to place beads across the entire DNA length or stop at the SMC location."

    spaced_beads_smc_clearance: float = spaced_beads_size
    "Length of bare DNA to keep next to SMC (nm) (useful to prevent large forces due to overlap in initial configuration)."

    spaced_beads_custom_stiffness: float = 1.0
    "Multiple of the default DNA stiffness."

    spaced_beads_type: int = 1
    "Type of beads to use, ``0`` -> fene/expand bonds, ``1`` -> rigid molecules."

    ##################### Geometry #####################

    arm_length: float = 50.0
    "Length of each coiled-coil arm (nm)."

    bridge_width: float = 7.5
    "Width of ATP bridge (nm)."

    use_toroidal_hinge: bool = True
    "Whether to use the toroidal hinge or the old hinge type."

    hinge_radius: float = 1.5
    "Hinge radius (nm)."

    rigid_hinge: bool = True
    "Whether to make the hinge a single rigid object or connect the two hinge sections by bonds."

    kleisin_radius: float = 7.0
    "Radius of lower circular-arc compartment (nm)."

    sigma_SMC_DNA: float = 2.5
    "SMC-DNA hard-core repulsion radius = LJ sigma (nm)."

    sigma: float = 2.5
    "TODO"

    folding_angle_APO: float = 45.0
    "Folding angle of lower compartment (degrees)."

    folding_angle_ATP: float = 160.0
    "Folding angle of lower compartment (degrees)."

    arms_angle_ATP: float = 130.0
    "Opening angle of arms in ATP-bound state (degrees)."

    #################### Binding sites ###################

    add_side_site: bool = False
    """Add a binding site on the lower SMC arm to act as the cycling site.
    If enabled, the lower site operates normally."""

    site_cycle_period: int = 0
    """The number of SMC cycles between events where the cycling site is disabled.
    A value of zero disables this and uses the default site dynamics."""

    site_toggle_delay: int = 0
    """The number of SMC cycles between the cycling site being turned off and then on again.
    A value of zero means that the site will be enabled in the same cycle."""

    site_cycle_when: str = "apo"
    """When to re-enable the cycling site. Allowed values: "apo", "adp"."""

    #################### LJ energies ###################

    # LJ energy (kT units)
    # DE for a cutoff of 3.0 nm: 0.11e
    # DE for a cutoff of 3.5 nm: 0.54e

    epsilon3: float = 3.0
    """Repulsion strength (kT units)."""
    epsilon4: float = 6.0
    """Upper site attraction strength (kT units)."""
    epsilon5: float = 6.0
    """Middle site attraction strength (kT units)."""
    epsilon6: float = 100.0
    """Lower site attraction strength (kT units)."""

    cutoff4: float = 3.5
    """Cutoff distance for upper site (nm)."""
    cutoff5: float = 3.5
    """Cutoff distance for middle site (nm)."""
    cutoff6: float = 3.0
    """Cutoff distance for lower site (nm)."""

    ################# Bending energies #################

    arms_stiffness: float = 100.0
    "Bending stiffness of arm-bridge angle (kT units)."

    elbows_stiffness: float = 30.0
    "Bending stiffness of elbows (kT units)."

    site_stiffness: float = 100.0
    "Alignment stiffness of binding sites (kT units)."

    folding_stiffness: float = 60.0
    "Folding stiffness of lower compartment (kT units)."

    asymmetry_stiffness: float = 100.0
    "Folding asymmetry stiffness of lower compartment (kT units)."

    ################# Bonds #################

    elbow_attraction: float = 30.0
    "Attractive energy between elbows in the APO state (kT units)."

    elbow_spacing: float = 2.5
    "Rest length between elbows in the APO state (nm)."

    ################# Other #################

    smc_force: float = 0.0
    "Extra force on SMC in the -x direction and +y direction (left & up) (pN)."

    use_charges: bool = False
    "Enable Coulomb interactions in LAMMPS."

    ################# Methods #################

    def average_steps_per_cycle(self) -> int:
        """Average steps for a full cycle = :py:attr:`steps_APO` + :py:attr:`steps_ATP` + :py:attr:`steps_ADP`.

        Returns:
            Average number of time steps for one SMC cycle.
        """
        return self.steps_APO + self.steps_ATP + self.steps_ADP

    def __setattr__(self, name: str, value: Any, /) -> None:
        if not hasattr(self, name):
            raise AttributeError("You cannot define new parameters.")
        super().__setattr__(name, value)
