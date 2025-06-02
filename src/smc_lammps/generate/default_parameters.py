from dataclasses import dataclass
from typing import Any


@dataclass
class Parameters:
    def __setattr__(self, name: str, value: Any, /) -> None:
        if not hasattr(self, name):
            raise AttributeError("You cannot define new parameters.")
        super().__setattr__(name, value)

    ################ General parameters ################

    loop = 100
    "Initial loop size (DNA beads)"

    diameter = 20
    "Diameter of initial loop (nm)"

    T = 300.0
    "Simulation temperature (K)"

    kB = 0.013806504
    "Boltzmann's constant (pN nm / K)"

    gamma = 0.5
    "Inverse of friction coefficient (ns)"

    output_steps = 10000
    "Printing period (time steps)"

    timestep = 2e-4
    "Simulation timestep (ns)"

    seed = 123
    "Random number seed"

    N: int = 501
    "Number of DNA beads"

    n = 5
    "Number of base pairs per DNA bead"

    force = 0.800
    """Stretching forces (pN) (set to any falsy value for no forces)
    WARNING: currently: if no forces -> ends are frozen"""

    runs = 10
    "Number of independent runs"

    cycles: int | None = 2
    """Number of SMC cycles (if set to None, will find approximate value using max_steps)
    Note: cycles are stochastic, so time per cycle is variable"""

    max_steps: int | None = None
    """Max steps for run (None -> no maximum, will complete every cycle)
    Note: this is not a hard limit, some extra steps may be performed to complete a cycle"""

    steps_ATP = 2000000
    "Average number of steps for ATP binding"

    steps_ADP = 8000000
    "Average number of steps for ATP hydrolysis"

    steps_APO = 2000000
    "Average number of steps for returning to APO"

    ##################### DNA #######################

    dna_config = "folded"
    "configuration to generate"

    add_stopper_bead = False
    "add a bead that prevents the SMC from slipping off of the wrong end of the DNA"

    add_RNA_polymerase = True
    "adds 10 nm bead at DNA-tether site"

    RNA_polymerase_size: float = 5.0
    "radius of RNA polymerase (nm)"

    RNA_polymerase_type = 1

    spaced_beads_interval: int | None = None
    "how many DNA beads to leave between small obstacles"

    spaced_beads_size: float = 5.0
    "radius of beads along DNA (nm)"

    ##################### Geometry #####################

    arm_length = 50.0
    "Length of each coiled-coil arm (nm)"

    bridge_width = 7.5
    "Width of ATP bridge (nm)"

    hinge_radius = 1.5
    "Hinge radius (nm)"

    rigid_hinge = True

    kleisin_radius = 7.0
    "Radius of lower circular-arc compartment (nm)"

    sigma_SMC_DNA = 2.5
    "SMC-DNA hard-core repulsion radius = LJ sigma (nm)"

    sigma = 2.5

    folding_angle_APO = 45.0
    "Folding angle of lower compartment (degrees)"

    folding_angle_ATP = 160.0
    "Folding angle of lower compartment (degrees)"

    arms_angle_ATP = 130.0
    "Opening angle of arms in ATP-bound state (degrees)"

    #################### LJ energies ###################

    # 3 = Repulsion
    # 4 = Upper site
    # 5 = Middle site
    # 6 = Lower site

    # LJ energy (kT units)
    # DE for a cutoff of 3.0 nm: 0.11e
    # DE for a cutoff of 3.5 nm: 0.54e

    epsilon3 = 3.0
    epsilon4 = 6.0
    epsilon5 = 6.0
    epsilon6 = 100.0

    # LJ cutoff (nm)
    cutoff4 = 3.5
    cutoff5 = 3.5
    cutoff6 = 3.0

    ################# Bending energies #################

    arms_stiffness = 100.0
    "Bending stiffness of arm-bridge angle (kT units)"

    elbows_stiffness = 30.0
    "Bending stiffness of elbows (kT units)"

    site_stiffness = 100.0
    "Alignment stiffness of binding sites (kT units)"

    folding_stiffness = 60.0
    "Folding stiffness of lower compartment (kT units)"

    asymmetry_stiffness = 100.0
    "Folding asymmetry stiffness of lower compartment (kT units)"

    ################# Other #################

    smc_force = 0.0
    "Extra force on SMC in the -x direction and +y direction (left & up)"

    use_charges = False
    "Enable Coulomb interactions in LAMMPS"
