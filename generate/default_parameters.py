from dataclasses import dataclass
from typing import Any


@dataclass
class Parameters:
    def __setattr__(self, name: str, value: Any, /) -> None:
        if not hasattr(self, name):
            raise AttributeError("You cannot define new parameters.")
        super().__setattr__(name, value)

    ################ General parameters ################

    # Initial loop size (DNA beads)
    loop = 100

    # Diameter of initial loop (nm)
    diameter = 20

    # Simulation temperature (K)
    T = 300.0

    # Boltzmann's constant (pN nm / K)
    kB = 0.013806504

    # Inverse of friction coefficient (ns)
    gamma = 0.5

    # Printing period (time steps)
    output_steps = 10000
    # output_steps = 200000

    # Simulation timestep (ns)
    timestep = 2e-4

    # Seed
    seed = 123

    # Number of DNA beads
    N: int = 501

    # Number of base pairs per DNA bead
    n = 5

    # Stretching forces (pN) (set to any falsy value for no forces)
    # WARNING: currently: if no forces -> ends are frozen

    # forces = 0.100 0.800 1.500 2.200 2.900 3.600 4.300 5.000
    force = 0.800

    # Number of independent runs
    runs = 10

    # Number of SMC cycles (if set to None, will find approximate value using max_steps)
    # Note: cycles are stochastic, so time per cycle is variable
    cycles: int | None = 2

    # Max steps for run (None -> no maximum, will complete every cycle)
    # Note: this is not a hard limit, some extra steps may be performed to complete a cycle
    max_steps: int | None = None

    # Average number of steps for ATP binding
    stepsATP = 2000000

    # Average number of steps for ATP hydrolysis
    stepsADP = 8000000

    # Average number of steps for returning to APO
    stepsAPO = 2000000

    ##################### DNA #######################

    # configuration to generate
    dnaConfig = "folded"

    # adds 10 nm bead at DNA-tether site
    # only relevant if dnaConfig includes tether!
    addRNAPolymerase = True
    RNAPolymeraseType = 1

    ##################### Geometry #####################

    # Length of each coiled-coil arm (nm)
    armLength = 50.0

    # Width of ATP bridge (nm)
    bridgeWidth = 7.5

    # Hinge radius (nm)
    hingeRadius = 1.5
    rigidHinge = True

    # Radius of lower circular-arc compartment (nm)
    HKradius = 7.0

    # SMC-DNA hard-core repulsion radius = LJ sigma (nm)
    intRadSMCvsDNA = 2.5
    sigma = 2.5

    # Folding angles of lower compartment (degrees)
    foldingAngleAPO = 45.0
    foldingAngleATP = 160.0

    # Opening angle of arms in ATP-bound state (degrees)
    armsAngleATP = 130.0

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

    # Bending stiffness of arm-bridge angle (kT units)
    armsStiffness = 100.0

    # Bending stiffness of elbows (kT units)
    elbowsStiffness = 30.0

    # Alignment stiffness of binding sites (kT units)
    siteStiffness = 100.0

    # Folding stiffness of lower compartment (kT units)
    foldingStiffness = 60.0

    # Folding asymmetry stiffness of lower compartment (kT units)
    asymmetryStiffness = 100.0

    # Extra force on SMC in the -x direction and +y direction (left & up)
    smc_force = 0.0
