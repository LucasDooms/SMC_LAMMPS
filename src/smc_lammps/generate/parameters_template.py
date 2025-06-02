#### DO NOT EDIT THIS BLOCK ####
from smc_lammps.generate.default_parameters import Parameters

p = Parameters()
#### END OF BLOCK ####

## Define your parameters using p.key = value
## See src/smc_lammps/generate/default_parameters.py for all parameters

## Below are some example parameters


# Radius of lower circular-arc compartment (nm)
p.kleisin_radius = 4.5

# Amount of DNA
p.N = 300

# Set number of cycles directly
# p.cycles = 20

# Or set max_steps
p.cycles = None
total_steps_per_cycle = p.steps_APO + p.steps_ADP + p.steps_ATP
p.max_steps = 20 * total_steps_per_cycle

# Hinge acts as one rigid part
p.rigid_hinge = True

# Small obstacle on DNA at tether location
p.add_RNA_polymerase = False
# p.RNA_polymerase_type = 1
# p.RNA_polymerase_size = 5.0

# Apply a weak force on the ends of the DNA
p.force = 0.05

# Choose a configuration
# p.dna_config = "advanced_obstacle_safety"
# p.dna_config = "obstacle_safety"
p.dna_config = "obstacle"
# p.dna_config = "line"

# Prevent SMC from falling off the DNA
p.add_stopper_bead = True

# Enable or disable charges
p.use_charges = False
