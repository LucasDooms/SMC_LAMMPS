#### DO NOT EDIT THIS BLOCK ####
from generate.default_parameters import Parameters

p = Parameters()
#### END OF BLOCK ####

## Define your parameters using p.key = value
## See generate/default_parameters.py for all parameters

# Radius of lower circular-arc compartment (nm)
p.kleisin_radius = 4.5

# amount of DNA
p.N = 350

# cycles = 20
p.cycles = None
p.max_steps = 240000000  # average number of steps for 20 cycles

p.rigid_hinge = True

# weak force
p.force = 0.05

# configuration
# p.dnaConfig = "advanced_obstacle_safety"
# p.dnaConfig = "obstacle_safety"
p.dna_config = "obstacle"
# dnaConfig = "line"

# smc_force = 0.01
