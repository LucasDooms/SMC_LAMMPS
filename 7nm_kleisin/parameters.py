# Radius of lower circular-arc compartment (nm)
HKradius = 7.
# HKradius = 4.5 # from original paper (used incorrect formula??)

# amount of DNA
N = 750

cycles = 10

# bottom site
epsilon6 = 500.0

# weak force
force = 0.08

# Average number of steps in each state
stepsrelaxed = 100000
stepsATP_1 = 400000
stepsATP_2 = 200000
stepsreleased_1 = 800000
stepsreleased_2 = 3000

# middle
epsilon5 = 50.0
# lower + HeatA
epsilon6 = 500.0

# configuration
dnaConfig = "safety_loop"
