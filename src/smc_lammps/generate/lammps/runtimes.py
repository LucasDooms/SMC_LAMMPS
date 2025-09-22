import math
from sys import maxsize

import numpy as np

from smc_lammps.generate.default_parameters import Parameters


def get_times(apo: int, atp1: int, atp2: int, adp: int, rng_gen: np.random.Generator) -> list[int]:
    """Returns a list of runtimes for each SMC state [APO, ATP1, ATP2, ADP] sampled from an exponential distribution."""

    def mult(x):
        # use 1.0 to get (0, 1] lower exclusive
        return -x * np.log(1.0 - rng_gen.uniform())

    return [math.ceil(mult(x)) for x in (apo, atp1, atp2, adp)]


def get_times_with_max_steps(parameters: Parameters, rng_gen: np.random.Generator) -> list[int]:
    """Returns a list of runtimes for a certain number of SMC cycles that fit within the maximum number of steps."""
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
