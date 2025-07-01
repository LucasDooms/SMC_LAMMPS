# Copyright (c) 2024-2025 Lucas Dooms

from glob import glob
from pathlib import Path
from sys import argv
from typing import List

import numpy as np
from scipy.optimize import curve_fit


def get_npz_files_from_args(args: List[str]):
    files = []

    # replace glob patterns
    matches = []
    for arg in args:
        matches += glob(arg)

    for match in matches:
        match = Path(match)
        if match.is_dir():
            for npzfile in match.glob("*.npz"):
                files.append(str(npzfile))
        else:
            if match.suffix != ".npz":
                print(f"WARNING: entered non npz file: {match}")
            files.append(str(match))

    return files


def get_data_raw(files: List[str]):
    if not files:
        raise ValueError("did not receive files to process")

    steps_array = []
    indices_array = []
    for file in files:
        data = np.load(file)
        steps_array.append(data["steps"])
        indices_array.append(data["ids"])

    return steps_array, indices_array


def get_averages(steps_array, indices_array, cutoff_to_shortest: bool = True):
    if cutoff_to_shortest:
        shortest_steps_length = min([len(x) for x in steps_array])
        shortest_non_min1 = min([max(np.where(ind != -1)[0]) for ind in indices_array])
        shortest_steps_length = min(shortest_steps_length, shortest_non_min1)
        steps_array = [steps[:shortest_steps_length] for steps in steps_array]
        steps = steps_array[0]

        assert all([np.array_equal(steps, others) for others in steps_array])
        steps_array = steps

        for i in range(len(indices_array)):
            indices_array[i] = indices_array[i][:shortest_steps_length]

    indices_array = np.array(indices_array).transpose()

    def custom_average(arr):
        return np.average(arr) if arr.size else -1

    averages = np.array(
        [
            custom_average(
                indices[indices != -1]
            )  # ignore -1, these are due to issues in process_displacement.py
            for indices in indices_array
        ]
    )

    return steps_array, averages


def create_figure_raw(steps, averages, num_samples: int):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6), dpi=144)
    plt.title(f"Average index of DNA bead inside SMC loop in time ({num_samples} samples)")
    plt.xlabel("timestep")
    plt.ylabel("DNA bead index")
    plt.scatter(steps, averages, s=0.5)
    plt.savefig("average_bead_id_in_time.png")


def create_figure_units(steps, averages, num_samples: int, linear_parameters=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6), dpi=144)
    plt.title(f"Average position of SMC along DNA in time ({num_samples} samples)")
    plt.xlabel("time [s]")
    plt.ylabel("distance [nm]")
    plt.scatter(steps, averages, s=0.5, color="purple", label="Simulation")

    if linear_parameters is not None:
        plt.plot(
            steps,
            linear(steps, *linear_parameters),
            color="green",
            label=f"Linear Fit (slope={linear_parameters[0]:.1f} nm / s)",
        )

    plt.legend()
    plt.savefig("average_bead_id_in_time.png")


def convert_units_time(steps):
    # conversion: 1 cycle = 12 * 10^6 ~ 0.13 seconds
    # TODO: look this up dynamically in parameterfile!
    steps_per_cycle = int(12 * 1e6)
    seconds_per_cycle = 0.13
    seconds_per_step = seconds_per_cycle / steps_per_cycle
    return steps * seconds_per_step


def convert_units_distance(indices):
    # conversion: 1 index = 1 bead of DNA = 5 bps = 1.7 nm
    # TODO: look this up dynamically in parameterfile!
    nanometers_per_index = 1.7
    return indices * nanometers_per_index


def linear(x, a, b):
    return a * x + b


def linear_fit(steps, averages):
    popt = curve_fit(linear, steps, averages, (-340, 500))[0]
    print(f"optimal values (a, b): {popt}")
    return popt


def process(globs: List[str]):
    files = get_npz_files_from_args(globs)
    steps_array, indices_array = get_data_raw(files)
    steps_array, averages = get_averages(steps_array, indices_array, True)
    steps, averages = convert_units_time(steps_array), convert_units_distance(averages)
    popt = linear_fit(steps, averages)
    create_figure_units(steps, averages, len(files), popt)


if __name__ == "__main__":
    argv = argv[1:]
    if not argv:
        raise Exception("Please provide glob patterns of npz files or folders containing them")
    process(argv)
