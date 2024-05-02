# Copyright (c) 2024 Lucas Dooms

from sys import argv
from glob import glob
from typing import List
from pathlib import Path
import numpy as np


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


def process(files: List[str]):
    files = get_npz_files_from_args(files)
    if not files:
        raise Exception("did not receive files to process")

    steps_array = []
    indices_array = []
    for file in files:
        data = np.load(file)
        steps_array.append(data["steps"])
        indices_array.append(data["ids"])

    shortest_steps_length = min([len(x) for x in steps_array])
    steps_array = [steps[:shortest_steps_length] for steps in steps_array]
    steps = steps_array[0]
    assert(all([np.array_equal(steps, others) for others in steps_array]))

    for i in range(len(indices_array)):
        indices_array[i] = indices_array[i][:shortest_steps_length]

    indices_array = np.array(indices_array).transpose()

    def custom_average(arr):
        return np.average(arr) if arr.size else -1

    averages = np.array([
        custom_average(indices[indices != -1]) # ignore -1, these are due to issues in process_displacement.py
        for indices in indices_array
    ])

    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6), dpi=144)
    plt.title("Averaged index of DNA bead inside SMC loop in time")
    plt.xlabel("time")
    plt.ylabel(f"Average DNA bead index ({len(files)} samples)")
    plt.scatter(steps, averages, s=0.5)
    plt.savefig("average_bead_id_in_time.png")

if __name__ == "__main__":
    argv = argv[1:]
    if not argv:
        raise Exception("Please provide glob patterns of npz files or folders containing them")
    process(argv)
