from sys import argv
from typing import List
import numpy as np


def process(files: List[str]):
    indices_array = []
    steps = None
    for file in files:
        data = np.load(file)
        indices_array.append(data["ids"])
        if steps is None:
            steps = data["steps"]
        else:
            assert(np.array_equal(steps, data["steps"]))
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
    if len(argv) < 1:
        raise Exception("Please provide npz files")
    process(argv)
