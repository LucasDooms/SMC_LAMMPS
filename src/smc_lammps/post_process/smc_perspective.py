# Copyright (c) 2025 Lucas Dooms

from pathlib import Path
from runpy import run_path
from sys import argv

import numpy as np


def read_lammpstrj(file_path):
    timesteps = []
    num_atoms = []
    box_bounds = []
    atom_data = []

    with open(file_path, "r") as file:
        lines = file.readlines()

    i = 0

    while i < len(lines):
        if lines[i].strip() == "ITEM: TIMESTEP":
            timestep = int(lines[i + 1].strip())

            timesteps.append(timestep)

            i += 2

        elif lines[i].strip() == "ITEM: NUMBER OF ATOMS":
            n_atoms = int(lines[i + 1].strip())

            num_atoms.append(n_atoms)

            i += 2

        elif lines[i].strip() == "ITEM: BOX BOUNDS ff ff ff":
            bounds = [list(map(float, lines[i + j].strip().split())) for j in range(1, 4)]

            box_bounds.append(bounds)

            i += 4

        elif lines[i].strip() == "ITEM: ATOMS id type x y z":
            atoms = []

            for j in range(num_atoms[-1]):
                atoms.append(list(map(float, lines[i + 1 + j].strip().split())))

            atom_data.append(atoms)

            i += 1 + num_atoms[-1]

        else:
            i += 1

    return timesteps, num_atoms, box_bounds, atom_data


def write_lammpstrj(file_path, timesteps, num_atoms, box_bounds, atom_data):
    with open(file_path, "w") as file:
        for t, n, bounds, atoms in zip(timesteps, num_atoms, box_bounds, atom_data):
            file.write("ITEM: TIMESTEP\n")

            file.write(f"{t}\n")

            file.write("ITEM: NUMBER OF ATOMS\n")

            file.write(f"{n}\n")

            file.write("ITEM: BOX BOUNDS ff ff ff\n")

            for b in bounds:
                file.write(f"{b[0]} {b[1]}\n")

            file.write("ITEM: ATOMS id type x y z\n")

            for atom in atoms:
                file.write(" ".join(map(str, atom)) + "\n")


def rigid_transform_3D(A, B):
    """implementation of Kabsch algorithm"""
    assert len(A) == len(B)

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Center the points
    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.transpose().dot(BB)
    U, _, Vt = np.linalg.svd(H)
    R = U.dot(Vt)

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U.dot(Vt)

    t = centroid_B - R.transpose().dot(centroid_A)

    return R.transpose(), t.transpose()


def transform_atoms(atom_data, index1, index2, index3, v0, v1, v2):
    for timestep_data in atom_data:
        A = np.array(
            [
                timestep_data[index1 - 1][2:5],
                timestep_data[index2 - 1][2:5],
                timestep_data[index3 - 1][2:5],
            ]
        )

        B = np.array([v0, v1, v2])

        R, t = rigid_transform_3D(A, B)

        for atom in timestep_data:
            atom_pos = np.array(atom[2:5])

            transformed_pos = np.dot(R, atom_pos) + t

            atom[2:5] = transformed_pos.tolist()

    return atom_data


def main(input_file, output_file, index1, index2, index3):
    timesteps, num_atoms, box_bounds, atom_data = read_lammpstrj(input_file)

    # Extract initial positions for the three atoms
    initial_pos1 = atom_data[0][index1 - 1][2:5]
    initial_pos2 = atom_data[0][index2 - 1][2:5]
    initial_pos3 = atom_data[0][index3 - 1][2:5]

    # Transform the atom positions to keep index1, index2, and index3 at their initial positions
    transformed_atom_data = transform_atoms(
        atom_data, index1, index2, index3, initial_pos1, initial_pos2, initial_pos3
    )

    write_lammpstrj(output_file, timesteps, num_atoms, box_bounds, transformed_atom_data)


if __name__ == "__main__":
    argv = argv[1:]
    if len(argv) < 2:
        raise ValueError("2 inputs required: output.lammpstrj and post_processing_parameters.py")

    output_file = Path(argv[0])

    post_processing_parameters_file = Path(argv[1])
    parameters = run_path(post_processing_parameters_file.as_posix())

    argv = argv[2:]
    if argv:
        use_reference = argv[0]
    else:
        # default is arms
        use_reference = "arms"

    if use_reference == "kleisin":
        kleisin_ids = parameters["kleisin_ids"]
        ref_ids = [kleisin_ids[1], kleisin_ids[len(kleisin_ids) // 2], kleisin_ids[-2]]
    elif use_reference == "arms":
        ref_ids = [
            parameters["top_left_bead_id"],
            parameters["left_bead_id"],
            parameters["right_bead_id"],
        ]
    else:
        raise ValueError(f"Unknown reference option {use_reference}")

    main(output_file, output_file.parent / f"perspective.{output_file.name}", *ref_ids)
