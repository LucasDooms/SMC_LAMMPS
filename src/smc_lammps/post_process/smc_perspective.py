# Copyright (c) 2025 Lucas Dooms

from pathlib import Path
from runpy import run_path
from sys import argv
from typing import Any, Sequence

import numpy as np


def read_lammpstrj(trajectory_file: Path):
    timesteps: list[int] = []
    num_atoms: list[int] = []
    box_bounds: list[list[list[float]]] = []
    atom_data: list[list[list[Any]]] = []

    with open(trajectory_file, "r") as file:
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
            atoms: list[list[Any]] = []

            for j in range(num_atoms[-1]):
                components = lines[i + 1 + j].strip().split()
                # convert x,y,z to float
                components: list[Any] = [*components[:2], *map(float, components[2:])]

                atoms.append(components)

            atom_data.append(atoms)

            i += 1 + num_atoms[-1]

        else:
            i += 1

    return timesteps, num_atoms, box_bounds, atom_data


def write_lammpstrj(
    file_path: Path,
    timesteps: Sequence[int],
    num_atoms: Sequence[int],
    box_bounds: Sequence[Sequence[list[float]]],
    atom_data: Sequence[Sequence[Sequence[Any]]],
):
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


def main(trajectory_file: Path, output_file: Path, index1: int, index2: int, index3: int):
    timesteps, num_atoms, box_bounds, atom_data = read_lammpstrj(trajectory_file)

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
    if len(argv) < 3:
        raise ValueError(
            "3 inputs required: output.lammpstrj, file_name, and post_processing_parameters.py"
        )

    trajectory_file = Path(argv[0])
    write_to_file = Path(argv[1])
    post_processing_parameters_file = Path(argv[2])
    parameters = run_path(post_processing_parameters_file.as_posix())

    if len(argv) > 3:
        use_reference = argv[3]
    else:
        # default is arms
        use_reference = "arms"

    if len(argv) > 4:
        force = argv[4].lower() in {"1", "true", "yes"}
    else:
        force = False

    if not force and write_to_file.exists():
        raise FileExistsError(f"cannot write to '{write_to_file}', file exists")

    if use_reference == "kleisin":
        kleisin_ids: list[int] = parameters["kleisin_ids"]
        ref_ids = [kleisin_ids[1], kleisin_ids[len(kleisin_ids) // 2], kleisin_ids[-2]]
    elif use_reference == "arms":
        ref_ids: list[int] = [
            parameters["top_left_bead_id"],
            parameters["left_bead_id"],
            parameters["right_bead_id"],
        ]
    else:
        raise ValueError(f"Unknown reference option {use_reference}")

    main(trajectory_file, write_to_file, *ref_ids)
