# Copyright (c) 2024-2026 Lucas Dooms

# post-processing to find the movement of the SMC relative to the DNA

from __future__ import annotations

import pickle
from copy import deepcopy
from pathlib import Path
from sys import argv
from typing import Any, Sequence

import numpy as np

from smc_lammps.console import warn
from smc_lammps.generate.generator import COORD_TYPE, Nx3Array
from smc_lammps.generate.util import load_parameters
from smc_lammps.post_process.types import ID_TAG_PAIR
from smc_lammps.post_process.util import (
    get_indices_array,
    get_moving_average,
    get_post_processing_parameters,
    get_scaling,
    scale_indices,
    scale_times,
)
from smc_lammps.reader.lammps_data import ID_TYPE, LammpsData, Plane, get_normal_direction
from smc_lammps.reader.parser import Parser


def write(file, steps: Sequence[int], positions: Sequence[Nx3Array]):
    for step, position in zip(steps, positions):
        file.write("ITEM: TIMESTEP\n")
        file.write(f"{step}\n")
        file.write("ITEM: NUMBER OF ATOMS\n")
        file.write("1\n")
        file.write(
            "ITEM: BOX BOUNDS ff ff ff\n"
            "-8.5170000000000005e+02 8.5170000000000005e+02\n"
            "-8.5170000000000005e+02 8.5170000000000005e+02\n"
            "-8.5170000000000005e+02 8.5170000000000005e+02\n"
        )
        file.write(Parser.ATOM_FORMAT)
        file.write(f"1 1 {position[0]} {position[1]} {position[2]}\n")


def is_sorted(lst: Sequence[Any]) -> bool:
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


def split_into_index_groups(indices: Sequence[int], margin: int = 1) -> list[list[int]]:
    """splits a (sorted) list of integers into groups of adjacent integers.
    adjacent means within [value, value + margin] (both bounds inclusive)."""

    if margin < 0:
        raise ValueError("Margin cannot be negative.")

    if not indices:
        return []

    if not is_sorted(indices):
        raise ValueError("indices must be sorted")

    groups = [[indices[0]]]
    for index in indices[1:]:
        if groups[-1][-1] <= index and index <= groups[-1][-1] + margin:
            groups[-1].append(index)
        else:
            groups.append([index])

    return groups


def get_bead_distances(positions: Nx3Array, pos1: Nx3Array, pos2: Nx3Array, pos3: Nx3Array):
    normal_vector = get_normal_direction(pos1, pos2, pos3)
    plane = Plane(pos1, normal_vector)
    return plane.distance(positions)


def remove_outside_planar_n_gon(
    data: LammpsData, points: Nx3Array, delta_out_of_plane: float, delta_extended_plane: float
):
    """removes all points outside a box created from an n-sided polygon in a plane.
    the points are the vertices of the n-gon and are assumed to be in the same plane.
    delta is the thickness of the box in each direction going out of the plane."""
    if len(points) < 3:
        raise ValueError("there must be at least 3 points in the n-gon")

    # Robust normal via Newell's method
    nx = np.sum(
        (points[:, 1] - np.roll(points[:, 1], -1)) * (points[:, 2] + np.roll(points[:, 2], -1))
    )
    ny = np.sum(
        (points[:, 2] - np.roll(points[:, 2], -1)) * (points[:, 0] + np.roll(points[:, 0], -1))
    )
    nz = np.sum(
        (points[:, 0] - np.roll(points[:, 0], -1)) * (points[:, 1] + np.roll(points[:, 1], -1))
    )
    normal = np.array([nx, ny, nz], dtype=COORD_TYPE)
    normal /= np.linalg.norm(normal)

    # delete points further than delta perpendicular to the n-gon
    plane = Plane(points[0] + delta_out_of_plane * normal, normal)
    data.delete_side_of_plane(plane, Plane.Side.OUTSIDE)
    plane = Plane(points[0] - delta_out_of_plane * normal, normal)
    data.delete_side_of_plane(plane, Plane.Side.INSIDE)

    # delete points far away in the plane of the n-gon
    for point1, point2 in zip(points, np.roll(points, shift=-1, axis=0)):
        side_vector = point2 - point1
        normal_to_side = np.cross(normal, side_vector)  # always points INSIDE
        normal_to_side = normal_to_side / np.linalg.norm(normal_to_side)
        side_plane = Plane(point1 - delta_extended_plane * normal_to_side, normal_to_side)
        # INSIDE of plane points out of the shape
        data.delete_side_of_plane(side_plane, Plane.Side.INSIDE)


def handle_dna_bead(
    full_data: LammpsData, filtered_dna: LammpsData, parameters, step
) -> tuple[ID_TYPE, Nx3Array, list[ID_TAG_PAIR]]:
    """Finds the DNA beads that are within the SMC ring and updates the indices, positions lists."""
    fallback = (
        ID_TYPE(-1),
        full_data.positions[-1],
        [(-1, "invalid")],
    )
    if len(filtered_dna.positions) == 0:
        return fallback

    pos_top_left = full_data.get_position_from_index(parameters["top_left_bead_id"])
    pos_top_right = full_data.get_position_from_index(parameters["top_right_bead_id"])
    pos_left = full_data.get_position_from_index(parameters["left_bead_id"])
    pos_right = full_data.get_position_from_index(parameters["right_bead_id"])
    pos_middle_left = full_data.get_position_from_index(parameters["middle_left_bead_id"])
    pos_middle_right = full_data.get_position_from_index(parameters["middle_right_bead_id"])
    pos_kleisins = np.array(
        [full_data.get_position_from_index(i) for i in parameters["kleisin_ids"]], dtype=COORD_TYPE
    )

    delta = 0.6 * parameters["dna_spacing"]

    def add_slice(data: LammpsData, points: Nx3Array):
        dup = deepcopy(filtered_dna)
        remove_outside_planar_n_gon(
            dup,
            points,
            delta,
            4.0 * delta,
        )
        data.combine_by_ids(dup)

    dna_in_smc = LammpsData.empty()

    # top left
    add_slice(
        dna_in_smc,
        np.array([pos_top_left, pos_left, pos_right], dtype=COORD_TYPE),
    )

    # top right
    add_slice(
        dna_in_smc,
        np.array([pos_top_left, pos_top_right, pos_right], dtype=COORD_TYPE),
    )

    # bottom left
    add_slice(
        dna_in_smc,
        np.array([pos_middle_right, pos_middle_left, pos_left], dtype=COORD_TYPE),
    )

    # bottom right
    add_slice(
        dna_in_smc,
        np.array([pos_middle_right, pos_left, pos_right], dtype=COORD_TYPE),
    )

    add_slice(
        dna_in_smc,
        np.array([pos_middle_right, pos_middle_left, pos_right], dtype=COORD_TYPE),
    )

    add_slice(
        dna_in_smc,
        np.array([pos_middle_left, pos_left, pos_right], dtype=COORD_TYPE),
    )

    # kleisin
    add_slice(dna_in_smc, pos_kleisins)

    if len(dna_in_smc.positions) == 0:
        print(f"No DNA found! Timestep: {step}")
        return fallback

    # find groups
    grps = split_into_index_groups(list(dna_in_smc.ids), margin=2)

    # get distances to four reference points:
    #  1. top site
    #  2. center of arms
    #  3. middle site
    #  4. bottom site
    references: dict[str, Nx3Array] = dict()
    references["pos_top"] = (pos_top_left + pos_top_right) / 2.0
    references["pos_center_arms"] = (pos_left + pos_right) / 2.0
    references["pos_middle"] = (pos_middle_left + pos_middle_right) / 2.0
    references["pos_lower"] = pos_kleisins[len(pos_kleisins) // 2]

    id_tag_pairs: list[ID_TAG_PAIR] = []
    for grp in grps:
        distances: dict[str, COORD_TYPE] = dict()
        min_ref = ""
        min_distance = float("inf")
        for ref in references:
            distances[ref] = np.linalg.norm(
                references[ref] - dna_in_smc.positions[np.where(dna_in_smc.ids == grp[0])[0][0]]
            )
            if distances[ref] < min_distance:
                min_ref = ref
                min_distance = distances[ref]

        id_tag_pairs.append((grp[0], min_ref))

    # backward compatibility: get lowest index
    id = grps[0][0]

    return (
        ID_TYPE(id),
        dna_in_smc.positions[np.where(dna_in_smc.ids == id)[0][0]],
        id_tag_pairs,
    )


def get_best_match_dna_bead_in_smc(
    base_path: Path, traj: Path = Path("output/output.lammpstrj"), continue_on_error: bool = False
) -> tuple[list[int], list[list[ID_TYPE]], list[list[Nx3Array]], list[list[list[ID_TAG_PAIR]]]]:
    """Find the DNA bead that is inside of the SMC complex.

    :param base_path: simulation directory (containing parameters.py and post_processing_parameters.py)
    :param traj: path to lammps trajectory file to analyze (either relative to base_path or absolute)
    :param continue_on_error: if an error occurs during parsing, try to analyze the data obtained up to that point
    """

    # -- general idea --
    # For each timestep:
    #     create box around SMC
    #     remove DNA not within box
    #     remove different DNA segments
    #     find DNA closest to SMC plane

    ppp = get_post_processing_parameters(base_path)

    par = Parser(base_path / traj, time_it=True)
    dna_indices_list = ppp["dna_indices_list"]
    steps: list[int] = []
    id_tag_pairs_array: list[list[list[ID_TAG_PAIR]]] = [[] for _ in range(len(dna_indices_list))]
    indices_array: list[list[ID_TYPE]] = [[] for _ in range(len(dna_indices_list))]
    positions_array: list[list[Nx3Array]] = [[] for _ in range(len(dna_indices_list))]
    while True:
        try:
            step, lmpData = par.next_step()
        except Parser.EndOfLammpsFile:
            print(par.timings)
            break
        except Exception as e:
            if continue_on_error:
                print(e)
                warn(f"Error after step {steps[-1]}.\nContinuing with available data...")
                break
            else:
                raise e

        steps.append(step)

        # TODO: get range from post_processing_parameters.py
        box = lmpData.create_box(ppp["SMC_types"])

        new_data = lmpData.delete_outside_box(box)
        new_data.filter_by_types(ppp["DNA_types"])
        # split, and call for each
        for i, (min_index, max_index) in enumerate(dna_indices_list):
            new_data_temp = deepcopy(new_data)
            new_data_temp.filter(lambda id, *_: np.logical_and(min_index <= id, id <= max_index))
            id, pos, id_tag_pairs = handle_dna_bead(lmpData, new_data_temp, ppp, step)
            indices_array[i].append(id)
            positions_array[i].append(pos)
            id_tag_pairs_array[i].append(id_tag_pairs)

    return steps, indices_array, positions_array, id_tag_pairs_array


def create_files(
    path: Path,
    steps: list[int],
    indices_array: list[list[ID_TYPE]],
    positions_array: list[list[Nx3Array]],
    id_tag_pairs_array: list[list[list[ID_TAG_PAIR]]],
) -> None:
    # delete old files
    for p in path.glob("marked_bead*.lammpstrj"):
        print(f"deleting old '{p}'")
        p.unlink()

    for i, positions in enumerate(positions_array):
        with open(path / f"marked_bead{i}.lammpstrj", "w", encoding="utf-8") as file:
            print(f"creating new '{file.name}'")
            write(file, steps, positions)

    for i, indices in enumerate(indices_array):
        np.savez(path / f"bead_indices{i}.npz", steps=steps, ids=indices)

    for i, tags in enumerate(id_tag_pairs_array):
        with open(path / f"idtags{i}.pickle", "wb") as handle:
            pickle.dump(tags, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_plot(
    path: Path,
    steps: list[int],
    indices_array: Sequence[list[ID_TYPE]],
    save_to: Path = Path("bead_id_in_time.svg"),
    use_real_units: bool = True,
    plot_cycle: bool = False,
    moving_average: int = 0,
) -> None:
    import matplotlib.pyplot as plt

    parameters = load_parameters(path)
    tscale, iscale, tunits, iunits = get_scaling(use_real_units, parameters)

    fig, ax = plt.subplots(1, 1, dpi=160)
    fig.set_size_inches((10, 8))
    ax.margins(0.003)

    ax.set_title("Position of SMC complex along DNA in time")

    ax.set_xlabel(f"time ({tunits})")
    ax.set_ylabel(f"position ({iunits})")

    for i, indices in enumerate(indices_array):
        times = scale_times(steps, tscale=tscale)
        positions = scale_indices(indices, iscale=iscale)

        if moving_average == 0:
            ax.scatter(
                times,
                positions,
                s=0.5,
                label=f"DNA {i}",
            )
        else:
            times = times[: len(times) - moving_average + 1]
            positions = get_moving_average(positions, n=moving_average)
            ax.plot(times, positions, label=f"DNA {i}")

    if plot_cycle:
        from smc_lammps.post_process.matplotlib.draw import (
            fill_between_runtime_lines,
            get_runtime_lines,
        )

        for lc in get_runtime_lines(ax, path, indices_array, tscale=tscale, iscale=iscale):
            ax.add_collection(lc)

        fill_between_runtime_lines(
            ax,
            path,
            tscale=tscale,
        )

    ax.legend()
    fig.tight_layout()

    fig.savefig(path / save_to)


def get_msd_obstacle(folder_path):
    try:
        par = Parser(folder_path / "obstacle.lammpstrj")
    except FileNotFoundError:
        print("Skipping obstacle MSD analysis: obstacle trajectory file not found")
        return

    steps = []
    positions = []
    while True:
        try:
            step, lmpData = par.next_step()
        except Parser.EndOfLammpsFile:
            break

        steps.append(step)
        assert len(lmpData.positions) == 1
        positions.append(lmpData.positions[0])

    # calculate msd in time chunks
    time_chunk_size = 2000  # number of timesteps to pick for one window

    def calculate_msd(array) -> float:
        return np.average((array - array[0]) ** 2)

    def apply_moving_window(window_size: int, func, array):
        """returns the array of the results from applying the func to
        windows along the array."""
        result = []
        for i in range(len(array) - window_size + 1):
            result.append(func(array[i : i + window_size]))
        return result

    msd_array = apply_moving_window(time_chunk_size, calculate_msd, positions)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6), dpi=144)
    plt.title(
        f"MSD over time in chunks of {(steps[1] - steps[0]) * time_chunk_size} (all average = {calculate_msd(positions)})"
    )
    plt.xlabel("time")
    plt.ylabel("MSD")
    plt.scatter(steps[: -time_chunk_size + 1], msd_array, s=0.3)
    plt.savefig(folder_path / "msd_in_time.png")


def main(argv: list[str]):
    argv = argv[1:]

    if len(argv) < 1:
        raise Exception("Please provide a folder path")
    path = Path(argv[0])

    args = []
    if len(argv) > 1:
        args.append(Path(argv[1]))

    steps, indices_array, positions_array, id_tag_pairs_array = get_best_match_dna_bead_in_smc(
        path, *args
    )
    create_files(path, steps, indices_array, positions_array, id_tag_pairs_array)
    create_plot(
        path,
        steps,
        [list(x) for x in get_indices_array(id_tag_pairs_array)],
        plot_cycle=True,
        moving_average=5,
    )
    get_msd_obstacle(path)


if __name__ == "__main__":
    main(argv)
