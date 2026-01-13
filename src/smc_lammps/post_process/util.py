# Copyright (c) 2025-2026 Lucas Dooms

import csv
import shutil
from itertools import product
from pathlib import Path
from runpy import run_path
from typing import Any, Iterator, Sequence, TypeVar

import numpy as np

from smc_lammps.console import warn
from smc_lammps.generate.default_parameters import Parameters
from smc_lammps.post_process.types import ID_TAG_PAIR
from smc_lammps.reader.lammps_data import ID_TYPE, IdArray
from smc_lammps.reader.parser import Parser


def get_scaling(
    use_real_units: bool = True, parameters: Parameters | None = None
) -> tuple[float, float, str, str]:
    """Returns the scaling factors to go from simulation units to real units.

    Returns the index scale (DNA bead index -> nm), and the time scale (timestep -> s).

    Args:
        use_real_units: If False, no conversion is performed (scaling = 1.0).
        parameters: Simulation parameters.

    Returns:
        Tuple of (time scale, index scale, time units, index units)

    Raises:
        TypeError: `use_real_units` is True, but `parameters` is set to None.
    """
    if use_real_units:
        if parameters is None:
            raise TypeError("If use_real_units is True, parameters must be provided and not None.")

        # convert to seconds and nanometers
        # time scale uses 0.13 seconds per SMC cycle as reference
        tscale = 0.13 / parameters.average_steps_per_cycle()
        # length per basepair is about 0.34 nm
        iscale = 0.34 * parameters.n

        tunits = "s"
        iunits = "nm"
    else:
        tscale = 1.0
        iscale = 1.0

        tunits = "sim steps"
        iunits = "bead index"

    return tscale, iscale, tunits, iunits


def get_post_processing_parameters(path: Path) -> dict[str, Any]:
    """Loads the post processing parameters from the post_processing_parameters.py file.

    Args:
        path: Simulation base path.

    Returns:
        The module globals dictionary (see :py:func:`run_path`).
    """
    return run_path((path / "post_processing_parameters.py").as_posix())


def get_cum_runtimes(runtimes: list[int]) -> dict[str, list[int]]:
    """Returns the number of time steps that have passed at the START of each SMC phase.

    Args:
        runtimes: The list of runtimes (acquired from the post processing parameters).

    Returns:
        Dictionary with keys for each phase (APO, ATP, ADP),
        as well as a key for all phases combined (all).
    """
    cum_runtimes: list[int] = list(np.cumsum(runtimes, dtype=int))

    map = {
        "all": [0],
        "APO": [0],
        "ATP": [],
        "ADP": [],
    }

    def append(map: dict[str, Any], key: str, value: Any) -> None:
        map[key].append(value)
        map["all"].append(value)

    for index in range(0, len(cum_runtimes), 4):
        append(map, "ATP", cum_runtimes[index])
        # skip over atp_bound_1
        append(map, "ADP", cum_runtimes[index + 2])
        append(map, "APO", cum_runtimes[index + 3])

    return map


def get_indices_array(
    id_tag_pairs_array: list[list[list[ID_TAG_PAIR]]],
    order: Sequence[str] = ["pos_lower", "pos_middle", "pos_top", "pos_center_arms"],
) -> list[IdArray]:
    """Returns arrays of indices based on an order of reference points."""
    new: list[IdArray] = []
    for i in range(len(id_tag_pairs_array)):
        new.append(np.zeros(len(id_tag_pairs_array[i]), dtype=ID_TYPE))
        for t in range(len(id_tag_pairs_array[i])):
            for reference, pair in product(order, id_tag_pairs_array[i][t]):
                if pair[1] == reference:
                    new[i][t] = pair[0]
                    break
            else:
                if id_tag_pairs_array[i][t]:
                    new[i][t] = id_tag_pairs_array[i][t][0][0]
                else:
                    new[i][t] = -1
    return new


def scale_times(times: list[int], tscale: float) -> list[float]:
    return [time * tscale for time in times]


def get_scaled_cum_runtimes(runtimes: list[int], tscale: float) -> dict[str, list[float]]:
    return {
        key: scale_times(value, tscale=tscale) for key, value in get_cum_runtimes(runtimes).items()
    }


def scale_indices(indices: list[ID_TYPE], iscale: float) -> list[float]:
    return [index * iscale for index in indices]


K = TypeVar("K")


def qzip(array: Sequence[K], n: int) -> Iterator[tuple[K, ...]]:
    """
    Creates a sequence of n arrays, each shifted to the left by i (its index).
    Useful to compute a moving average.

    e.g. array = [1,2,3,4] and n = 3
    returns => zip([1,2,3,4], [2,3,4], [3,4])
    """
    arrays: list[Sequence[K]] = []
    for i in range(n):
        arrays.append(array[i:])
    return zip(*arrays)


def get_moving_average(array, n: int) -> np.typing.NDArray:
    """Computes the moving average with window size n."""
    assert n > 0

    window = []
    for values in qzip(array, n):
        window.append(np.average(values))

    return np.array(window)


def get_site_cycle_segments(path: Path) -> list[tuple[int, int]]:
    """Return timesteps when cycling site is on: list[(start_timestep, end_timestep)]."""
    data: list[tuple[int, bool]] = []
    with open(path / "output" / "site_cycle_times.csv", "r", encoding="utf-8") as file:
        csv_data = csv.reader(file, delimiter=",")
        for row in csv_data:
            assert row[1] in {"on", "off"}
            data.append((int(row[0]), row[1] == "on"))

    if not data:
        return []

    if data[0][0] != 0:
        # prepend zero timestep, with opposite on/off state of next recorded step
        data.insert(0, (0, not data[0][1]))

    on_states: list[tuple[int, int]] = []
    # get (start, end) values for the on states
    current: int | None = None  # start off
    for step, on in data:
        if on:
            if current is None:
                current = step
            else:
                warn("Invalid data: site toggled on twice in a row!")
        else:
            if current is None:
                warn("Invalid data: site toggled off twice in a row!")
            else:
                on_states.append((current, step))
                current = None

    if current is not None:
        on_states.append((current, -1))

    return on_states


def merge_lammpstrj(base_file: Path, second_file: Path, delete_after: bool = True) -> None:
    """Merges two lammpstrj files together by removing any overlapping
    timesteps from the base file, and concatenatin the two files."""

    par = Parser(second_file)
    try:
        merge_step, _ = par.next_step()
    except Parser.EndOfLammpsFile:
        raise ValueError(f"Lammpstrj file '{second_file}' does not contain any data, cannot merge!")

    # use 'r+' to allow file.truncate() call
    par = Parser(base_file, mode="r+")
    last_step = 0
    while True:
        try:
            current_pos = par.file.tell()
            last_step, _ = par.next_step()
            if last_step >= merge_step:
                break
        except Parser.EndOfLammpsFile:
            raise ValueError(
                f"Lammpstrj file '{base_file}' does not contain enough data to perform a merge!\n"
                f"Found step {last_step} which is smaller than the required time step {merge_step}."
            )

    par.file.seek(current_pos)
    par.file.truncate()

    # close file
    del par

    with open(base_file, "ab") as first:
        with open(second_file, "rb") as second:
            shutil.copyfileobj(second, first)

    if delete_after:
        second_file.unlink()
