# Copyright (c) 2025 Lucas Dooms

from pathlib import Path
from runpy import run_path
from typing import Any, Iterator, Sequence, TypeVar

import numpy as np

from smc_lammps.generate.default_parameters import Parameters
from smc_lammps.reader.lammps_data import ID_TYPE


def get_scaling(
    use_real_units: bool = True, parameters: Parameters | None = None
) -> tuple[float, float, str, str]:
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
    """Load the post processing parameters from the post_processing_parameters.py file."""
    return run_path((path / "post_processing_parameters.py").as_posix())


def get_cum_runtimes(runtimes: list[int]) -> dict[str, list[int]]:
    """Return the number of time steps that have passed at the START of each SMC phase."""
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
