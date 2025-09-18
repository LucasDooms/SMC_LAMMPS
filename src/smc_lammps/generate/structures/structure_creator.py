# Copyright (c) 2024-2025 Lucas Dooms

from typing import List

import numpy as np
import numpy.typing as npt
from scipy import interpolate
from scipy.integrate import trapezoid
from scipy.spatial.transform import Rotation

from smc_lammps.generate.generator import Nx3Array


def get_interpolated(spacing: float, values) -> Nx3Array:
    """spacing: distance between points along curve
    values: list of 3d points to use in the interpolation
    returns n equidistant points on an interpolated curve"""
    tck, u = interpolate.splprep(values.transpose())
    mi, ma = min(u), max(u)

    # calculate the length integral at many points
    sampling = np.linspace(mi, ma, 10000)
    derivatives_along_curve = np.array(
        interpolate.splev(sampling, tck, der=1)
    ).transpose()
    integrands = np.sqrt(np.sum(derivatives_along_curve**2, axis=1))
    lengths = np.array(
        [
            trapezoid(integrands[: i + 1], x=sampling[: i + 1])
            for i in range(len(integrands))
        ]
    )

    equidistant_points = [values[0]]
    while True:
        try:
            lengths -= spacing
            index = np.where(lengths >= 0)[0][0]
        except IndexError:
            break
        equidistant_points.append(interpolate.splev(sampling[index], tck))

    return np.array(equidistant_points)


# print(get_interpolated(100, np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0, 1, 0], [-4, 1, 0]])))


def get_straight_segment(n: int, direction=(1, 0, 0)) -> Nx3Array:
    """returns a straight segment of n beads with unit spacing starting at
    the origin and going the in provided direction (positive x-axis by default)"""
    direction = np.array(direction, dtype=np.float32)
    length: np.float32 = np.linalg.norm(direction)
    normalized_direction: npt.NDArray[np.float32] = direction / length
    segment = np.repeat(normalized_direction, n).reshape(3, n) * np.arange(n)
    return segment.transpose()


def get_circle_segment_unit_radius(
    n: int,
    end_inclusive: bool,
    theta_start: float = 0,
    theta_end: float = 2 * np.pi,
    normal_direction=(0, 0, 1),
) -> Nx3Array:
    normal_direction = np.array(normal_direction, dtype=float)

    arange = np.arange(n) / (n - 1 if end_inclusive else n)
    thetas = theta_start + arange * (theta_end - theta_start)
    segment = np.array([np.cos(thetas), np.sin(thetas), np.zeros(len(thetas))]).reshape(
        3, n
    )

    normal_direction /= np.linalg.norm(normal_direction)
    xy_normal = np.array([0, 0, 1], dtype=float)

    if np.linalg.norm(normal_direction - xy_normal) > 10 ** (-13):
        rotation_vector = np.cross(xy_normal, normal_direction)
        rotation_angle = np.arcsin(np.linalg.norm(rotation_vector))
        rotation = Rotation.from_rotvec(
            rotation_vector / np.linalg.norm(rotation_vector) * rotation_angle
        )
        segment = rotation.as_matrix().dot(segment)

    return segment.transpose()


def get_circle_segment(
    n: int,
    end_inclusive: bool,
    theta_start: float = 0,
    theta_end: float = 2 * np.pi,
    normal_direction=(0, 0, 1),
) -> Nx3Array:
    """returns a segment of a circle of n beads with unit spacing centered at the origin
    within the plane perpendical to the given normal_direction (in the x-y plane by default)"""
    segment = get_circle_segment_unit_radius(
        n, end_inclusive, theta_start, theta_end, normal_direction
    )
    if n < 2:
        return segment
    distance: np.float32 = np.linalg.norm(segment[0] - segment[1])
    return segment / distance


def attach(
    reference_segment: Nx3Array,
    other_segment: Nx3Array,
    delete_overlap: bool,
    extra_distance: float = 0.0,
) -> Nx3Array:
    """attaches the other_segment by moving its beginning to the end of the reference_segment"""
    extra_vector = np.zeros(len(reference_segment[0]))
    if isinstance(extra_distance, float):
        if extra_distance != 0.0:
            average_vector = (
                reference_segment[-1]
                - reference_segment[-2]
                + other_segment[1]
                - other_segment[0]
            ) / 2.0
            extra_vector = extra_distance * average_vector

    other_segment += reference_segment[-1] - other_segment[0] + extra_vector
    if delete_overlap:
        other_segment = other_segment[1:]

    return other_segment


def attach_chain(reference_segment: Nx3Array, list_of_args) -> List[Nx3Array]:
    """returns a list of the updated segments"""
    first_segment = reference_segment
    for i in range(len(list_of_args)):
        list_of_args[i][0] = attach(
            reference_segment, list_of_args[i][0], *list_of_args[i][1:]
        )
        reference_segment = list_of_args[i][0]
    return [first_segment] + [args[0] for args in list_of_args]
