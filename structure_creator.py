import numpy as np
from scipy.spatial.transform import Rotation


def get_straight_segment(n: int, direction = (1, 0, 0)):
    """returns a straight segment of n beads with unit spacing starting at
    the origin and going the in provided direction (positive x-axis by default)"""
    direction = np.array(direction, dtype=float)
    normalized_direction = direction / np.linalg.norm(direction)
    segment = np.repeat(normalized_direction, n).reshape(3, n) * np.arange(n)
    return segment.transpose()


def get_circle_segment_unit_radius(n: int, end_inclusive: bool, theta_start: float = 0, theta_end: float = 2 * np.pi, normal_direction = (0, 0, 1)):
    arange = np.arange(n) / (n - 1 if end_inclusive else n)
    thetas = theta_start + arange * (theta_end - theta_start)
    segment = np.array([np.cos(thetas), np.sin(thetas), np.zeros(len(thetas))]).reshape(3, n)

    normal_direction /= np.linalg.norm(normal_direction)
    xy_normal = np.array([0, 0, 1], dtype=float)
    
    if np.linalg.norm(normal_direction - xy_normal) > 10**(-13):
        rotation_vector = np.cross(xy_normal, normal_direction)
        rotation_angle = np.arcsin(np.linalg.norm(rotation_vector))
        rotation = Rotation.from_rotvec(rotation_vector / np.linalg.norm(rotation_vector) * rotation_angle)
        segment = rotation.as_matrix().dot(segment)

    return segment.transpose()


def get_circle_segment(n: int, end_inclusive: bool, theta_start: float = 0, theta_end: float = 2 * np.pi, normal_direction = (0, 0, 1)):
    """returns a segment of a circle of n beads with unit spacing centered at the origin
    within the plane perpendical to the given normal_direction (in the x-y plane by default)"""
    segment = get_circle_segment_unit_radius(n, end_inclusive, theta_start, theta_end, normal_direction)
    if n < 2:
        return segment
    distance = np.linalg.norm(segment[0] - segment[1])
    return segment / distance


def attach(reference_segment, other_segment, delete_overlap: bool, extra_distance: float = 0.0):
    """attaches the other_segment by moving its beginning to the end of the reference_segment"""
    extra_vector = np.zeros(len(reference_segment[0]))
    if isinstance(extra_distance, float):
        if extra_distance != 0.0:
            average_vector = (reference_segment[-1] - reference_segment[-2] + other_segment[1] - other_segment[0]) / 2.0
            extra_vector = extra_distance * average_vector

    other_segment += reference_segment[-1] - other_segment[0] + extra_vector
    if delete_overlap:
        other_segment = other_segment[1:]

    return other_segment


def attach_chain(reference_segment, list_of_args):
    """the elements of list_of_args MUST be MUTABLE (specifically args[0])"""
    for args in list_of_args:
        args[0] = attach(reference_segment, args[0], *args)
        reference_segment = args[0]
