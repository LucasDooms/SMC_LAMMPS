import numpy as np

from smc_lammps.generate.generator import COORD_TYPE
from smc_lammps.reader.lammps_data import Plane, get_normal_direction


def test_plane_distances():
    p1 = np.array([0, 1, 0], dtype=COORD_TYPE)
    p2 = np.array([0, 1, 2], dtype=COORD_TYPE)
    p3 = np.array([1, 0, 1], dtype=COORD_TYPE)
    n = get_normal_direction(p1, p2, p3)
    print(n)
    point = np.array([2, 2, 5], dtype=COORD_TYPE)
    another_one = np.array([1, 1, 1], dtype=COORD_TYPE)
    plane = Plane(another_one, n)
    print(plane.distance(np.array([point, another_one, another_one])))


def test_plane_comparisons():
    point_on_plane = np.array([0, 0, 0], dtype=float)
    n = np.array([1, 0, 0], dtype=float)
    plane = Plane(point_on_plane, n)
    points = np.array([[0, 0, 0], [0.5, 0, 0], [-20, 0, 0]])
    print(plane.is_on_side(Plane.Side.INSIDE, points))  # should be [True, False, True]
    print(plane.is_on_side(Plane.Side.OUTSIDE, points))  # should be [True, True, False]

    plane2 = Plane(np.array([1, 0, 0], dtype=float), np.array([1, 1, 0], dtype=float))
    print(plane2.is_on_side(Plane.Side.INSIDE, points))  # should be [True, True, True]
    print(
        plane2.is_on_side(Plane.Side.INSIDE, np.array([1, 0.2, 0], dtype=float))
    )  # should be False


def test():
    test_plane_distances()
    test_plane_comparisons()


if __name__ == "__main__":
    test()
