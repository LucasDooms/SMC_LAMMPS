from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeAlias

import numpy as np
import numpy.typing as npt

from smc_lammps.generate.generator import COORD_TYPE, Nx3Array


@dataclass
class Box:
    """
    A box in some region of space.
    """

    lows: Nx3Array
    """Lower bounds of the box (x, y, z)"""
    highs: Nx3Array
    """Upper bounds of the box (x, y, z)"""

    def is_in_box(self, xyz: Nx3Array) -> npt.NDArray[np.bool]:
        """Checks if the point(s) is/are in the box.

        Points on the surface are included.

        Args:
            xyz: Array of 3D point or points (N, 3)

        Returns:
            True if in the box, False otherwise.
        """
        condition_x = np.logical_and(self.lows[0] <= xyz[:, 0], xyz[:, 0] <= self.highs[0])
        condition_y = np.logical_and(self.lows[1] <= xyz[:, 1], xyz[:, 1] <= self.highs[1])
        condition_z = np.logical_and(self.lows[2] <= xyz[:, 2], xyz[:, 2] <= self.highs[2])

        return np.logical_and.reduce([condition_x, condition_y, condition_z], axis=0)


class Plane:
    class Side(Enum):
        OUTSIDE = -1  # on the side of the plane that the normal vector is pointing to
        INSIDE = 1  # opposite of OUTSIDE

        @classmethod
        def get_opposite(cls, side: Plane.Side) -> Plane.Side:
            if side == cls.INSIDE:
                return cls.OUTSIDE
            elif side == cls.OUTSIDE:
                return cls.INSIDE

    def __init__(self, point: Nx3Array, normal: Nx3Array):
        """point: a point on the plain,
        normal: normal vector of the plain (always normalized)"""
        normal_length = np.linalg.norm(normal)
        if normal_length == 0:
            raise ValueError("normal vector may not be zero")
        self.normal = normal / normal_length
        # take point vector to be parallel to normal vector for convenience
        # this is garantueed to still be on the same plane
        # self.point = point.dot(self.normal) * self.normal
        self.point = point

    def is_on_side(self, side: Plane.Side, points: Nx3Array) -> npt.NDArray[np.bool]:
        """Checks which side of the plane the given points are on.

        Args:
            side: Side to check.
            points: Array of 3D points.

        Returns:
            Array of bools, True for every point on `side`, False for other points.
        """
        # includes points on the plane itself
        compare = self.point.dot(self.normal)
        # for checking if inside: (point - self.point) . normal <= 0
        # thus point . normal <= self.point . normal
        # for outside: the inequality is flipped, which is equivalent
        # to multiplying both sides by (-1) (without actually flipping the inequality)
        return points.dot(self.normal) * side.value <= compare * side.value

    def distance(self, point) -> float:
        return abs((point - self.point).dot(self.normal))


def get_normal_direction(p1: Nx3Array, p2: Nx3Array, p3: Nx3Array) -> Nx3Array:
    """Returns the direction normal to the plane constructed by the three given points.

    Args:
        p1: Point 1.
        p2: Point 2.
        p3: Point 3.

    Returns:
        A normalized (1, 3) vector representing the normal direction.
    """
    perpendicular: Nx3Array = np.cross(p1 - p2, p1 - p3)
    return np.divide(perpendicular, np.linalg.norm(perpendicular))


ID_TYPE: TypeAlias = np.int64
"""LAMMPS id."""
IdArray: TypeAlias = npt.NDArray[ID_TYPE]
"""An array of LAMMPS ids."""
TYPE_TYPE: TypeAlias = np.int64
"""LAMMPS atom type."""
TypeArray: TypeAlias = npt.NDArray[TYPE_TYPE]
"""An array of LAMMPS atom types."""


@dataclass
class LammpsData:
    """
    Stores data from a LAMMPS trajectory file (\\*.lammpstrj).
    """

    ids: IdArray
    """Array of atom ids."""
    types: TypeArray
    """Array of atom types, same length as `ids`."""
    positions: Nx3Array
    """Array of atom positions, same length as `ids`."""

    @classmethod
    def empty(cls):
        """Creates an empty `LammpsData` instance."""
        return cls(
            ids=np.array([], dtype=np.int64),
            types=np.array([], dtype=np.int64),
            positions=np.array([], dtype=np.float32).reshape(-1, 3),
        )

    def filter(
        self,
        keep: Callable[[ID_TYPE, TYPE_TYPE, COORD_TYPE], np.bool]
        | Callable[[IdArray, TypeArray, Nx3Array], npt.NDArray[np.bool]],
    ) -> None:
        """Filters the arrays, edits in-place.

        Args:
            keep: Filter definition, a function from (id, type, pos) -> bool.
        """
        keep = np.vectorize(keep)
        keep_indices = keep(self.ids, self.types, self.positions)

        self.ids = self.ids[keep_indices]
        self.types = self.types[keep_indices]
        self.positions = self.positions[keep_indices]

    def filter_by_types(self, types: TypeArray) -> None:
        """Filters the arrays by the type value, edits in-place.

        Args:
            types: Types to keep.
        """
        self.filter(lambda _, t, __: np.isin(t, types))

    def __deepcopy__(self, memo) -> LammpsData:
        return LammpsData(np.copy(self.ids), np.copy(self.types), np.copy(self.positions))

    def delete_outside_box(self, box: Box) -> LammpsData:
        """Creates a new LammpsData instance with points outside of `box` removed.

        Args:
            box: Points inside of this box are kept.

        Returns:
            New LammpsData instance with only the points inside of `box`.
        """
        new = deepcopy(self)
        new.filter(lambda _, __, position: box.is_in_box(position))
        return new

    def delete_side_of_plane(self, plane: Plane, side: Plane.Side) -> None:
        """Filters the arrays by removing points on one `side` of a `plane`.

        Args:
            plane: Plane used to filter points.
            side: The side of the `plane` that will be removed.
        """
        self.filter(lambda _, __, pos: plane.is_on_side(Plane.Side.get_opposite(side), pos))

    def combine_by_ids(self, other: LammpsData) -> None:
        """Appends the values of `other` in-place.

        Args:
            other: Second LammpsData instance.
        """
        # WARNING: making no guarantees about the order
        all_ids = np.concatenate([self.ids, other.ids])
        all_types = np.concatenate([self.types, other.types])
        all_positions = np.concatenate([self.positions, other.positions])
        self.ids, indices = np.unique(all_ids, return_index=True)
        self.types = all_types[indices]
        self.positions = all_positions[indices]

    def get_position_from_index(self, index: int) -> Nx3Array:
        """Returns the position(s) at the given index / indices.

        Args:
            index: A LAMMPS atom id.
        """
        return self.positions[np.where(index == self.ids)[0][0]]

    def create_box(self, types: TypeArray) -> Box:
        """Creates the smallest box containing all atoms of the given `types`.

        Args:
            types: Types to place in box.

        Returns:
            Box containing atoms of `types`.
        """
        copy_data = deepcopy(self)
        copy_data.filter_by_types(types)
        reduced_xyz = copy_data.positions

        return Box(
            lows=np.array([np.min(reduced_xyz[:, i], axis=0) for i in range(3)], dtype=COORD_TYPE),
            highs=np.array([np.max(reduced_xyz[:, i], axis=0) for i in range(3)], dtype=COORD_TYPE),
        )
