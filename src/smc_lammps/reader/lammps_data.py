from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from smc_lammps.generate.generator import COORD_TYPE, Nx3Array


@dataclass
class Box:
    lows: Nx3Array
    highs: Nx3Array

    def is_in_box(self, xyz: Nx3Array) -> bool:
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
            raise ValueError("unknown Side value")

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

    def is_on_side(self, side: Plane.Side, point: Nx3Array) -> bool:
        # includes points on the plane itself
        compare = self.point.dot(self.normal)
        # for checking if inside: (point - self.point) . normal <= 0
        # thus point . normal <= self.point . normal
        # for outside: the inequality is flipped, which is equivalent
        # to multiplying both sides by (-1) (without actually flipping the inequality)
        return point.dot(self.normal) * side.value <= compare * side.value

    def distance(self, point) -> float:
        return abs((point - self.point).dot(self.normal))


def get_normal_direction(p1: Nx3Array, p2: Nx3Array, p3: Nx3Array) -> Nx3Array:
    perpendicular: Nx3Array = np.cross(p1 - p2, p1 - p3)
    return np.divide(perpendicular, np.linalg.norm(perpendicular))


ID_TYPE: TypeAlias = np.int64
IdArray: TypeAlias = npt.NDArray[ID_TYPE]
"""An array of LAMMPS ids."""
TYPE_TYPE: TypeAlias = np.int64
TypeArray: TypeAlias = npt.NDArray[TYPE_TYPE]
"""An array of LAMMPS atom types."""


@dataclass
class LammpsData:
    ids: IdArray
    types: TypeArray
    positions: Nx3Array

    @classmethod
    def empty(cls):
        return cls(
            ids=np.array([], dtype=np.int64),
            types=np.array([], dtype=np.int64),
            positions=np.array([], dtype=np.float32).reshape(-1, 3),
        )

    def filter(self, keep) -> None:
        """filters the current lists
        keep takes id (int), type (int), pos (array[float]) as input and returns bool"""
        keep_indices = keep(self.ids, self.types, self.positions)

        self.ids = self.ids[keep_indices]
        self.types = self.types[keep_indices]
        self.positions = self.positions[keep_indices]

    def filter_by_types(self, types: TypeArray) -> None:
        self.filter(lambda _, t, __: np.isin(t, types))

    def __deepcopy__(self, memo) -> LammpsData:
        new = LammpsData(np.copy(self.ids), np.copy(self.types), np.copy(self.positions))
        return new

    def delete_outside_box(self, box: Box) -> LammpsData:
        """creates a new LammpsData instance with points outside of the Box removed"""
        new = deepcopy(self)
        new.filter(lambda _, __, position: box.is_in_box(position))
        return new

    def delete_side_of_plane(self, plane: Plane, side: Plane.Side) -> None:
        """filters the current LammpsData instance to remove points on one side of a plane
        side: the side of the Plane that will get deleted"""
        self.filter(lambda _, __, pos: plane.is_on_side(Plane.Side.get_opposite(side), pos))

    def combine_by_ids(self, other: LammpsData):
        """merges two LammpsData instances by keeping all values present in any of the two
        mutates the self argument"""
        # WARNING: making no guarantees about the order
        all_ids = np.concatenate([self.ids, other.ids])
        all_types = np.concatenate([self.types, other.types])
        all_positions = np.concatenate([self.positions, other.positions])
        self.ids, indices = np.unique(all_ids, return_index=True)
        self.types = all_types[indices]
        self.positions = all_positions[indices]

    def get_position_from_index(self, index):
        return self.positions[np.where(index == self.ids)[0][0]]

    def create_box(self, types: TypeArray) -> Box:
        copy_data = deepcopy(self)
        copy_data.filter_by_types(types)
        reduced_xyz = copy_data.positions

        return Box(
            lows=np.array([np.min(reduced_xyz[:, i], axis=0) for i in range(3)], dtype=COORD_TYPE),
            highs=np.array([np.max(reduced_xyz[:, i], axis=0) for i in range(3)], dtype=COORD_TYPE),
        )
