# Copyright (c) 2024 Lucas Dooms

# post-processing to find the movement of the SMC relative to the DNA

from __future__ import annotations
from runpy import run_path
from sys import argv
from pathlib import Path
from itertools import islice
from copy import deepcopy
from typing import Tuple, List, Dict
from io import StringIO
from dataclasses import dataclass
from enum import Enum
import numpy as np
from time import time


def timer(func):
    def timed_func(*args, **kwargs):
        time_start = time()
        ret_value = func(*args, **kwargs)
        time_end = time()
        print(time_end - time_start)
        return ret_value
    return timed_func


cached = dict()
def timer_accumulator(func):
    global cached
    if func not in cached:
        cached[func] = 0.0
    def timed_func(*args, **kwargs):
        time_start = time()
        ret_value = func(*args, **kwargs)
        time_end = time()
        cached[func] += time_end - time_start
        return ret_value
    return timed_func


@dataclass
class Box:
    xlo: float
    xhi: float
    ylo: float
    yhi: float
    zlo: float
    zhi: float

    def is_in_box(self, xyz) -> bool:
        xyz = np.array(xyz)
        condition_x = np.logical_and(self.xlo <= xyz[:,0], xyz[:,0] <= self.xhi)
        condition_y = np.logical_and(self.ylo <= xyz[:,1], xyz[:,1] <= self.yhi)
        condition_z = np.logical_and(self.zlo <= xyz[:,2], xyz[:,2] <= self.zhi)

        return np.logical_and.reduce([condition_x, condition_y, condition_z], axis=0)


class Plane:

    class Side(Enum):
        OUTSIDE = -1 # on the side of the plane that the normal vector is pointing to
        INSIDE = 1 # opposite of OUTSIDE

        @classmethod
        def get_opposite(cls, side: Plane.Side) -> Plane.Side:
            if side == cls.INSIDE:
                return cls.OUTSIDE
            elif side == cls.OUTSIDE:
                return cls.INSIDE
            raise ValueError("unknown Side value")
    
    def __init__(self, point: List[float], normal: List[float]):
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

    def is_on_side(self, side: Plane.Side, point) -> bool:
        # includes points on the plane itself
        compare = self.point.dot(self.normal)
        # for checking if inside: (point - self.point) . normal <= 0
        # thus point . normal <= self.point . normal
        # for outside: the inequality is flipped, which is equivalent
        # to multiplying both sides by (-1) (without actually flipping the inequality)
        return point.dot(self.normal) * side.value <= compare * side.value


@dataclass
class LammpsData:

    ids: List[int]
    types: List[int]
    positions: List[List[float]]

    @timer_accumulator
    def filter(self, keep) -> None:
        """filters the current lists
        keep takes id (int), type (int), pos (array[float]) as input and returns bool"""
        keep_indices = keep(self.ids, self.types, self.positions)

        self.ids = self.ids[keep_indices]
        self.types = self.types[keep_indices]
        self.positions = self.positions[keep_indices]

    def filter_by_types(self, types: List[int]) -> None:
        self.filter(lambda _, t, __: np.isin(t, types))

    def __deepcopy__(self, memo) -> LammpsData:
        new = LammpsData(
            np.copy(self.ids),
            np.copy(self.types),
            np.copy(self.positions)
        )
        return new

    def delete_outside_box(self, box: Box) -> LammpsData:
        """creates a new LammpsData instance with points outside of the Box removed"""
        new = deepcopy(self)
        new.filter(lambda _, __, position: box.is_in_box(position))
        return new

    def delete_side_of_plane(self, plane: Plane, side: Plane.Side) -> None:
        """filters the current LammpsData instance to remove points on one side of a plane
        side: the side of the Plane that will get deleted"""
        self.filter(
            lambda _, __, pos: plane.is_on_side(Plane.Side.get_opposite(side), pos)
        )

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


class Parser:

    atom_format = "ITEM: ATOMS id type x y z\n"

    class EndOfLammpsFile(Exception):
        pass

    def __init__(self, file_name: str) -> None:
        self.file = open(file_name, 'r')

    def skip_to_atoms(self) -> Dict[str, str]:
        saved = dict()
        current_line = None
        empty = True
        for line in self.file:
            empty = False
            if line.startswith("ITEM: ATOMS"):
                if line != self.atom_format:
                    raise ValueError(f"Wrong format of atoms, found\n{line}\nshould be\n{self.atom_format}\n")
                return saved
            # remove newline
            line = line[:-1]
            if line.startswith("ITEM:"):
                saved[line] = []
                current_line = line
            else:
                saved[current_line].append(line)
        if empty:
            raise self.EndOfLammpsFile()
        raise ValueError("reached end of file unexpectedly")

    @staticmethod
    @timer_accumulator
    def get_array(lines):
        lines = "".join(lines)
        with StringIO(lines) as file:
            array = np.loadtxt(file)
        return array

    def next_step(self) -> Tuple[int, List[List[float]]]:
        """returns timestep and list of [id, type, x, y, z] for each atom"""

        saved = self.skip_to_atoms()
        timestep = int(saved["ITEM: TIMESTEP"][0])
        number_of_atoms = int(saved["ITEM: NUMBER OF ATOMS"][0])

        lines = list(islice(self.file, number_of_atoms))
        if len(lines) != number_of_atoms:
            raise ValueError("reached end of file unexpectedly")

        array = self.get_array(lines)
        
        return timestep, array
    
    @staticmethod
    def split_data(array) -> LammpsData:
        """split array into ids, types, xyz"""
        ids, types, x, y, z = array.transpose()
        xyz = np.concatenate([x, y, z]).reshape(3, -1).transpose()
        return LammpsData(np.array(ids, dtype=int), np.array(types, dtype=int), xyz)

    def __del__(self) -> None:
        self.file.close()


def create_box(data: LammpsData, types: List[int]) -> Box:
    copy_data = deepcopy(data)
    copy_data.filter_by_types(types)
    reduced_xyz = copy_data.positions
    
    return Box(
        xlo = np.min(reduced_xyz[:,0]),
        xhi = np.max(reduced_xyz[:,0]),
        ylo = np.min(reduced_xyz[:,1]),
        yhi = np.max(reduced_xyz[:,1]),
        zlo = np.min(reduced_xyz[:,2]),
        zhi = np.max(reduced_xyz[:,2])
    )


def distance_point_to_plane(point, point_on_plane, normal_direction) -> float:
    return abs((point - point_on_plane).dot(normal_direction))


def get_normal_direction(p1, p2, p3):
    perpendicular = np.cross(p1 - p2, p1 - p3)
    return perpendicular / np.linalg.norm(perpendicular)


def write(file, steps, positions):
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
        file.write(Parser.atom_format)
        file.write(f"1 1 {position[0]} {position[1]} {position[2]}\n")


def is_sorted(lst) -> bool:
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


def split_into_index_groups(indices):
    """splits a (sorted) list of integers into groups of adjacent integers"""
    indices = list(indices)
    if not indices:
        return indices

    if not is_sorted(indices):
        raise ValueError("indices must be sorted")
    
    groups = [[indices[0]]]
    for index in indices[1:]:
        if groups[-1][-1] == index - 1:
            groups[-1].append(index)
        else:
            groups.append([index])

    return groups


def get_bead_distances(positions, pos1: int, pos2: int, pos3: int):
    normal_vector = get_normal_direction(pos1, pos2, pos3)

    return distance_point_to_plane(positions, pos1, normal_vector)


def remove_outside_planar_n_gon(data: LammpsData, points, delta: float):
    """removes all points outside a box created from an n-sided polygon in a plane.
    the points are the vertices of the n-gon and are assumed to be in the same plane.
    delta is the thickness of the box in each direction going out of the plane."""
    if len(points) < 3:
        raise ValueError("there must be at least 3 points in the n-gon")
    # normal to n-gon plane
    try:
        normal = np.cross(points[1] - points[0], points[10] - points[9])
    except IndexError:
        normal = np.cross(points[1] - points[0], points[2] - points[1])

    # delete points further than delta perpendicular to the n-gon
    plane = Plane(points[0] + delta * normal, normal)
    data.delete_side_of_plane(plane, Plane.Side.OUTSIDE)
    plane = Plane(points[0] - delta * normal, normal)
    data.delete_side_of_plane(plane, Plane.Side.INSIDE)

    # delete points far away in the plane of the n-gon
    for point1, point2 in zip(points, points[1:] + [points[0]]):
        side_vector = point2 - point1
        normal_to_side = np.cross(normal, side_vector) # always points INSIDE
        side_plane = Plane(point1, normal_to_side)
        # INSIDE of plane points out of the shape
        data.delete_side_of_plane(side_plane, Plane.Side.INSIDE)


def handle_dna_bead(data: LammpsData, new_data: LammpsData, indices, positions, parameters, step):
    if len(new_data.positions) == 0:
        indices.append(-1)
        positions.append(data.positions[indices[-1]])
        return

    pos_top = data.get_position_from_index(parameters["top_bead_id"])
    pos_left = data.get_position_from_index(parameters["left_bead_id"])
    pos_right = data.get_position_from_index(parameters["right_bead_id"])
    pos_middle_left = data.get_position_from_index(parameters["middle_left_bead_id"])
    pos_middle_right = data.get_position_from_index(parameters["middle_right_bead_id"])
    pos_kleisins = [data.get_position_from_index(i) for i in parameters["kleisin_ids"]]

    new_data_copy1 = deepcopy(new_data)
    remove_outside_planar_n_gon(new_data, [pos_top, pos_left, pos_right], 0.5 * parameters["dna_spacing"])

    new_data_copy2 = deepcopy(new_data_copy1)
    # TODO shape is not planar, currently use two triangles
    delta = 0.5 * parameters["dna_spacing"]
    remove_outside_planar_n_gon(new_data_copy2, [pos_middle_right, pos_middle_left, pos_left], delta)
    new_data.combine_by_ids(new_data_copy2)

    new_data_copy3 = deepcopy(new_data_copy1)
    delta = 0.5 * parameters["dna_spacing"]
    remove_outside_planar_n_gon(new_data_copy3, [pos_middle_right, pos_left, pos_right], delta)
    new_data.combine_by_ids(new_data_copy3)

    remove_outside_planar_n_gon(new_data_copy1, pos_kleisins, 0.5 * parameters["dna_spacing"])
    new_data.combine_by_ids(new_data_copy1)

    if len(new_data.positions) == 0:
        print(f"call: {step}")
        indices.append(-1)
        positions.append(data.positions[indices[-1]])
        return

    # find groups
    grps = split_into_index_groups(new_data.ids)
    # TODO: there are still bugs in finding the index
    # if step == 50250000:
    #     print(grps)
    #     print(new_data_copy1.ids)
    #     print(new_data_copy2.ids)
    #     print(new_data_copy3.ids)

    grp = grps[0]
    grp = [np.where(new_data.ids == id)[0][0] for id in grp]

    distances = get_bead_distances(new_data.positions, pos_top, pos_left, pos_right)

    closest_val = np.min(distances[grp])
    closest_bead_index = np.where(distances == closest_val)[0][0]
    indices.append(new_data.ids[closest_bead_index])
    positions.append(new_data.positions[closest_bead_index])


def get_best_match_dna_bead_in_smc(folder_path):
    """
    For each timestep:
        create box around SMC
        remove DNA not within box
        remove DNA part of lower segment (we only want the upper segment here)
        find DNA closest to SMC plane
    """
    parameters = run_path((path / "post_processing_parameters.py").as_posix())

    par = Parser(folder_path / "output.lammpstrj")
    dna_indices_list = parameters["dna_indices_list"]
    steps = []
    indices_array = [[] for _ in range(len(dna_indices_list))]
    positions_array = [[] for _ in range(len(dna_indices_list))]
    while True:
        try:
            step, arr = par.next_step()
        except Parser.EndOfLammpsFile:
            break

        steps.append(step)

        data = Parser.split_data(arr)
        box = create_box(data, list(range(2, 10)))

        new_data = data.delete_outside_box(box)
        new_data.filter_by_types([1])
        # split, and call for each
        for i, (min_index, max_index) in enumerate(dna_indices_list):
            # if i == 1:
            #     continue
            new_data_temp = deepcopy(new_data)
            new_data_temp.filter(lambda id, _, __: np.logical_and(min_index <= id, id <= max_index))
            handle_dna_bead(data, new_data_temp, indices_array[i], positions_array[i], parameters, step)

    # delete old files
    for p in folder_path.glob("marked_bead*.lammpstrj"):
        p.unlink()

    for i, positions in enumerate(positions_array):
        with open(folder_path / f"marked_bead{i}.lammpstrj", 'w') as file:
            write(file, steps, positions)

    print(cached)

    for i, indices in enumerate(indices_array):
        np.savez(folder_path / f"bead_indices{i}.npz", steps=steps, ids=indices)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6), dpi=144)
    plt.title("Index of DNA bead inside SMC loop in time")
    plt.xlabel("time")
    plt.ylabel("DNA bead index")
    for i in range(len(indices_array)):
        plt.scatter(steps, indices_array[i], s=0.5, label=f"DNA {i}")
    plt.legend()
    plt.savefig(folder_path / "bead_id_in_time.png")


def get_msd_obstacle(folder_path):
    par = Parser(folder_path / "obstacle.lammpstrj")
    steps = []
    positions = []
    while True:
        try:
            step, arr = par.next_step()
        except Parser.EndOfLammpsFile:
            break

        steps.append(step)
        positions.append(arr[2:5])

    steps = np.array(steps)
    positions = np.array(positions)

    print(cached)

    # calculate msd in time chunks
    time_chunk_size = 2000 # number of timesteps to pick for one window

    def calculate_msd(array) -> float:
        return np.average((array - array[0])**2)

    def apply_moving_window(window_size: int, func, array):
        """returns the array of the results from applying the func to
        windows along the array."""
        result = []
        for i in range(len(array) - window_size + 1):
            result.append(
                func(array[i:i + window_size])
            )
        return result

    import matplotlib.pyplot as plt

    for i, text in enumerate(("$x$", "$y$", "$z$", "total")):
        if i != 3:
            msd_array = apply_moving_window(time_chunk_size, calculate_msd, positions[:,i])
            all_av = calculate_msd(positions[:,i])
        else:
            msd_array = apply_moving_window(time_chunk_size, calculate_msd, positions)
            all_av = calculate_msd(positions)
        plt.figure(figsize=(8, 6), dpi=144)
        plt.title(f"{text} MSD over time in chunks of {(steps[1] - steps[0]) * time_chunk_size} (all average = {all_av})")
        plt.xlabel("time")
        plt.ylabel(f"MSD ({text})")
        plt.scatter(steps[:-time_chunk_size+1], msd_array, s=0.5)
        plt.savefig(folder_path / f"msd_in_time{i}.png")


def test_plane_distances():
    p1 = np.array([0, 1, 0], dtype=float)
    p2 = np.array([0, 1, 2], dtype=float)
    p3 = np.array([1, 0, 1], dtype=float)
    n = get_normal_direction(p1, p2, p3)
    print(n)
    point = np.array([2, 2, 5], dtype=float)
    another_one = np.array([1, 1 ,1], dtype=float)
    print(distance_point_to_plane(np.array([point, another_one, another_one]), p1, n))


def test_plane_comparisons():
    point_on_plane = np.array([0, 0, 0], dtype=float)
    n = np.array([1, 0, 0], dtype=float)
    plane = Plane(point_on_plane, n)
    points = np.array([[0, 0, 0], [0.5, 0, 0], [-20, 0, 0]])
    print(plane.is_on_side(Plane.Side.INSIDE, points)) # should be [True, False, True]
    print(plane.is_on_side(Plane.Side.OUTSIDE, points)) # should be [True, True, False]

    plane2 = Plane(np.array([1, 0, 0], dtype=float), np.array([1, 1, 0], dtype=float))
    print(plane2.is_on_side(Plane.Side.INSIDE, points)) # should be [True, True, True]
    print(plane2.is_on_side(Plane.Side.INSIDE, np.array([1, 0.2, 0], dtype=float))) # should be False


if __name__ == "__main__":
    argv = argv[1:]
    if len(argv) != 1:
        raise Exception("Please provide a folder path")
    path = Path(argv[0])
    get_best_match_dna_bead_in_smc(path)
    get_msd_obstacle(path)
