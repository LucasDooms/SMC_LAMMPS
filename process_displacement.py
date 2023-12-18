# post-processing to find the movement of the SMC relative to the DNA

from __future__ import annotations
from importlib import import_module
from sys import argv
from pathlib import Path
from itertools import islice
from copy import deepcopy
from typing import Tuple, List, Dict
from io import StringIO
from dataclasses import dataclass
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

    def __deepcopy_(self, memo) -> LammpsData:
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


class Parser:

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
    def strio(lines):
        lines = "".join(lines)
        with StringIO(lines) as file:
            array = np.loadtxt(file)
        return array

    @staticmethod
    @timer_accumulator
    def strio_2(lines):
        lst = [[float(num) for num in line.split(" ")] for line in lines]
        return np.array(lst)
    
    @staticmethod
    @timer_accumulator
    def strio_3(lines, n):
        array = np.zeros((n, 5), dtype=float)
        for i, line in enumerate(lines):
            for j, num in enumerate(line.split(" ")):
                array[i][j] = float(num)
        return array

    def next_step(self) -> Tuple[int, List[List[float]]]:
        """returns timestep and list of [x, y, z] for each atom"""

        saved = self.skip_to_atoms()
        timestep = int(saved["ITEM: TIMESTEP"][0])
        number_of_atoms = int(saved["ITEM: NUMBER OF ATOMS"][0])

        lines = list(islice(self.file, number_of_atoms))
        if len(lines) != number_of_atoms:
            raise ValueError("reached end of file unexpectedly")

        array = self.strio(lines)
        
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
        file.write("ITEM: ATOMS id type xs ys zs\n")
        file.write(f"1 1 {position[0]} {position[1]} {position[2]}\n")


def get_best_match_dna_bead_in_smc(folder_name_or_path):
    """
    For each timestep:
        create box around SMC
        remove DNA not within box
        remove DNA part of lower segment (we only want the upper segment here)
        find DNA closest to SMC plane
    """
    parameters = import_module((path / "post_processing_parameters").as_posix().replace('/', '.'))

    par = Parser(folder_name_or_path / "output.lammpstrj")
    steps = []
    indices = []
    positions = []
    t_read = 0.0
    t_other_first = 0.0
    t_other_second = 0.0
    while True:
        try:
            t0 = time()
            step, arr = par.next_step()
            t_read += time() - t0
        except Parser.EndOfLammpsFile:
            break
        
        t0 = time()
        steps.append(step)

        data = Parser.split_data(arr)
        box = create_box(data, list(range(2, 10)))

        new_data = data.delete_outside_box(box)
        new_data.filter_by_types([1])
        new_data.filter(lambda id, _, __: id <= parameters.upper_dna_max_id)
        filtered = new_data.positions

        t_other_first += time() - t0
        t0 = time()

        pos_top = data.positions[parameters.top_bead_id - 1]
        pos_left = data.positions[parameters.left_bead_id - 1]
        pos_right = data.positions[parameters.right_bead_id - 1]

        normal_vector = get_normal_direction(pos_top, pos_left, pos_right)

        distances = distance_point_to_plane(filtered, pos_top, normal_vector)
        
        # sorted_indices = np.argsort(distances)
        closest_val = np.min(distances)
        closest_bead_index = np.where(distances == closest_val)[0][0]
        indices.append(new_data.ids[closest_bead_index])
        positions.append(filtered[closest_bead_index])
        
        t_other_second += time() - t0

    t0 = time()

    with open(folder_name_or_path / "marked_bead.lammpstrj", 'w') as file:
        write(file, steps, positions)

    print("write", time() - t0)
    print("read", t_read)
    print(f"{t_other_first = }")
    print(f"{t_other_second = }")
    print(cached)

    import matplotlib.pyplot as plt

    plt.scatter(steps, indices)
    plt.savefig(folder_name_or_path / "bead_id_in_time.png")


def test():
    par = Parser("test/output.lammpstrj")
    _, arr = par.next_step()
    data = Parser.split_data(arr)
    box = create_box(data, list(range(2, 10)))

    new_data = data.delete_outside_box(box)
    print(new_data.positions.shape)
    new_data.filter_by_types([1])
    filtered = new_data.positions
    print(filtered.shape)

    pos_top = data.positions[parameters.top_bead_id - 1]
    pos_left = data.positions[parameters.left_bead_id - 1]
    pos_right = data.positions[parameters.right_bead_id - 1]
    print(pos_top, pos_left, pos_right)

    normal_vector = get_normal_direction(pos_top, pos_left, pos_right)

    distances = distance_point_to_plane(filtered, pos_top, normal_vector)

    closest_val = np.min(distances)
    closest_bead_index = np.where(distances == closest_val)[0][0]
    print(filtered[closest_bead_index])


def test_plane():
    p1 = np.array([0, 1, 0], dtype=float)
    p2 = np.array([0, 1, 2], dtype=float)
    p3 = np.array([1, 0, 1], dtype=float)
    n = get_normal_direction(p1, p2, p3)
    print(n)
    point = np.array([2, 2, 5], dtype=float)
    another_one = np.array([1, 1 ,1], dtype=float)
    print(distance_point_to_plane(np.array([point, another_one, another_one]), p1, n))


if __name__ == "__main__":
    argv = argv[1:]
    if len(argv) != 1:
        raise Exception("Please provide a folder path")
    path = Path(argv[0])
    get_best_match_dna_bead_in_smc(path)
