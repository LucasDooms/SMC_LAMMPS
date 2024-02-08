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
        """returns timestep and list of [x, y, z] for each atom"""

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


def get_bead_distances(data: LammpsData, positions, id1: int, id2: int, id3: int):
    pos1 = data.positions[np.where(id1 == data.ids)[0][0]]
    pos2 = data.positions[np.where(id2 == data.ids)[0][0]]
    pos3 = data.positions[np.where(id3 == data.ids)[0][0]]

    normal_vector = get_normal_direction(pos1, pos2, pos3)

    return distance_point_to_plane(positions, pos1, normal_vector)


def handle_dna_bead(data: LammpsData, new_data: LammpsData, indices, positions, parameters):
    filtered = new_data.positions

    if len(filtered) == 0:
        indices.append(-1)
        positions.append(data.positions[indices[-1]])
        return

    distances = get_bead_distances(data, filtered, parameters.top_bead_id, parameters.left_bead_id, parameters.right_bead_id)

    # take close beads
    close_beads_indices = np.where(distances <= parameters.dna_spacing)[0]

    # find groups
    grps = split_into_index_groups(close_beads_indices)
    
    try:
        grp = grps[0]
    except IndexError:
        distances = get_bead_distances(data, filtered, parameters.left_kleisin_id, parameters.right_kleisin_id, parameters.bottom_kleisin_id)

        # take close beads
        close_beads_indices = np.where(distances <= parameters.dna_spacing)[0]

        # find groups
        grps = split_into_index_groups(close_beads_indices)

        try:
            grp = grps[0]
        except IndexError:
            print("skipped")
            grp = [0]

    closest_val = np.min(distances[grp])
    closest_bead_index = np.where(distances == closest_val)[0][0]
    indices.append(new_data.ids[closest_bead_index])
    positions.append(filtered[closest_bead_index])


def get_best_match_dna_bead_in_smc(folder_path):
    """
    For each timestep:
        create box around SMC
        remove DNA not within box
        remove DNA part of lower segment (we only want the upper segment here)
        find DNA closest to SMC plane
    """
    parameters = import_module((path / "post_processing_parameters").as_posix().replace('/', '.'))

    par = Parser(folder_path / "output.lammpstrj")
    steps = []
    indices_array = [[] for _ in range(len(parameters.dna_indices_list))]
    positions_array = [[] for _ in range(len(parameters.dna_indices_list))]
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
        for i, (min_index, max_index) in enumerate(parameters.dna_indices_list):
            new_data_temp = deepcopy(new_data)
            new_data_temp.filter(lambda id, _, __: np.logical_and(min_index <= id, id <= max_index))
            handle_dna_bead(data, new_data_temp, indices_array[i], positions_array[i], parameters)

    with open(folder_path / "marked_bead.lammpstrj", 'w') as file:
        write(file, steps, positions_array[0])

    print(cached)

    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.title("Index of DNA bead inside SMC loop in time")
    plt.xlabel("time")
    plt.ylabel("DNA bead index")
    for i in range(len(indices_array)):
        plt.scatter(steps, indices_array[i], s=0.5, label=f"DNA {i}")
    plt.legend()
    plt.savefig(folder_path / "bead_id_in_time.png")


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
