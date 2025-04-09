# Copyright (c) 2024 Lucas Dooms

import argparse
from pathlib import Path
from typing import List, Tuple
from runpy import run_path
import subprocess

# TODO: VMD uses zero-indexed arrays!

parser = argparse.ArgumentParser(
    prog='Visualize with VMD',
    description='Creates appropriate vmd.init file and runs vmd',
    epilog='End'
)


parser.add_argument('directory', help='the directory containing LAMMPS output files')

args = parser.parse_args()
path = Path(args.directory)

parameters = run_path((path / "post_processing_parameters.py").as_posix())

class Molecules:

    nice_color_ids = [
        7, # green
        1, # red
        9, # pink
        6, # silver
    ]

    def __init__(self, path_to_vmd_init: Path) -> None:
        self.index = -1
        self.rep_index = 0
        self.color_index = 0
        self.path = path_to_vmd_init / "vmd.init"
        # clear the file
        with open(self.path, 'w'):
            pass

    def get_color_id(self) -> int:
        color_id = self.nice_color_ids[self.color_index % len(self.nice_color_ids)]
        self.color_index += 1
        return color_id

    def create_new(self, file_name: str, other_args: str) -> None:
        with open(self.path, 'a', encoding="utf-8") as file:
            file.write(f"mol new {file_name} {other_args}\n")
            self.index += 1

    def create_new_marked(self, file_name: str) -> None:
        self.create_new(file_name, "waitfor all")
        with open(self.path, 'a', encoding="utf-8") as file:
            file.write(f"mol modstyle 0 {self.index} vdw\n")

    def create_new_dna(self, file_name: str, dna_pieces: List[Tuple[int, int]], remove_ranges: List[Tuple[int, int]]) -> None:
        self.create_new(file_name, "waitfor all")
        with open(self.path, 'a', encoding="utf-8") as file:
            # show everything, slightly smaller
            file.write(f"mol modstyle 0 {self.index} cpk 0.9\n")

            # remove from ranges
            selections = []
            for rng in remove_ranges:
                selections.append(f"index < {rng[0] - 1} or index > {rng[1] - 1}")
            file.write(f"mol modselect 0 {self.index} " + " and ".join(selections) + "\n")

            self.add_dna_pieces(file, dna_pieces)

    def add_dna_pieces(self, file, dna_pieces: List[Tuple[int, int]]) -> None:
        # color the pieces differently
        file.write("mol rep cpk\n")
        for piece in dna_pieces:
            file.write(f"mol addrep {self.index}\n")
            self.rep_index += 1
            file.write(f"mol modselect {self.rep_index} {self.index} index >= {piece[0] - 1} and index <= {piece[1] - 1}\n")
            file.write(f"mol modcolor {self.rep_index} {self.index} colorID {self.get_color_id()}\n")

    def add_piece(self, rng: Tuple[int, int]) -> None:
        with open(self.path, 'a', encoding="utf-8") as file:
            file.write("mol rep cpk\n")
            file.write(f"mol addrep {self.index}\n")
            self.rep_index += 1
            file.write(f"mol modselect {self.rep_index} {self.index} index >= {rng[0] - 1} and index <= {rng[1] - 1}\n")
            file.write(f"mol modcolor {self.rep_index} {self.index} colorID {self.get_color_id()}\n")


mol = Molecules(path)

for p in path.glob("marked_bead*.lammpstrj"):
    mol.create_new_marked(p.name)

kleisins = parameters["kleisin_ids"]
kleisin_rng = (min(kleisins), max(kleisins))
mol.create_new_dna("output.lammpstrj", parameters["dna_indices_list"], [kleisin_rng])
mol.add_piece(kleisin_rng)

cmd = ["vmd", "-e", f"{mol.path.absolute()}"]
subprocess.run(cmd, cwd=path, check=True)
