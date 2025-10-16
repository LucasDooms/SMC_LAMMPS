# Copyright (c) 2024-2025 Lucas Dooms

import argparse
import subprocess
from pathlib import Path
from runpy import run_path
from typing import Sequence

# WARN: VMD uses zero-indexed arrays!

parser = argparse.ArgumentParser(
    prog="Visualize with VMD",
    description="Creates appropriate vmd.init file and runs vmd",
    epilog="End",
)


parser.add_argument("directory", help="the directory containing LAMMPS output files")
fn_arg = parser.add_argument(
    "-f",
    "--file_name",
    help="name of file, default: 'output.lammpstrj'",
    default="output.lammpstrj",
)

args = parser.parse_args()
path = Path(args.directory)

ppp = run_path((path / "post_processing_parameters.py").as_posix())


class Molecules:
    nice_color_ids = [
        7,  # green
        1,  # red
        9,  # pink
        6,  # silver
    ]

    def __init__(self, path_to_vmd_init: Path) -> None:
        self.index = -1
        self.rep_index = 0
        self.color_index = 0
        self.path = path_to_vmd_init / "vmd.init"
        # open the file (overwrite previous contents)
        self.file = open(self.path, "w", encoding="utf-8")

    def __del__(self) -> None:
        if hasattr(self, "file"):
            self.file.close()

    def run_vmd(self) -> None:
        # make sure to close the file first!
        self.file.close()

        cmd = ["vmd", "-e", f"{self.path.absolute()}"]
        subprocess.run(cmd, cwd=self.path.parent, check=True)

    def get_color_id(self) -> int:
        color_id = self.nice_color_ids[self.color_index % len(self.nice_color_ids)]
        self.color_index += 1
        return color_id

    def create_new(self, file_name: str, other_args: str) -> None:
        self.file.write(f"mol new {file_name} {other_args}\n")
        self.index += 1

    def create_new_marked(self, file_name: str) -> None:
        self.create_new(file_name, "waitfor all")
        self.file.write(f"mol modstyle 0 {self.index} vdw\n")

    def create_new_dna(
        self,
        file_name: str,
        dna_pieces: Sequence[tuple[int, int]],
        remove_ranges: Sequence[tuple[int, int]],
    ) -> None:
        self.create_new(file_name, "waitfor all")
        # show everything, slightly smaller
        self.file.write(f"mol modstyle 0 {self.index} cpk 1.3\n")

        # remove from ranges
        selections = []
        for rng in remove_ranges:
            selections.append(f"index < {rng[0] - 1} or index > {rng[1] - 1}")
        self.file.write(f"mol modselect 0 {self.index} " + " and ".join(selections) + "\n")

        self.add_dna_pieces(dna_pieces)

    def add_dna_pieces(self, dna_pieces: Sequence[tuple[int, int]]) -> None:
        # color the pieces differently
        self.file.write("mol rep cpk\n")
        for piece in dna_pieces:
            self.file.write(f"mol addrep {self.index}\n")
            self.rep_index += 1
            self.file.write(
                f"mol modselect {self.rep_index} {self.index} index >= {piece[0] - 1} and index <= {piece[1] - 1}\n"
            )
            self.file.write(
                f"mol modcolor {self.rep_index} {self.index} colorID {self.get_color_id()}\n"
            )
            self.file.write(f"mol modstyle {self.rep_index} {self.index} cpk 1.4\n")

    def add_piece(self, rng: tuple[int, int]) -> None:
        self.file.write("mol rep cpk\n")
        self.file.write(f"mol addrep {self.index}\n")
        self.rep_index += 1
        self.file.write(
            f"mol modselect {self.rep_index} {self.index} index >= {rng[0] - 1} and index <= {rng[1] - 1}\n"
        )
        self.file.write(
            f"mol modcolor {self.rep_index} {self.index} colorID {self.get_color_id()}\n"
        )
        self.file.write(f"mol modstyle {self.rep_index} {self.index} cpk 1.4\n")

    def add_spaced_beads(self, spaced_beads: Sequence[int]) -> None:
        self.file.write("mol rep vdw\n")
        self.file.write(f"mol addrep {self.index}\n")
        self.rep_index += 1
        vmd_indices = " ".join(str(id - 1) for id in spaced_beads)
        self.file.write(f"mol modselect {self.rep_index} {self.index} index {vmd_indices}\n")
        # choose color based on index
        self.file.write(f"mol modcolor {self.rep_index} {self.index} PosX\n")
        # TODO: get size dynamically
        self.file.write(f"mol modstyle {self.rep_index} {self.index} vdw 3.5\n")


mol = Molecules(path)

if args.file_name == fn_arg.default:
    for p in path.glob("marked_bead*.lammpstrj"):
        mol.create_new_marked(p.name)

kleisins = ppp["kleisin_ids"]
kleisin_rng = (min(kleisins), max(kleisins))
mol.create_new_dna(args.file_name, ppp["dna_indices_list"], [kleisin_rng])
mol.add_piece(kleisin_rng)
mol.add_spaced_beads(ppp["spaced_bead_indices"])

mol.run_vmd()
