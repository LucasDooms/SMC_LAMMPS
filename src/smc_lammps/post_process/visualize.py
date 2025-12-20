# Copyright (c) 2024-2025 Lucas Dooms

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path
from runpy import run_path
from typing import Sequence

# WARN: VMD uses zero-indexed arrays!

parser = argparse.ArgumentParser(
    prog="Visualize with VMD",
    description="creates appropriate vmd.init file and runs vmd",
    epilog="End",
)


parser.add_argument("directory", help="the directory containing LAMMPS output files")
fn_arg = parser.add_argument(
    "-f",
    "--file_name",
    help="name of file, default: 'output/output.lammpstrj'",
    default="output/output.lammpstrj",
)
parser.add_argument(
    "-s",
    "--smoothing",
    type=int,
    help="trajectory smoothing window size (default=0)",
    default=0,
)

args = parser.parse_args()
path = Path(args.directory)

ppp = run_path((path / "post_processing_parameters.py").as_posix())

output_file = path / Path(args.file_name)
if not output_file.exists():
    raise FileNotFoundError(f"Cannot visualize '{output_file}', file does not exist.")

if args.smoothing < 0:
    raise ValueError(f"smoothing must be positive ({args.smoothing} < 0)")


class Molecules:
    nice_color_ids = [
        7,  # green
        1,  # red
        9,  # pink
        6,  # silver
    ]

    def __init__(self, base_path: Path) -> None:
        self.index = -1
        self._rep_index: defaultdict[int, int] = defaultdict(
            lambda: 0
        )  # store the current rep for each mol (index)
        self.color_index = 0
        path_to_vmd = base_path / "vmd"
        path_to_vmd.mkdir(exist_ok=True)
        self.path = path_to_vmd / "vmd.init"
        # open the file (overwrite previous contents)
        self.file = open(self.path, "w", encoding="utf-8")

    @property
    def rep_index(self) -> int:
        if self.index < 0:
            raise ValueError(f"index ({self.index}) is negative, did you load a molecule?")

        return self._rep_index[self.index]

    @rep_index.setter
    def rep_index(self, value: int) -> None:
        # get the current value to do error checking
        _ = self.rep_index
        self._rep_index[self.index] = value

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

    def add_rep(self) -> None:
        self.rep_index += 1
        self.file.write(f"mol addrep {self.index}\n# ----- rep {self.rep_index} -----\n")

    def set_all_smoothing(self, smoothing: int) -> None:
        if smoothing < 0:
            raise ValueError(f"smoothing must be positive ({smoothing} < 0)")

        if smoothing == 0:
            # this is the default, no need to add anything
            return

        self.file.write("\n# === smoothing ===\n")
        for mol_index in range(self.index + 1):
            for rep_index in range(self._rep_index[mol_index] + 1):
                self.file.write(f"mol smoothrep {mol_index} {rep_index} {smoothing}\n")

    def set_animate_once(self) -> None:
        self.file.write("animate style once\n")

    def set_animate_start(self) -> None:
        self.file.write("animate goto start\n")

    def create_new(self, file: Path, **other_args: str) -> None:
        # waitfor all by default, greatly reduces load time for large files
        other_args.setdefault("waitfor", "all")
        # assume lammpstrj,
        # useful for restart output files named 'output.lammpstrj.*'
        # which vmd does not automatically recognize
        other_args.setdefault("type", "lammpstrj")

        space_separated_args = " ".join([f'{key} "{value}"' for key, value in other_args.items()])

        self.index += 1
        self.file.write(f"\n# >>> mol {self.index}\n")
        self.file.write(
            f'mol new "{file.relative_to(self.path.parent, walk_up=True)}" {space_separated_args}\n'
        )
        self.file.write(f"# ----- rep {self.rep_index} -----\n")

    def create_new_marked(self, file: Path) -> None:
        self.create_new(file)
        self.file.write(f"mol modstyle {self.rep_index} {self.index} vdw\n")

    def load_trajectory(
        self,
        file: Path,
        remove_ranges: Sequence[tuple[int, int]],
    ) -> None:
        self.create_new(file)
        # show everything, slightly smaller
        self.file.write(f"mol modstyle {self.rep_index} {self.index} vdw 0.4\n")

        # remove from ranges
        selections = []
        for rng in remove_ranges:
            selections.append(f"index < {rng[0] - 1} or index > {rng[1] - 1}")
        self.file.write(
            f"mol modselect {self.rep_index} {self.index} " + " and ".join(selections) + "\n"
        )

    def add_dna_pieces(self, dna_pieces: Sequence[tuple[int, int]]) -> None:
        # color the pieces differently
        for piece in dna_pieces:
            self.add_rep()
            self.file.write(
                f"mol modselect {self.rep_index} {self.index} index >= {piece[0] - 1} and index <= {piece[1] - 1}\n"
            )
            self.file.write(
                f"mol modcolor {self.rep_index} {self.index} colorID {self.get_color_id()}\n"
            )
            self.file.write(f"mol modstyle {self.rep_index} {self.index} cpk 1.4\n")

    def add_piece(self, rng: tuple[int, int]) -> None:
        self.add_rep()
        self.file.write(
            f"mol modselect {self.rep_index} {self.index} index >= {rng[0] - 1} and index <= {rng[1] - 1}\n"
        )
        self.file.write(
            f"mol modcolor {self.rep_index} {self.index} colorID {self.get_color_id()}\n"
        )
        self.file.write(f"mol modstyle {self.rep_index} {self.index} cpk 1.4\n")

    def add_spaced_beads(self, spaced_beads: Sequence[int]) -> None:
        if not spaced_beads:
            return

        self.add_rep()
        vmd_indices = " ".join(str(id - 1) for id in spaced_beads)
        self.file.write(f"mol modselect {self.rep_index} {self.index} index {vmd_indices}\n")
        # choose color based on index
        self.file.write(f"mol modcolor {self.rep_index} {self.index} PosX\n")
        # TODO: get size dynamically
        self.file.write(f"mol modstyle {self.rep_index} {self.index} vdw 1.5\n")


mol = Molecules(path)

if args.file_name == fn_arg.default:
    for p in path.glob("marked_bead*.lammpstrj"):
        mol.create_new_marked(p)

kleisins = ppp["kleisin_ids"]
kleisin_rng = (min(kleisins), max(kleisins))
mol.load_trajectory(output_file, [*ppp["dna_indices_list"], kleisin_rng])
mol.add_dna_pieces(ppp["dna_indices_list"])
mol.add_piece(kleisin_rng)
mol.add_spaced_beads(ppp["spaced_bead_indices"])

# run after all mols which should be smoothed have been added
mol.set_all_smoothing(args.smoothing)
mol.file.write("\n")
mol.file.write("# === other options ===\n")
mol.file.write("\n")
mol.set_animate_once()
mol.set_animate_start()

mol.run_vmd()
