import argparse
from pathlib import Path
from typing import List, Tuple
from importlib import import_module
import subprocess


parser = argparse.ArgumentParser(
    prog='Visualize with VMD',
    description='Creates appropriate vmd.init file and runs vmd',
    epilog='End'
)


parser.add_argument('directory', help='the directory containing LAMMPS output files')

args = parser.parse_args()
path = Path(args.directory)

parameters = import_module((path / "post_processing_parameters").as_posix().replace('/', '.'))

class Molecules:

    nice_color_ids = [
        7, # green
        1, # red
        9, # pink
        6, # silver
    ]

    def __init__(self, path_to_vmd_init: Path) -> None:
        self.index = -1
        self.color_index = 0
        self.path = path_to_vmd_init / "vmd.init"
        # clear the file
        with open(self.path, 'w'):
            pass

    def get_color_id(self) -> int:
        color_id = self.nice_color_ids[self.color_index % len(self.nice_color_ids)]
        self.color_index += 1
        return color_id

    def create_new(self, file_name: str) -> None:
        with open(self.path, 'a') as file:
            file.write(f"mol new {file_name}\n")
            self.index += 1
    
    def create_new_marked(self, file_name: str) -> None:
        self.create_new(file_name)
        with open(self.path, 'a') as file:
            file.write(f"mol modstyle 0 {self.index} vdw\n")
    
    def create_new_dna(self, file_name: str, dna_pieces: List[Tuple[int, int]]) -> None:
        self.create_new(file_name)
        with open(self.path, 'a') as file:
            # show everything, slightly smaller
            file.write(f"mol modstyle 0 {self.index} cpk 0.9\n")

            # color the pieces differently
            file.write(f"mol rep cpk\n")
            rep_index = 0
            for piece in dna_pieces:
                file.write(f"mol addrep {self.index}\n")
                rep_index += 1
                file.write(f"mol modselect {rep_index} {self.index} index >= {piece[0]} and index <= {piece[1]}\n")
                file.write(f"mol modcolor {rep_index} {self.index} colorID {self.get_color_id()}\n")


mol = Molecules(path)

mol.create_new_marked("marked_bead.lammpstrj")
try:
    mol.create_new_marked("marked_bead2.lammpstrj")
except:
    pass

mol.create_new_dna("output.lammpstrj", parameters.dna_indices_list)

cmd = ["vmd", "-e", f"{mol.path.absolute()}"]
subprocess.run(cmd, cwd=path)
