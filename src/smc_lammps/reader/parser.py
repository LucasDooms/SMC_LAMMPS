from __future__ import annotations

from io import StringIO
from itertools import islice
from pathlib import Path

import numpy as np

from smc_lammps.generate.generator import COORD_TYPE
from smc_lammps.reader.lammps_data import ID_TYPE, TYPE_TYPE, LammpsData
from smc_lammps.reader.util import get_timer_accumulator


class Parser:
    ATOM_FORMAT = "ITEM: ATOMS id type x y z\n"

    class EndOfLammpsFile(Exception):
        pass

    def __init__(self, file: Path, time_it: bool = False) -> None:
        self.file = open(file, "r", encoding="utf-8")

        self.timings = None
        if time_it:
            self.timings = dict()
            timer_accumulator = get_timer_accumulator(self.timings)
            self.next_step = timer_accumulator(self.next_step)

    def skip_to_atoms(self) -> dict[str, str]:
        saved = dict()
        current_line = None
        empty = True

        for line in self.file:
            empty = False

            if line.startswith("ITEM: ATOMS"):
                if line != self.ATOM_FORMAT:
                    raise ValueError(
                        f"Wrong format of atoms, found\n{line}\nshould be\n{self.ATOM_FORMAT}\n"
                    )
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
    def get_array(lines):
        lines = "".join(lines)
        with StringIO(lines) as file:
            array = np.loadtxt(file, ndmin=2)
        return array

    @staticmethod
    def split_data(array) -> LammpsData:
        """split array into ids, types, xyz"""
        ids, types, x, y, z = array.transpose()
        xyz = np.array(np.concatenate([x, y, z]).reshape(3, -1).transpose(), dtype=COORD_TYPE)
        return LammpsData(np.array(ids, dtype=ID_TYPE), np.array(types, dtype=TYPE_TYPE), xyz)

    def next_step(self) -> tuple[int, LammpsData]:
        """returns timestep and the lammps data of all the atoms."""

        saved = self.skip_to_atoms()
        timestep = int(saved["ITEM: TIMESTEP"][0])
        number_of_atoms = int(saved["ITEM: NUMBER OF ATOMS"][0])

        lines = list(islice(self.file, number_of_atoms))
        if len(lines) != number_of_atoms:
            raise ValueError("reached end of file unexpectedly")

        data = self.split_data(self.get_array(lines))

        return timestep, data

    def __del__(self) -> None:
        # check if attribute exists, since __init__ may fail
        if hasattr(self, "file"):
            self.file.close()
