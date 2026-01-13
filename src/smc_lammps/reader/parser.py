from __future__ import annotations

import typing
from io import StringIO
from pathlib import Path

if typing.TYPE_CHECKING:
    from _typeshed import OpenTextMode

import numpy as np
import numpy.typing as npt

from smc_lammps.generate.generator import COORD_TYPE
from smc_lammps.reader.lammps_data import ID_TYPE, TYPE_TYPE, LammpsData
from smc_lammps.reader.util import get_timer_accumulator


class Parser:
    """
    LAMMPS trajectory file parser.

    Attributes:
        file: Open LAMMPS trajectory file handle.
        time_it: If True, record the time spent parsing in `timings`.
        mode: Custom file mode (e.g. "r+" if you want to edit the file after reaching a certain timestep).
    """

    ATOM_FORMAT = "ITEM: ATOMS id type x y z\n"
    """Format of header for atoms section of trajectory file."""

    class EndOfLammpsFile(Exception):
        """
        Raised when the end of the file has been reached.
        """

        pass

    def __init__(self, file: Path, time_it: bool = False, mode: OpenTextMode = "r") -> None:
        self.file = open(file, mode, encoding="utf-8")

        self.timings = None
        if time_it:
            self.timings = dict()
            timer_accumulator = get_timer_accumulator(self.timings)
            self.next_step = timer_accumulator(self.next_step)

    def skip_to_atoms(self) -> dict[str, list[str]]:
        """Iterate through the file

        Returns:
            Dictionary which maps the header ('ITEM: .*') to a list of lines under the header.

        Raises:
            ValueError: Invalid format found, must match :py:attr:`Parser.ATOM_FORMAT`.
            self.EndOfLammpsFile: Successfully reached the end of the file.
            ValueError: Reached the end of the file while parsing.
        """
        saved: dict[str, list[str]] = dict()
        current_line: None | str = None
        empty: bool = True

        # NOTE: use readline instead of a for loop,
        # since the latter breaks file.seek() calls
        while line := self.file.readline():
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
                assert current_line is not None
                saved[current_line].append(line)

        if empty:
            raise self.EndOfLammpsFile()

        raise ValueError("reached end of file unexpectedly")

    @staticmethod
    def get_array(lines: list[str]) -> npt.NDArray:
        """Returns the id, type, x, y, z data for a single timestep.

        Args:
            lines: Lines in the atom section of the LAMMPS trajectory file.
        """
        all_lines = "".join(lines)
        with StringIO(all_lines) as file:
            array = np.loadtxt(file, ndmin=2)
        return array

    @staticmethod
    def split_data(array) -> LammpsData:
        """Converts array to LammpsData.

        Args:
            array: Array of id, type, x, y, z data.

        Returns:
            A LammpsData instance for a single timestep.
        """
        ids, types, x, y, z = array.transpose()
        xyz = np.array(np.concatenate([x, y, z]).reshape(3, -1).transpose(), dtype=COORD_TYPE)
        return LammpsData(np.array(ids, dtype=ID_TYPE), np.array(types, dtype=TYPE_TYPE), xyz)

    def next_step(self) -> tuple[int, LammpsData]:
        """Iterates through the file and retrieves the next timestep.

        Returns:
            Tuple of (timestep, data).

        Raises:
            ValueError: Reached the end of the file while parsing.
        """

        saved = self.skip_to_atoms()
        timestep = int(saved["ITEM: TIMESTEP"][0])
        number_of_atoms = int(saved["ITEM: NUMBER OF ATOMS"][0])

        lines: list[str] = []
        for _ in range(number_of_atoms):
            try:
                next = self.file.readline()
            except Exception as e:
                raise ValueError("reached end of file unexpectedly") from e
            else:
                lines.append(next)

        data = self.split_data(self.get_array(lines))

        return timestep, data

    def __del__(self) -> None:
        # check if attribute exists, since __init__ may fail
        if hasattr(self, "file"):
            self.file.close()
