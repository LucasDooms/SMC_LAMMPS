from typing import Sequence, Any


def list_to_space_str(lst: Sequence[Any], surround="") -> str:
    """turn list into space separated string
    example: [1, 2, 6] -> 1 2 6"""
    return " ".join([surround + str(val) + surround for val in lst])


def prepend_or_empty(string: str, prepend: str) -> str:
    """prepend something if the string is non-empty
    otherwise replace it with the string "empty"."""
    if string:
        return prepend + string
    return "empty"


def get_string_def(name: str, value: str) -> str:
    """define a LAMMPS string"""
    return f'variable {name} string "{value}"\n'


def get_universe_def(name: str, values: Sequence[Any]) -> str:
    """define a LAMMPS universe"""
    return f"""variable {name} universe {list_to_space_str(values, surround='"')}\n"""


def get_index_def(name: str, values: Sequence[Any]) -> str:
    """define a LAMMPS universe"""
    return f"variable {name} index {list_to_space_str(values)}\n"
