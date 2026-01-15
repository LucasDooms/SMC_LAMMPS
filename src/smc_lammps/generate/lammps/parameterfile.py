from typing import Any, Sequence


def get_def_dynamically(name: str, value: Any) -> str:
    """Generates a LAMMPS variable definition based on the type of :py:attr:`value`.

    Args:
        name: LAMMPS variable name (not checked for validity).
        value: Value to assign to LAMMPS variable.

    Returns:
        String containing LAMMPS variable definition.

    Raises:
        TypeError: Unsupported type.
    """
    if isinstance(value, (int, float)):
        return get_equal_def(name, value)
    elif isinstance(value, str):
        return get_string_def(name, value)
    else:
        raise TypeError(f"Values of type '{type(value)}' are not supported.")


def list_to_space_str(lst: Sequence[Any], surround: str = "") -> str:
    """Converts a list into a space separated string.

    :Example:
        >>> from smc_lammps.generate.lammps.parameterfile import list_to_space_str
        >>> list_to_space_str(["hello", 1, 2.4])
        'hello 1 2.4'

    :Example:
        >>> from smc_lammps.generate.lammps.parameterfile import list_to_space_str
        >>> list_to_space_str([1, 2, 3], surround="'")
        "'1' '2' '3'"

    Args:
        lst: List of values.
        surround: Used to surround each converted value in :py:attr:`lst`.

    Returns:
        Space separated string.
    """
    return " ".join([surround + str(val) + surround for val in lst])


def prepend_or_empty(string: str, prepend: str) -> str:
    """Prepends something if the string is non-empty, otherwise returns the string "empty".

    .. Note::
        This is used when passing potentially unset values to LAMMPS.

    Args:
        string: If this is not empty, :py:attr:`prepend` is prepended to this.
        prepend: String to prepend.

    Returns:
        Prepended string or "empty".
    """
    if string:
        return prepend + string
    return "empty"


def get_equal_def(name: str, value: int | float) -> str:
    """Generates a LAMMPS equal style variable definition.

    Args:
        name: LAMMPS variable name.
        value: Integer or float value.

    Returns:
        String containing LAMMPS variable definition.
    """
    return f"variable {name} equal {value}\n"


def get_string_def(name: str, value: str) -> str:
    """Generates a LAMMPS string definition.

    Args:
        name: LAMMPS variable name.
        value: String value, will be quoted with ``"``.

    Returns:
        String containing LAMMPS variable definition.
    """
    return f'variable {name} string "{value}"\n'


def get_universe_def(name: str, values: Sequence[Any]) -> str:
    """Generates a LAMMPS universe definition.

    Args:
        name: LAMMPS variable name.
        values: Universe values. Each individual value will be quoted with ``"``.

    Returns:
        String containing LAMMPS variable definition.
    """
    return f"""variable {name} universe {list_to_space_str(values, surround='"')}\n"""


def get_index_def(name: str, values: Sequence[Any]) -> str:
    """Generates a LAMMPS index definition.

    Args:
        name: LAMMPS variable name.
        values: Index values. Values should contain no spaces, since they will not be quoted.

    Returns:
        String containing LAMMPS variable definition.
    """
    return f"variable {name} index {list_to_space_str(values)}\n"
