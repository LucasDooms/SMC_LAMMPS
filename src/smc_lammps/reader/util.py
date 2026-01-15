from time import time


def timer(func):
    """Decorator that prints the time spent calling a function."""
    def timed_func(*args, **kwargs):
        time_start = time()
        ret_value = func(*args, **kwargs)
        time_end = time()
        print(time_end - time_start)
        return ret_value

    return timed_func


def get_timer_accumulator(cache):
    """
    :Usage:
        >>> from smc_lammps.reader.util import get_timer_accumulator
        >>> # create a global cache
        >>> cache = {}
        >>> timer_accumulator = get_timer_accumulator(cache)
        >>>
        >>> @timer_accumulator
        ... def my_func():
        ...     # do stuff here
        ...     pass
        ...
        >>> # call your function
        >>> for _ in range(1000):
        ...    my_func()
        >>>
        >>> # each element shows the time spent per function
        >>> print(len(cache))
        1
    """

    def timer_accumulator(func):
        """Store total time elapsed across any number of function calls."""
        nonlocal cache
        if func not in cache:
            cache[func] = 0.0

        def timed_func(*args, **kwargs):
            time_start = time()
            ret_value = func(*args, **kwargs)
            time_end = time()
            cache[func] += time_end - time_start
            return ret_value

        return timed_func

    return timer_accumulator
