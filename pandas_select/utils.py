from collections.abc import Iterable


def is_seq(x):
    """Return True if x is iterable, except if x is a string"""
    return isinstance(x, Iterable) and not isinstance(x, str)


def to_list(x):
    return x if is_seq(x) else [x]
