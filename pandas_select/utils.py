import typing

from collections.abc import Sequence
from typing import Any, Set, Union


def is_list_like(x: Any) -> bool:
    """  Check if the object is list-like.

    Strings objects are not considered list-like.
    """
    return isinstance(x, Sequence) and not isinstance(x, (str, tuple))


def to_list(x: Union[Any, typing.Sequence]) -> typing.Sequence:
    return x if is_list_like(x) else [x]


def to_set(x: Any) -> Set:
    return set(to_list(x))
