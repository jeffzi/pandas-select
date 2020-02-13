import typing

from collections.abc import Sequence
from typing import Any, Set, Union


def is_seq_like(x: Any) -> bool:
    """Check if the object is sequence-like.

    Strings objects are not considered sequence-like.
    """
    return isinstance(x, Sequence) and not isinstance(x, (str, tuple))


def to_seq(x: Union[Any, typing.Sequence]) -> typing.Sequence:
    return x if is_seq_like(x) else [x]


def to_set(x: Any) -> Set:
    if isinstance(x, set):
        return x
    return set(to_seq(x))
