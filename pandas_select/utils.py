import typing

from collections.abc import Sequence
from typing import Any, Union


def is_list_like(x: Any) -> bool:
    """  Check if the object is list-like.

    Strings objects are not considered list-like.
    """
    return isinstance(x, Sequence) and not isinstance(x, str)


def to_list(x: Union[Any, typing.Sequence[Any]]) -> typing.Sequence[Any]:
    return x if is_list_like(x) else [x]
