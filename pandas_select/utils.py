import typing

from collections.abc import Sequence
from typing import Any, Union


def is_seq(x: Any) -> bool:
    """ Return True if x is a Sequence, except if x is a string """
    return isinstance(x, Sequence) and not isinstance(x, str)


def to_list(x: Union[Any, typing.Sequence[Any]]) -> typing.Sequence[Any]:
    return x if is_seq(x) else [x]
