# -*- coding: utf-8 -*-
import typing

from collections.abc import Sequence
from typing import Any, Set, Tuple, Union

import pandas as pd


def to_seq(obj: Union[Any, typing.Sequence]) -> typing.Sequence:
    """Wrap the object in a list if it is not already sequence-like.

    Strings and tuples are not considered sequence-like.
    """
    if isinstance(obj, Sequence) and not isinstance(obj, (str, tuple)):
        return obj
    return [obj]


def to_set(obj: Any) -> Set:
    """Wrap the object in a set if it is not already a set."""
    if isinstance(obj, set):
        return obj
    return set(to_seq(obj))


# flake8: noqa: C901, WPS210, WPS221, WPS435, WPS437, WPS507
def mi_intersection(left: pd.MultiIndex, right: pd.MultiIndex) -> pd.MultiIndex:
    """ Intersection of MultiIndexes, preserving order.
    Fix https://github.com/pandas-dev/pandas/issues/31325
    """

    if left.equals(right):
        return left

    lvals = left._ndarray_values
    rvals = right._ndarray_values

    uniq_tuples = None  # flag whether _inner_indexer was successful
    if left.is_monotonic and right.is_monotonic:
        try:
            uniq_tuples = left._inner_indexer(lvals, rvals)[0]
        except TypeError:
            pass  # noqa:WPS420
    if uniq_tuples is None:
        right_uniq = set(rvals)
        seen: Set[Tuple] = set()
        uniq_tuples = [
            x
            for x in lvals
            if x in right_uniq and not (x in seen or seen.add(x))  # type: ignore
        ]
    names = left.names if left.names == right.names else None

    if len(uniq_tuples) == 0:
        return pd.MultiIndex(
            levels=left.levels,
            codes=[[]] * left.nlevels,
            names=names,
            verify_integrity=False,
        )
    return pd.MultiIndex.from_tuples(uniq_tuples, sortorder=0, names=names)