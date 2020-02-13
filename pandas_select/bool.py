from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from pandas.api.types import is_bool_dtype

from ._base import LogicalOp, Selector
from ._utils import to_set


__all__ = ["Anywhere", "Everywhere"]

Cond = Callable[[pd.Series], Sequence[bool]]


class _BoolIndexerMixin(Selector, ABC):
    """
    Base class for selectors that filters rows by value
    """

    def __init__(
        self, cond: Cond, columns: Optional[Union[str, Iterable[str], Callable]] = None
    ):
        self.cond = cond
        if callable(columns):
            self.columns = columns
        else:
            self.columns = columns and to_set(columns)  # type:ignore

    @abstractmethod
    def _join(self, df: pd.DataFrame) -> Sequence[bool]:
        raise NotImplementedError()

    def select(self, df: pd.DataFrame) -> Sequence[bool]:
        """
        Apply the condition to each column and return a boolean array
        with size `df.shape[0]`
        """
        if self.columns is not None:
            df = df[self.columns]
        masks = df.apply(self.cond)
        return self._join(masks)


class _BoolOpsMixin:
    """
    Common logical operators mixin
    """

    def intersection(self, other: _BoolIndexerMixin) -> "BoolOp":
        """
        Return a new selector that selects elements in both selectors.
        """
        return BoolOp(np.logical_and, "&", self, other)  # type: ignore

    def union(self, other: _BoolIndexerMixin) -> "BoolOp":
        """
        Return a new selector that selects elements in the left side but not in right
        side.
        """
        return BoolOp(np.logical_or, "|", self, other)  # type: ignore

    def symmetric_difference(self, other: Any) -> "BoolOp":
        """
        Return a new selector that selects elements that are either in the left side or
        the right side but not in both.
        """
        return BoolOp(np.logical_xor, "^", self, other)  # type: ignore

    def __and__(self, other: _BoolIndexerMixin) -> "BoolOp":
        return self.intersection(other)

    def __rand__(self, other: _BoolIndexerMixin) -> "BoolOp":
        return BoolOp(np.logical_and, "&", other, self)  # type: ignore

    def __or__(self, other: _BoolIndexerMixin) -> "BoolOp":
        return self.union(other)

    def __ror__(self, other: _BoolIndexerMixin) -> "BoolOp":
        return BoolOp(np.logical_or, "|", other, self)  # type: ignore

    def __xor__(self, other: _BoolIndexerMixin) -> "BoolOp":
        return self.symmetric_difference(other)

    def __rxor__(self, other: _BoolIndexerMixin) -> "BoolOp":
        return BoolOp(np.logical_xor, "^", other, self)  # type: ignore

    def __invert__(self) -> "BoolOp":
        return BoolOp(np.invert, "~", self)  # type: ignore


class _BoolMask(Selector, _BoolOpsMixin):
    def __init__(self, mask: Sequence[bool]):
        self.mask = mask

    def select(self, df: pd.DataFrame) -> Sequence[bool]:
        return self.mask


class BoolOp(LogicalOp, _BoolOpsMixin):
    """
    A logical operation between two ``Where`` selectors
    """

    def __init__(
        self,
        op: Callable[[Sequence[bool], Optional[Sequence[bool]]], Sequence[bool]],
        op_name: str,
        left: Selector,
        right: Optional[Selector] = None,
    ):
        bool_selectors = []
        for sel in (left, right):
            if sel is not None and not isinstance(sel, _BoolOpsMixin):
                if not is_bool_dtype(sel):
                    raise TypeError(f"{sel} does not support logical operations.")
                sel = _BoolMask(sel)  # type:ignore
            bool_selectors.append(sel)

        super().__init__(op, op_name, *bool_selectors)  # type:ignore


class BoolIndexer(_BoolIndexerMixin, _BoolOpsMixin, ABC):
    pass


class Anywhere(BoolIndexer):
    """
    Filter rows where any column matches a condition.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 4], 'B': [2, -3, 1]}, index=["a", "b", "c"])
    >>> df
       A  B
    a  1  2
    b  1 -3
    c  4  1
    >>> df.loc[Anywhere(lambda x : x % 2 == 0)]
       A  B
    a  1  2
    c  4  1
    >>> df.loc[Anywhere(lambda x : x % 2 == 0, columns="A")]
       A  B
    c  4  1
    """

    def _join(self, df: pd.DataFrame) -> np.ndarray:
        return df.any(axis="columns").values


class Everywhere(BoolIndexer):
    """
    Filter rows where all columns match a condition.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 4], 'B': [2, -3, 1]}, index=["a", "b", "c"])
    >>> df
       A  B
    a  1  2
    b  1 -3
    c  4  1
    >>> df.loc[Everywhere(lambda x : x > 0)]
       A  B
    a  1  2
    c  4  1
    >>> df.loc[Everywhere(lambda x : x == 1, columns="A")]
       A  B
    a  1  2
    b  1 -3
    """

    def _join(self, df: pd.DataFrame) -> np.ndarray:
        return df.all(axis="columns").values
