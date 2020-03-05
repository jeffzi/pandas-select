# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd

from pandas.api.types import is_bool_dtype, is_list_like

from pandas_select import iterutils
from pandas_select.base import LogicalOp, PrettyPrinter, Selector


Cond = Callable[[pd.Series], Iterable[bool]]
Columns = Union[str, Iterable[str], Callable]


class _BoolIndexerMixin(PrettyPrinter, ABC):
    """Base class for selectors that filters rows by value."""

    def __init__(self, cond: Cond, columns: Optional[Columns] = None):
        self.cond = cond
        if callable(columns):
            self.columns = columns
        else:
            self.columns = columns and iterutils.to_set(columns)  # type:ignore

    def __call__(self, df: pd.DataFrame) -> Iterable[bool]:
        """Apply the condition to each column.

        Return a boolean array with size `df.shape[0]`.
        """
        if self.columns is not None:
            df = df[self.columns]
        masks = df.apply(self.cond)
        return self._join(masks)

    @abstractmethod
    def _join(self, df: pd.DataFrame) -> Iterable[bool]:
        raise NotImplementedError()


class _BoolOpsMixin:
    """Common logical operators mixin."""

    def intersection(self, other: _BoolIndexerMixin) -> "BoolOp":
        """Select elements in both selectors."""
        return BoolOp(np.logical_and, "&", self, other)  # type: ignore

    def union(self, other: _BoolIndexerMixin) -> "BoolOp":
        """Select elements in the left side but not in right side."""
        return BoolOp(np.logical_or, "|", self, other)  # type: ignore

    def symmetric_difference(self, other: Any) -> "BoolOp":
        """Select elements either in the left side or the right side but not in both."""
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


class _BoolMask(PrettyPrinter, _BoolOpsMixin):
    def __init__(self, mask: Iterable[bool]):
        self.mask = np.asarray(mask)

    def __call__(self, df: pd.DataFrame) -> Iterable[bool]:
        return self.mask


BoolOperation = Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]
BoolOperand = Union[Selector, Iterable[bool]]


class BoolOp(LogicalOp, _BoolOpsMixin):
    """A logical operation between two `Where` selectors."""

    def __init__(
        self,
        op: BoolOperation,
        op_name: str,
        left: BoolOperand,
        right: Optional[BoolOperand] = None,
    ):
        bool_selectors = [self._validate_operand(operand) for operand in (left, right)]
        super().__init__(op, op_name, *bool_selectors)  # type:ignore

    def _validate_operand(self, sel: Any) -> Union[Selector, Iterable[bool]]:
        if sel is None or callable(sel):
            return sel

        if not is_list_like(sel):
            raise TypeError("Operand does not support logical operations.")

        sel = np.asarray(sel)
        if not is_bool_dtype(sel):
            raise TypeError(f"Operand is not boolean dtype.")

        return _BoolMask(sel)


class BoolIndexer(_BoolIndexerMixin, _BoolOpsMixin, ABC):
    """Base class for boolean indexing and support logical operations."""


class Anywhere(BoolIndexer):
    """Filter rows where *any* column matches a condition.

    Parameters
    ----------
    cond: callable or boolean array-like
        Select labels where `cond` is True. If `cond` is a callable, it is computed on
        each column and should return a boolean array.

    columns: optional
        Subset of columns on which to apply `cond`.

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
        return df.any(axis="columns").to_numpy()


class Everywhere(BoolIndexer):
    """Filter rows where *all* columns match a condition.

    Parameters
    ----------
    cond: callable or boolean array-like
        Select labels where `cond` is True. If `cond` is a callable, it is computed on
        each column and should return a boolean array.

    columns: optional
        Subset of columns on which to apply `cond`.

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
        return df.all(axis="columns").to_numpy()
