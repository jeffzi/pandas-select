from abc import ABC, abstractmethod
from typing import Any, Callable, List, Sequence, Union

import numpy as np
import pandas as pd

from .base import LogicalOp, Selector


Cond = Callable[[pd.Series], Sequence[bool]]


class Where(Selector, ABC):
    def __init__(self, cond: Cond, columns: Union[str, List[str]] = None):
        self.cond = cond
        self.columns = columns

    @abstractmethod
    def _join(self, df: pd.DataFrame) -> Sequence[bool]:
        raise NotImplementedError()

    def select(self, df: pd.DataFrame) -> Sequence[bool]:
        if self.columns is not None:
            df = df[self.columns]
        masks = df.apply(self.cond)
        return self._join(masks)


class WhereOpsMixin:
    """ Common logical operators mixin """

    @staticmethod
    def _assert_can_do_logical_op(x: Any) -> None:
        if not isinstance(x, WhereOpsMixin):
            raise TypeError("Input does not support logical operations.")

    def __and__(self, other: Where) -> "WhereOp":
        self._assert_can_do_logical_op(other)
        return WhereOp(np.logical_and, "&", self, other)  # type: ignore

    def __rand__(self, other: Where) -> "WhereOp":
        self._assert_can_do_logical_op(other)
        return WhereOp(np.logical_and, "&", other, self)  # type: ignore

    def __or__(self, other: Where) -> "WhereOp":
        self._assert_can_do_logical_op(other)
        return WhereOp(np.logical_or, "|", self, other)  # type: ignore

    def __ror__(self, other: Where) -> "WhereOp":
        self._assert_can_do_logical_op(other)
        return WhereOp(np.logical_or, "|", other, self)  # type: ignore

    def __xor__(self, other: Where) -> "WhereOp":
        self._assert_can_do_logical_op(other)
        return WhereOp(np.logical_xor, "^", self, other)  # type: ignore

    def __rxor__(self, other: Where) -> "WhereOp":
        self._assert_can_do_logical_op(other)
        return WhereOp(np.logical_xor, "^", other, self)  # type: ignore

    def __invert__(self) -> "WhereOp":
        return WhereOp(np.invert, "~", self)  # type: ignore


class WhereOp(LogicalOp, WhereOpsMixin):
    pass


class Anywhere(Where, WhereOpsMixin):
    def _join(self, df: pd.DataFrame) -> np.ndarray:
        return df.any(axis="columns").values


class Everywhere(Where, WhereOpsMixin):
    def _join(self, df: pd.DataFrame) -> np.ndarray:
        return df.all(axis="columns").values
