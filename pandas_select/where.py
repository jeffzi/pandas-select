from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .base import LogicalOp, Selector
from .utils import to_set


Cond = Callable[[pd.Series], Sequence[bool]]


class Where(Selector, ABC):
    def __init__(self, cond: Cond, columns: Optional[Union[str, List[str]]] = None):
        self.cond = cond
        self.columns = to_set(columns) if columns is not None else columns

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

    def intersection(self, other: Where) -> "WhereOp":
        return WhereOp(np.logical_and, "&", self, other)  # type: ignore

    def union(self, other: Where) -> "WhereOp":
        return WhereOp(np.logical_or, "|", self, other)  # type: ignore

    def symmetric_difference(self, other: Any) -> "WhereOp":
        return WhereOp(np.logical_xor, "^", self, other)  # type: ignore

    def __and__(self, other: Where) -> "WhereOp":
        return self.intersection(other)

    def __rand__(self, other: Where) -> "WhereOp":
        return WhereOp(np.logical_and, "&", other, self)  # type: ignore

    def __or__(self, other: Where) -> "WhereOp":
        return self.union(other)

    def __ror__(self, other: Where) -> "WhereOp":
        return WhereOp(np.logical_or, "|", other, self)  # type: ignore

    def __xor__(self, other: Where) -> "WhereOp":
        return self.symmetric_difference(other)

    def __rxor__(self, other: Where) -> "WhereOp":
        return WhereOp(np.logical_xor, "^", other, self)  # type: ignore

    def __invert__(self) -> "WhereOp":
        return WhereOp(np.invert, "~", self)  # type: ignore


class WhereOp(LogicalOp, WhereOpsMixin):
    def __init__(
        self,
        op: Callable[[Sequence, Optional[Sequence]], Sequence],
        op_name: str,
        left: Selector,
        right: Optional[Selector] = None,
    ):
        for sel in (left, right):
            if sel is not None and not isinstance(sel, WhereOpsMixin):
                raise TypeError(f"{sel} does not support logical operations.")
        super().__init__(op, op_name, left, right)


class Anywhere(Where, WhereOpsMixin):
    def _join(self, df: pd.DataFrame) -> np.ndarray:
        return df.any(axis="columns").values


class Everywhere(Where, WhereOpsMixin):
    def _join(self, df: pd.DataFrame) -> np.ndarray:
        return df.all(axis="columns").values
