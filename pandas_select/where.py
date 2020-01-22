from abc import ABC, abstractmethod
from typing import Any, Callable, List, Sequence, Union

import numpy as np
import pandas as pd

from .base import BinarySelector, Selector


Cond = Callable[[pd.Series], Sequence[bool]]


def _assert_can_do_logical_op(other: Any) -> None:
    if not isinstance(other, Where):
        raise TypeError("Input must be a 'Where' selector.")


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

    def __and__(self, other: "Where") -> Selector:
        _assert_can_do_logical_op(other)
        return BinarySelector(self, other, np.logical_and, "&")

    def __rand__(self, other: "Where") -> Selector:
        _assert_can_do_logical_op(other)
        return BinarySelector(other, self, np.logical_and, "&")

    def __or__(self, other: "Where") -> Selector:
        _assert_can_do_logical_op(other)
        return BinarySelector(self, other, np.logical_or, "&")

    def __ror__(self, other: "Where") -> Selector:
        _assert_can_do_logical_op(other)
        return BinarySelector(other, self, np.logical_or, "&")

    def __xor__(self, other: "Where") -> Selector:
        _assert_can_do_logical_op(other)
        return BinarySelector(self, other, np.logical_xor, "^")

    def __rxor__(self, other: "Where") -> Selector:
        _assert_can_do_logical_op(other)
        return BinarySelector(other, self, np.logical_xor, "^")

    def __invert__(self) -> Selector:
        return WhereNot(self)


class WhereNot(Selector):
    def __init__(self, selector: Where):
        self.selector = selector

    def select(self, df: pd.DataFrame) -> np.ndarray:
        return np.invert(self.selector(df))

    def __repr__(self) -> str:
        return f"~{self.selector}"


class Anywhere(Where):
    def __init__(self, cond: Cond, columns: Union[str, List[str]] = None):
        super().__init__(cond, columns)

    def _join(self, df: pd.DataFrame) -> np.ndarray:
        return df.any(axis="columns").values


class Everywhere(Where):
    def __init__(self, cond: Cond, columns: Union[str, List[str]] = None):
        super().__init__(cond, columns)

    def _join(self, df: pd.DataFrame) -> np.ndarray:
        return df.all(axis="columns").values
