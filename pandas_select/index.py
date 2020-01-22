from collections import Counter
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .base import BinarySelector, Selector
from .utils import to_list


IndexMaskValues = Union[
    Sequence[int], Sequence[bool], Sequence[str], Sequence[Tuple[Any]]
]


class IndexSelector(Selector):
    def __init__(self, axis: Union[int, str] = "columns", level: Optional[int] = None):
        self.axis = axis
        self.level = level

    def _get_index_mask(self, index: pd.Index) -> IndexMaskValues:
        raise NotImplementedError()

    def select(self, df: pd.DataFrame) -> pd.Index:
        index = df._get_axis(self.axis)
        if self.level is not None:
            level = index.get_level_values(self.level)
        else:
            level = index
        return index[self._get_index_mask(level)]

    def __and__(self, other: Any) -> Selector:
        return _make_binary_op(self, other, _logical_and, "&")

    def __rand__(self, other: Any) -> Selector:
        return _make_binary_op(other, self, _logical_and, "&")

    def __or__(self, other: Any) -> Selector:
        return _make_binary_op(self, other, _logical_or, "|")

    def __ror__(self, other: Any) -> Selector:
        return _make_binary_op(other, self, _logical_or, "|")

    def __xor__(self, other: Any) -> Selector:
        return _make_binary_op(self, other, _logical_xor, "^")

    def __rxor__(self, other: Any) -> Selector:
        return _make_binary_op(other, self, _logical_xor, "^")

    def __invert__(self) -> Selector:
        return NotSelector(self)


def _check_selector(
    x: Any, axis: Union[int, str] = "columns", level: Optional[int] = None
) -> IndexSelector:
    if not isinstance(x, IndexSelector):
        return Exact(x, axis=axis, level=level)
    return x


def _make_binary_op(
    left: IndexSelector,
    right: Any,
    op: Callable[[pd.Index, pd.Index], pd.Index],
    op_name: str,
) -> BinarySelector:
    left = _check_selector(left)
    right = _check_selector(right, left.axis, left.level)
    if left.axis != right.axis:
        raise ValueError(f"{left} and {right} must target the same axis.")
    return BinarySelector(left, right, op, op_name)


def _logical_and(left: pd.Index, right: pd.Index) -> pd.Index:
    return left.intersection(right, sort=False)


def _logical_or(left: pd.Index, right: pd.Index) -> pd.Index:
    return left.union(right, sort=False)


def _logical_xor(left: pd.Index, right: pd.Index) -> pd.Index:
    return left.symmetric_difference(right, sort=False)


class NotSelector(IndexSelector):
    def __init__(self, sel: IndexSelector):
        super().__init__(sel.axis, sel.level)
        self.sel = sel

    def _get_index_mask(self, index: pd.Index, values) -> np.ndarray:  # type: ignore
        return ~index.isin(values)

    def select(self, df: pd.DataFrame) -> pd.Index:
        values = self.sel.select(df)
        index = df._get_axis(self.axis)
        level = index.get_level_values(self.level)
        return index[self._get_index_mask(level, values)]


class Exact(IndexSelector):
    def __init__(
        self,
        values: Union[Any, List],
        axis: Union[int, str] = "columns",
        level: Optional[int] = None,
    ):
        super().__init__(axis, level)
        self.values = to_list(values)
        self._check_duplicates(self.values)

    @staticmethod
    def _check_duplicates(values: Iterable) -> None:
        dups = [x for x, cnt in Counter(values).items() if cnt > 1]
        if dups:
            raise ValueError(f"Found duplicated values: {dups}")

    def _get_index_mask_from_unique(self, index: pd.Index) -> np.ndarray:
        indexer = index.get_indexer(self.values)
        missing = np.asarray(self.values)[indexer == -1].tolist()
        if missing:
            raise KeyError(missing)
        return indexer

    def _get_index_mask(self, index: pd.Index) -> Union[List[int], np.ndarray]:
        if not index.has_duplicates:
            return self._get_index_mask_from_unique(index)

        locs = [index.get_loc(val) for val in self.values]

        if index.is_monotonic:
            # locs contains a mixture of slices and ints
            indexer = []
            for loc in locs:
                indices = loc
                if isinstance(loc, slice):
                    indices = np.arange(loc.start, loc.stop, loc.step)
                indexer.append(indices)
            locs = np.ravel(indexer)
        else:
            # locs contains a mixture of boolean arrays and ints
            masks = []
            for loc in locs:
                new_mask = loc
                if isinstance(loc, int):
                    new_mask = np.zeros(len(index), dtype=bool)
                    new_mask[loc] = True
                masks.append(new_mask)
            locs = np.logical_or.reduce(masks)

        return locs


class OneOf(IndexSelector):
    def __init__(
        self, values: List[Any], axis: Union[int, str] = "columns", level: int = 0
    ):
        super().__init__(axis, level)
        self.values = to_list(values)

    def _get_index_mask(self, index: pd.Index) -> np.ndarray:
        return index.isin(self.values)


class Everything(IndexSelector):
    def __init__(self, axis: Union[int, str] = "columns"):
        super().__init__(axis, None)

    def _get_index_mask(self, index: pd.Index) -> np.ndarray:
        return np.arange(0, index.size)
