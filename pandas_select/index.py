from abc import ABC
from collections import Counter
from functools import partial
from typing import Any, Callable, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from .base import LogicalOp, Selector
from .utils import to_list


IndexMaskValues = Union[Sequence[int], Sequence[bool], Sequence[str], Sequence[Tuple]]


class IndexerMixin(Selector, ABC):
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


def _logical_and_multi_index(
    left: pd.MultiIndex, right: pd.MultiIndex
) -> pd.MultiIndex:
    if left.equals(right):
        return left

    result_names = left.names if left.names == right.names else None

    unique_right = set(right.values)
    seen: Set[Tuple] = set()
    unique_tuples = [
        x for x in left if x in unique_right and not (x in seen or seen.add(x))  # type: ignore
    ]

    if len(unique_tuples) == 0:
        return pd.MultiIndex(
            levels=left.levels,
            codes=[[]] * left.nlevels,
            names=result_names,
            verify_integrity=False,
        )
    return pd.MultiIndex.from_tuples(unique_tuples, sortorder=0, names=result_names)


def _intersection(left: pd.Index, right: pd.Index) -> pd.Index:
    if isinstance(left, pd.MultiIndex) and isinstance(right, pd.MultiIndex):
        # pandas.MultiIndex.intersection(..., sort=False) does not preserve order
        return _logical_and_multi_index(left, right)
    return left.intersection(right, sort=False)


def _union(left: pd.Index, right: pd.Index) -> pd.Index:
    return left.union(right, sort=False)


def _difference(left: pd.Index, right: pd.Index) -> pd.Index:
    return left.difference(right, sort=False)


def _symmetric_difference(left: pd.Index, right: pd.Index) -> pd.Index:
    return left.symmetric_difference(right, sort=False)


class IndexerOpsMixin:
    """ Common logical operators mixin """

    @staticmethod
    def _check_selector(
        x: Any, axis: Union[int, str] = "columns", level: Optional[int] = None
    ) -> IndexerMixin:
        if not isinstance(x, IndexerOpsMixin):
            return Exact(x, axis=axis, level=level)
        return x  # type:ignore

    @staticmethod
    def _make_binary_op(
        left: IndexerMixin,
        right: Any,
        op: Callable[[pd.Index, pd.Index], pd.Index],
        op_name: str,
    ) -> "IndexerOp":
        left = IndexerOpsMixin._check_selector(left)
        right = IndexerOpsMixin._check_selector(
            right, getattr(left, "axis", "columns"), getattr(left, "level", None)
        )
        if left.axis != right.axis:
            raise ValueError(f"{left} and {right} must target the same axis.")
        return IndexerOp(op, op_name, left, right, left.axis)

    def intersection(self, other: Any) -> "IndexerOp":
        return self._make_binary_op(self, other, _intersection, "&")  # type:ignore

    def union(self, other: Any) -> "IndexerOp":
        return self._make_binary_op(self, other, _union, "|")  # type:ignore

    def difference(self, other: Any) -> "IndexerOp":
        return self._make_binary_op(self, other, _difference, "-")  # type:ignore

    def symmetric_difference(self, other: Any) -> "IndexerOp":
        return self._make_binary_op(
            self, other, _symmetric_difference, "^"  # type:ignore
        )

    def __and__(self, other: Any) -> "IndexerOp":
        return self.intersection(other)

    def __rand__(self, other: Any) -> "IndexerOp":
        return self._make_binary_op(other, self, _intersection, "&")

    def __or__(self, other: Any) -> "IndexerOp":
        return self.union(other)

    def __ror__(self, other: Any) -> "IndexerOp":
        return self._make_binary_op(other, self, _union, "|")

    def __sub__(self, other: Any) -> "IndexerOp":
        return self.difference(other)

    def __rsub__(self, other: Any) -> "IndexerOp":
        return self._make_binary_op(other, self, _difference, "-")

    def __xor__(self, other: Any) -> "IndexerOp":
        return self.symmetric_difference(other)

    def __rxor__(self, other: Any) -> "IndexerOp":
        return self._make_binary_op(other, self, _symmetric_difference, "^")

    def __invert__(self) -> "IndexerOp":
        return NotSelector(self)  # type:ignore


class IndexerOp(LogicalOp, IndexerOpsMixin):
    def __init__(
        self,
        op: Callable[[Sequence, Optional[Sequence]], Sequence],
        op_name: str,
        left: "IndexerMixin",
        right: Optional["Indexer"] = None,
        axis: Union[int, str] = "columns",
    ):
        super().__init__(op, op_name, left, right)
        self.axis = axis


class Indexer(IndexerMixin, IndexerOpsMixin, ABC):
    pass


class NotSelector(IndexerOp):
    def __init__(self, selector: Indexer):
        super().__init__(np.logical_not, "~", selector)
        self.axis = selector.axis
        self.level = selector.level

    def select(self, df: pd.DataFrame) -> pd.Index:
        index = df._get_axis(self.axis)
        if self.level is not None:
            level_index = index.get_level_values(self.level)
        else:
            level_index = index
        values = self.left(df).get_level_values(self.level)  # type:ignore
        return index[~level_index.isin(values)]


class Exact(Indexer):
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
            indexer: List[int] = []
            for loc in locs:
                if isinstance(loc, slice):
                    indices = np.arange(loc.start, loc.stop, loc.step)
                    indexer.extend(indices)
                else:
                    indexer.append(loc)
            locs = np.ravel(indexer)
        else:
            # locs contains a mixture of boolean arrays and ints
            masks = []
            for loc in locs:
                if isinstance(loc, int):
                    mask = np.zeros(len(index), dtype=bool)
                    mask[loc] = True
                else:
                    mask = loc
                masks.append(mask)
            locs = np.logical_or.reduce(masks)

        return locs


class OneOf(Indexer):
    def __init__(
        self,
        values: List[Any],
        axis: Union[int, str] = "columns",
        level: Optional[int] = None,
    ):
        super().__init__(axis, level)
        self.values = to_list(values)

    def _get_index_mask(self, index: pd.Index) -> np.ndarray:
        return index.isin(self.values)


class Everything(Indexer):
    def __init__(self, axis: Union[int, str] = "columns"):
        super().__init__(axis, None)

    def _get_index_mask(self, index: pd.Index) -> np.ndarray:
        return np.arange(0, index.size)


class _PandasStr(Indexer):
    def __init__(
        self,
        func: Callable[[np.ndarray, str], np.ndarray],
        pat: str,
        axis: Union[int, str] = "columns",
        level: Optional[int] = None,
    ):
        super().__init__(axis, level)
        self.func = func
        self.pat = pat

    def _get_index_mask(self, index: pd.Index) -> np.ndarray:
        return self.func(index.values, self.pat)


class _PandasStrCase(_PandasStr):
    def __init__(
        self,
        func: Callable[[np.ndarray, str], np.ndarray],
        pat: str,
        case: bool = False,
        axis: Union[int, str] = "columns",
        level: Optional[int] = None,
    ):
        super().__init__(func, pat, axis, level)
        self.case = case

    def _get_index_mask(self, index: pd.Index) -> np.ndarray:
        if self.case:
            pat = self.pat.lower()
            cols = index.str.lower()
        else:
            pat = self.pat
            cols = index.values
        return self.func(cols, pat)


class StartsWith(_PandasStrCase):
    def __init__(
        self,
        pat: str,
        case: bool = False,
        axis: Union[int, str] = "columns",
        level: Optional[int] = None,
    ):
        super().__init__(pd.core.strings.str_startswith, pat, case, axis, level)


class EndsWith(_PandasStrCase):
    def __init__(
        self,
        pat: str,
        case: bool = False,
        axis: Union[int, str] = "columns",
        level: Optional[int] = None,
    ):
        super().__init__(pd.core.strings.str_endswith, pat, case, axis, level)


class Contains(_PandasStr):
    def __init__(
        self,
        pat: str,
        case: bool = False,
        flags: int = 0,
        na: Any = np.nan,
        regex: bool = True,
        axis: Union[int, str] = "columns",
        level: Optional[int] = None,
    ):
        func = partial(
            pd.core.strings.str_contains, case=case, flags=flags, na=na, regex=regex
        )
        super().__init__(func, pat, axis, level)


class Match(_PandasStr):
    def __init__(
        self,
        pat: str,
        flags: int = 0,
        na: Any = np.nan,
        axis: Union[int, str] = "columns",
        level: Optional[int] = None,
    ):
        func = partial(pd.core.strings.str_match, flags=flags, na=na)
        super().__init__(func, pat, axis, level)
