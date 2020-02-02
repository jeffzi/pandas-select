from abc import ABC
from collections import Counter
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union, cast

import numpy as np
import pandas as pd

from .base import LogicalOp, Selector
from .utils import to_list


IndexMaskValues = Union[Sequence[int], Sequence[bool], Sequence[str], Sequence[Tuple]]


class IndexerMixin(Selector, ABC):
    def __init__(self, axis: Union[int, str] = "columns", level: Optional[int] = None):
        self.axis = self._validate_axis(axis)
        self.level = level

    @staticmethod
    def _validate_axis(axis: Union[int, str]) -> Union[int, str]:
        allowed = [0, 1, "columns"]
        if axis not in allowed:
            raise ValueError(f"axis must be one of {allowed}.")
        return axis

    def _get_index_mask(self, index: pd.Index) -> IndexMaskValues:
        raise NotImplementedError()

    def select(self, df: pd.DataFrame) -> pd.Index:
        index = df._get_axis(self.axis)
        if self.level is not None:
            level = index.get_level_values(self.level)
        else:
            level = index
        selected: pd.Index = index[self._get_index_mask(level)]
        if selected.has_duplicates:
            raise RuntimeError(f"Found duplicated values in selection")
        return selected


def _logical_and_multi_index(
    left: pd.MultiIndex, right: pd.MultiIndex
) -> pd.MultiIndex:
    # https://github.com/pandas-dev/pandas/pull/31312

    if left.equals(right):
        return left

    lvals = left._ndarray_values
    rvals = right._ndarray_values

    if left.is_monotonic and right.is_monotonic:
        return left._inner_indexer(lvals, rvals)[0]

    runiq = set(rvals)
    seen: Set[Tuple] = set()
    uniques = [
        x for x in lvals if x in runiq and not (x in seen or seen.add(x))  # type:ignore
    ]

    names = left.names if left.names == right.names else None

    if len(uniques) == 0:
        return pd.MultiIndex(
            levels=left.levels,
            codes=[[]] * left.nlevels,
            names=names,
            verify_integrity=False,
        )
    return pd.MultiIndex.from_tuples(uniques, sortorder=0, names=names)


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

    def intersection(self, other: Any) -> "IndexerOp":
        return IndexerOp(_intersection, "&", self, other)  # type:ignore

    def union(self, other: Any) -> "IndexerOp":
        return IndexerOp(_union, "|", self, other)  # type:ignore

    def difference(self, other: Any) -> "IndexerOp":
        return IndexerOp(_difference, "-", self, other)  # type:ignore

    def symmetric_difference(self, other: Any) -> "IndexerOp":
        return IndexerOp(
            _symmetric_difference, "^", self, other  # type:ignore
        )

    def __and__(self, other: Any) -> "IndexerOp":
        return self.intersection(other)

    def __rand__(self, other: Any) -> "IndexerOp":
        return IndexerOp(_intersection, "&", other, self)  # type:ignore

    def __or__(self, other: Any) -> "IndexerOp":
        return self.union(other)

    def __ror__(self, other: Any) -> "IndexerOp":
        return IndexerOp(_union, "|", other, self)  # type:ignore

    def __sub__(self, other: Any) -> "IndexerOp":
        return self.difference(other)

    def __rsub__(self, other: Any) -> "IndexerOp":
        return IndexerOp(_difference, "-", other, self)  # type:ignore

    def __xor__(self, other: Any) -> "IndexerOp":
        return self.symmetric_difference(other)

    def __rxor__(self, other: Any) -> "IndexerOp":
        return IndexerOp(_symmetric_difference, "^", other, self)  # type:ignore

    def __invert__(self) -> "IndexerOp":
        return NotSelector(self)  # type:ignore


class IndexerOp(LogicalOp, IndexerOpsMixin):
    def __init__(
        self,
        op: Callable[[Sequence, Optional[Sequence]], Sequence],
        op_name: str,
        left: IndexerMixin,
        right: Optional[IndexerMixin] = None,
    ):
        left = self._validate_selector(left)
        right = self._validate_selector(
            right, getattr(left, "axis", "columns"), getattr(left, "level", None)
        )

        if left.axis != right.axis:
            raise ValueError(f"{left} and {right} must target the same axis.")

        super().__init__(op, op_name, left, right)
        self.axis = left.axis

    @staticmethod
    def _validate_selector(
        x: Any, axis: Union[int, str] = "columns", level: Optional[int] = None
    ) -> IndexerMixin:
        if not hasattr(x, "select"):
            return Exact(x, axis=axis, level=level)
        return cast(IndexerMixin, x)


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
        values: Union[Any, Sequence],
        axis: Union[int, str] = "columns",
        level: Optional[int] = None,
    ):
        super().__init__(axis, level)
        self.values = self._validate_values(values)

    @staticmethod
    def _validate_values(values: Union[Any, Sequence]) -> Sequence:
        values = to_list(values)
        dups = [x for x, cnt in Counter(values).items() if cnt > 1]
        if dups:
            raise ValueError(f"Found duplicated values")
        return values

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

class AnyOf(Indexer):

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


class AllOf(AnyOf):
    def select(self, df: pd.DataFrame) -> pd.Index:
        selected = super().select(df)

        missing = self.values.difference(selected)
        if missing:
            raise KeyError(missing)

        return selected


class Everything(Indexer):
    def __init__(self, axis: Union[int, str] = "columns"):
        super().__init__(axis, None)

    def _get_index_mask(self, index: pd.Index) -> np.ndarray:
        return np.arange(0, index.size)


class _SeriesFunc(Indexer):
    def __init__(
        self,
        func: Callable[[np.ndarray, str], np.ndarray],
        axis: Union[int, str] = "columns",
        level: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(axis, level)
        self.func = partial(func, **kwargs)

    def _get_index_mask(self, index: pd.Index) -> np.ndarray:
        if isinstance(index, pd.MultiIndex):
            mi_df = index.to_frame()
            selected = mi_df[Everywhere(self.func)]
            values = pd.MultiIndex.from_frame(selected).to_numpy()
            return index.get_locs(values)
        return self.func(index.values)


class _IgnoreCase(_SeriesFunc):
    def __init__(
        self,
        func: Callable[[np.ndarray, str], np.ndarray],
        pat: str,
        case: bool = True,
        axis: Union[int, str] = "columns",
        level: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(func, axis, level, **kwargs)
        self.pat = pat
        self.case = case

    def _get_index_mask(self, index: pd.Index) -> np.ndarray:
        if self.case:
            pat = self.pat
            cols = index.values
        else:
            pat = self.pat.lower()
            cols = index.str.lower()
        return self.func(cols, pat)


class StartsWith(_IgnoreCase):
    def __init__(
        self,
        pat: str,
        case: bool = True,
        axis: Union[int, str] = "columns",
        level: Optional[int] = None,
    ):
        super().__init__(
            pd.core.strings.str_startswith, pat, case, axis, level, na=False
        )


class EndsWith(_IgnoreCase):

    def __init__(
        self,
        pat: str,
        case: bool = True,
        axis: Union[int, str] = "columns",
        level: Optional[int] = None,
    ):
        super().__init__(pd.core.strings.str_endswith, pat, case, axis, level, na=False)


class Contains(_SeriesFunc):

    def __init__(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
        axis: Union[int, str] = "columns",
        level: Optional[int] = None,
    ):
        super().__init__(
            pd.core.strings.str_contains,
            axis,
            level,
            pat=pat,
            case=case,
            flags=flags,
            na=False,
            regex=regex,
        )


class Match(_SeriesFunc):
    def __init__(
        self,
        pat: str,
        flags: int = 0,
        axis: Union[int, str] = "columns",
        level: Optional[int] = None,
    ):
        super().__init__(
            pd.core.strings.str_match, axis, level, pat=pat, flags=flags, na=False
        )
