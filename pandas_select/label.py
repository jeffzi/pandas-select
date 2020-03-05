# -*- coding: utf-8 -*-

from collections import Counter
from functools import partial
from typing import Any, Callable, Iterable, Optional, Union, cast

import numpy as np
import pandas as pd

from pandas.util import Substitution

from pandas_select import iterutils
from pandas_select.base import LogicalOp, PrettyPrinter
from pandas_select.bool import Everywhere


Axis = Union[int, str]

AXIS_DOC = (
    "axis: default 'columns'\n"
    + "\tAxis along which the function is applied, {0 or 'index', 1 or 'columns'}\n"
)

LEVEL_DOC = (
    "level: optional\n"
    + "\tEither the integer position of the level or its name.\n"
    + "\tIt should only be set if ``axis`` targets a MultiIndex, "
    + "otherwise a :exc:`IndexError` will be raised.\n"
)


def _validate_axis(axis: Axis) -> Axis:
    allowed = [0, 1, "index", "columns"]
    if axis not in allowed:
        raise ValueError(f"axis must be one of {allowed}.")
    return axis


def _validate_indexer(indexer: Iterable) -> Iterable:
    """Ensure `indexer` can be used as an indexer on another index.
    https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html
    """
    if not isinstance(indexer, pd.MultiIndex) and pd.isna(indexer).any():
        # isna is not defined for MultiIndex
        return list(indexer)
    return indexer


@Substitution(axis=AXIS_DOC, level=LEVEL_DOC)
class _LabelSelectorMixin(PrettyPrinter):
    """
    Base class for selecting indexes or columns based on their labels.

    Parameters
    ----------
    %(axis)s
    %(level)s
    """

    def __init__(self, axis: Axis = "columns", level: Optional[Axis] = None):
        self.axis = _validate_axis(axis)
        self.level = level

    def __call__(self, df: pd.DataFrame) -> Iterable:
        labels = df._get_axis(self.axis)  # noqa: WPS437
        if self.level is not None:
            index = labels.get_level_values(self.level)
        else:
            index = labels

        selection: pd.Index = labels[self._get_indexer(index)]

        if selection.has_duplicates:
            raise RuntimeError(f"Found duplicated values in selection")

        return _validate_indexer(selection)

    def _get_indexer(self, index: pd.Index) -> Iterable:
        raise NotImplementedError()


def _intersection(left: pd.Index, right: pd.Index) -> pd.Index:
    if (  # noqa: WPS337
        isinstance(left, pd.MultiIndex)
        and isinstance(right, pd.MultiIndex)
        and pd.__version__ < "1.1.0"  # noqa: WPS609
    ):
        return iterutils.mi_intersection(left, right)
    return left.intersection(right, sort=False)


def _union(left: pd.Index, right: pd.Index) -> pd.Index:
    return left.union(right, sort=False)


def _difference(left: pd.Index, right: pd.Index) -> pd.Index:
    return left.difference(right, sort=False)


def _symmetric_difference(left: pd.Index, right: pd.Index) -> pd.Index:
    return left.symmetric_difference(right, sort=False)


class _LabelOpsMixin:
    """Common logical operators mixin."""

    def intersection(self, other: Any) -> "LabelOp":
        """Select elements in both selectors."""
        return LabelOp(_intersection, "&", self, other)  # type:ignore

    def union(self, other: Any) -> "LabelOp":
        """Select elements in the left side but not in the right side."""
        return LabelOp(_union, "|", self, other)  # type:ignore

    def difference(self, other: Any) -> "LabelOp":
        """Select elements in the left side but not in the right side."""
        return LabelOp(_difference, "-", self, other)  # type:ignore

    def symmetric_difference(self, other: Any) -> "LabelOp":
        """Select elements that are either in the left side or the right side
        but not in both.
        """
        return LabelOp(
            _symmetric_difference, "^", self, other  # type:ignore
        )

    def __and__(self, other: Any) -> "LabelOp":
        return self.intersection(other)

    def __rand__(self, other: Any) -> "LabelOp":
        return LabelOp(_intersection, "&", other, self)  # type:ignore

    def __or__(self, other: Any) -> "LabelOp":
        return self.union(other)

    def __ror__(self, other: Any) -> "LabelOp":
        return LabelOp(_union, "|", other, self)  # type:ignore

    def __sub__(self, other: Any) -> "LabelOp":
        return self.difference(other)

    def __rsub__(self, other: Any) -> "LabelOp":
        return LabelOp(_difference, "-", other, self)  # type:ignore

    def __xor__(self, other: Any) -> "LabelOp":
        return self.symmetric_difference(other)

    def __rxor__(self, other: Any) -> "LabelOp":
        return LabelOp(_symmetric_difference, "^", other, self)  # type:ignore

    def __invert__(self) -> "LabelOp":
        return LabelInvertOp(self)  # type:ignore


def _to_index(obj: Union[Iterable, pd.Index]) -> pd.Index:
    if isinstance(obj, pd.Index):
        return obj

    obj = iterutils.to_list(obj)
    if isinstance(obj[0], tuple):
        return pd.MultiIndex.from_tuples(obj)

    return pd.Index(obj)


class LabelOp(LogicalOp, _LabelOpsMixin):
    """A logical operation between two `:class:`_LabelSelectorMixin` selectors."""

    def __init__(
        self,
        op: Callable[[Iterable, Optional[Iterable]], Iterable],
        op_name: str,
        left: _LabelSelectorMixin,
        right: Optional[_LabelSelectorMixin] = None,
    ):
        left = self._validate_selector(left)

        if right is not None:
            right = self._validate_selector(
                right, getattr(left, "axis", "columns"), getattr(left, "level", None)
            )

            if left.axis != right.axis:
                raise ValueError(f"{left} and {right} must target the same axis.")

        super().__init__(op, op_name, left, right)
        self.axis = left.axis

    def __call__(self, df: pd.DataFrame) -> Iterable:
        lvals = _to_index(self.left(df))
        operands = [lvals]

        if self.right is not None:
            rvals = _to_index(self.right(df))
            operands.append(rvals)

        selection = self.op(*operands)
        return _validate_indexer(selection)

    def _validate_selector(
        self, obj: Any, axis: Axis = "columns", level: Optional[int] = None
    ) -> _LabelSelectorMixin:
        if callable(obj):
            return cast(_LabelSelectorMixin, obj)
        return Exact(obj, axis=axis, level=level)


class LabelSelector(_LabelSelectorMixin, _LabelOpsMixin):
    """Base class for label selection and support logical operations."""


class LabelInvertOp(LabelOp):
    """Invert operation on a :class:`_LabelSelectorMixin`."""

    def __init__(self, selector: LabelSelector):
        super().__init__(np.logical_not, "~", selector)
        self.axis = selector.axis
        self.level = selector.level

    def __call__(self, df: pd.DataFrame) -> pd.Index:
        index = df._get_axis(self.axis)  # noqa: WPS437
        if self.level is not None:
            level_index = index.get_level_values(self.level)
        else:
            level_index = index

        values = self.left(df).get_level_values(self.level)  # type: ignore
        selection = index[~level_index.isin(values)]

        return _validate_indexer(selection)


@Substitution(axis=AXIS_DOC, level=LEVEL_DOC)
class Exact(LabelSelector):
    """
    Select labels from a list,
    sorted by the order they appear in the list.

    Parameters
    ----------
    values: single label or list-like
        Index or column labels to select

    %(axis)s
    %(level)s

    Raises
    ------
    ValueError:
        If ``values`` contains duplicates.

    Examples
    --------
    >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["a", "b"])
    >>> df
       A  B
    a  1  3
    b  2  4
    >>> df[Exact(["B", "A"])] # Same as df[["B", "A"]]:
       B  A
    a  3  1
    b  4  2
    >>> df.loc[Exact("b", axis="index")] # Same as df.loc[["b"]]:
       A  B
    b  2  4
    """

    def __init__(
        self,
        values: Union[Any, Iterable],
        axis: Axis = "columns",
        level: Optional[int] = None,
    ):
        super().__init__(axis, level)
        self.values = self._validate_values(values)

    def _validate_values(self, values: Union[Any, Iterable]) -> Iterable:
        values = iterutils.to_list(values)
        counts = Counter(values)
        dups = [val for val, cnt in counts.items() if cnt > 1]
        if dups:
            raise ValueError(f"Found duplicated values")
        return values

    def _get_indexer(self, index: pd.Index) -> Union[Iterable[int], np.ndarray]:
        indexer = index.get_indexer_for(self.values)

        missing_mask = indexer == -1
        if missing_mask.any():
            missing = np.asarray(self.values)[missing_mask].tolist()
            raise KeyError(missing)

        return indexer


@Substitution(axis=AXIS_DOC, level=LEVEL_DOC)
class AnyOf(LabelSelector):
    """Select labels from a list.

    The labels are sorted by the order they appear in the :class:`~pandas.DataFrame`.

    ``AnyOf`` is similar to :meth:`pandas.Series.isin`.

    Parameters
    ----------
    values: single label or list-like
        Index or column labels to select

    %(axis)s
    %(level)s

    Notes
    -----
    ``AnyOf`` is similar to :meth:`pandas.Series.isin`.

    Examples
    --------
    >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["a", "b"])
    >>> df
       A  B
    a  1  3
    b  2  4
    >>> df[AnyOf(["B", "A", "invalid"])]
       A  B
    a  1  3
    b  2  4
    >>> df.loc[AnyOf("b", axis="index")]
       A  B
    b  2  4
    """

    def __init__(
        self, values: Any, axis: Axis = "columns", level: Optional[int] = None,
    ):
        super().__init__(axis, level)
        self.values = iterutils.to_set(values)  # noqa: WPS110

    def _get_indexer(self, index: pd.Index) -> np.ndarray:
        return index.isin(self.values)


@Substitution(axis=AXIS_DOC, level=LEVEL_DOC)
class AllOf(AnyOf):
    """Same as :class:`AnyOf`, except that a :exc:`KeyError` is raised for labels
    that don't exist.

    Parameters
    ----------
    values: single label or list-like
        Index or column labels to select

    %(axis)s
    %(level)s

    Examples
    --------
    >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["a", "b"])
    >>> df
       A  B
    a  1  3
    b  2  4
    >>> df[AllOf(["B", "A"])]
       A  B
    a  1  3
    b  2  4
    >>> df[AllOf(["B", "A", "invalid"])]
    Traceback (most recent call last):
    ...
    KeyError: {'invalid'}
    """

    def __call__(self, df: pd.DataFrame) -> pd.Index:
        selected = super().__call__(df)

        missing = self.values.difference(selected)
        if missing:
            raise KeyError(missing)

        return selected

    def __invert__(self) -> LabelInvertOp:
        return LabelInvertOp(AnyOf(self.values))


@Substitution(axis=AXIS_DOC, level=LEVEL_DOC)
class Everything(LabelSelector):
    """
    Select all labels.

    Parameters
    ----------
    %(axis)s
    %(level)s
    """

    def _get_indexer(self, index: pd.Index) -> np.ndarray:
        return np.arange(0, index.size)


MASK_MI_DOC = (
    "If applied to a :class:`~pandas.MultiIndex` with ``level=None``,"
    + " all the levels will be tested."
)

IndexMask = Callable[[pd.Index], Iterable[bool]]


@Substitution(axis=AXIS_DOC, level=LEVEL_DOC, mask_mi=MASK_MI_DOC)
class LabelMask(LabelSelector):
    """
    Select labels where the condition is True.

    Parameters
    ----------
    cond: bool Series/DataFrame, array-like, or callable
        Select labels where cond is True. If `cond` is a callable, it is computed on the
        :class:`~pandas.Index` and should return a boolean array.

    %(axis)s
    %(level)s
    kwargs:
        If ``cond`` is a :func:`callable`, keyword arguments to pass to it.

    Notes
    -----
    %(mask_mi)s

    Examples
    --------
    >>> df = pd.DataFrame({"A": [1], "B": [1]})
    >>> df
       A  B
    0  1  1
    >>> df[LabelMask([True, False])]
       A
    0  1
    >>> df[LabelMask(lambda x: x == "A")]
       A
    0  1
    """

    def __init__(
        self,
        cond: Union[Iterable[bool], IndexMask],
        axis: Axis = "columns",
        level: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(axis, level)
        self.cond = cond
        self.kwargs = kwargs

    def _get_indexer(self, index: pd.Index) -> Iterable[bool]:
        if not callable(self.cond):
            return self.cond

        func = partial(self.cond, **self.kwargs)

        if isinstance(index, pd.MultiIndex):
            mi_df = index.to_frame()
            selected = mi_df[Everywhere(func)]
            mi_values = pd.MultiIndex.from_frame(selected).to_numpy()
            return index.get_locs(mi_values)

        return func(index)


class _IgnoreCase(LabelMask):
    def __init__(  # noqa: WPS211
        self,
        cond: Callable[[np.ndarray, str], np.ndarray],
        pat: str,
        case: bool = True,
        axis: Axis = "columns",
        level: Optional[int] = None,
        **kwargs: Any,
    ):
        kwargs["pat"] = pat if case else pat.lower()
        super().__init__(cond, axis, level, **kwargs)  # type:ignore
        self.case = case

    def _get_indexer(self, index: pd.Index) -> np.ndarray:
        index = index if self.case else index.str.lower()
        return super()._get_indexer(index)


PAT_DOC = "pat:\n\tCharacter sequence. Regular expressions are not accepted."
CASE_DOC = "case: default True\n\tIf True, case sensitive."


@Substitution(
    axis=AXIS_DOC, level=LEVEL_DOC, pat=PAT_DOC, case=CASE_DOC, mask_mi=MASK_MI_DOC
)
class StartsWith(_IgnoreCase):
    """
    Select labels that start with a prefix.

    Parameters
    ----------
    %(pat)s
    %(case)s
    %(axis)s
    %(level)s

    See Also
    --------
    EndsWith: Same as `StartsWith`, but tests the end of string.

    Notes
    -----
    %(mask_mi)s

    Examples
    --------
    >>> df = pd.DataFrame({"bat": [1], "Bear": [1], "cat": [1]})
    >>> df
       bat  Bear  cat
    0    1     1    1
    >>> df[StartsWith("b")]
       bat
    0    1
    >>> df[StartsWith("b", case=False)]
       bat  Bear
    0    1     1
    """

    def __init__(
        self,
        pat: str,
        case: bool = True,
        axis: Axis = "columns",
        level: Optional[int] = None,
    ):
        super().__init__(
            pd.core.strings.str_startswith, pat, case, axis, level, na=False
        )


@Substitution(
    axis=AXIS_DOC, level=LEVEL_DOC, pat=PAT_DOC, case=CASE_DOC, mask_mi=MASK_MI_DOC
)
class EndsWith(_IgnoreCase):
    """Select labels that end with a suffix.

    Parameters
    ----------
    %(pat)s
    %(case)s
    %(axis)s
    %(level)s

    See Also
    --------
    StartsWith: Same as `EndsWith`, but tests the start of string.

    Notes
    -----
    %(mask_mi)s

    Examples
    --------
    >>> df = pd.DataFrame({"bat": [1], "Bear": [1], "caT": [1]})
    >>> df
       bat  Bear  caT
    0    1     1    1
    >>> df[EndsWith("t")]
       bat
    0    1
    >>> df[EndsWith("t", case=False)]
       bat  caT
    0    1    1
    """

    def __init__(
        self,
        pat: str,
        case: bool = True,
        axis: Axis = "columns",
        level: Optional[int] = None,
    ):
        super().__init__(pd.core.strings.str_endswith, pat, case, axis, level, na=False)


FLAGS_DOC = (
    "flags: default 0, i.e no flags\n\tFlags to pass through to the re module, "
    + "e.g. :data:`re.IGNORECASE`."
)


@Substitution(
    axis=AXIS_DOC, level=LEVEL_DOC, pat=PAT_DOC, case=CASE_DOC, flags=FLAGS_DOC
)
class Contains(LabelMask):
    """Select labels that contain a pattern or regular expression.

    Parameters
    ----------
    %(pat)s
    %(case)s
    %(flags)s
    regex : default True
        If True, assumes that ``pat`` is a regular expression.
        If False, treats that ``pat`` as a literal string.
    %(axis)s
    %(level)s

    See Also
    --------
    :meth:`pandas.Series.str.contains`: Base implementation
    Match: Analogous, but stricter, relying on :func:`re.match` instead of
        :func:`re.search`.

    Examples
    --------
    >>> df = pd.DataFrame({"Mouse": [1], "dog": [1], "house and parrot": [1]})
    >>> df
       Mouse  dog  house and parrot
    0      1    1                 1
    >>> df[Contains("og", regex=False)]
       dog
    0    1
    >>> df[Contains("house|dog")]
       dog  house and parrot
    0    1                 1
    """

    def __init__(  # noqa: WPS211
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
        axis: Axis = "columns",
        level: Optional[int] = None,
    ):
        contains_kw = {
            "pat": pat,
            "case": case,
            "flags": flags,
            "na": False,
            "regex": regex,
        }
        super().__init__(pd.core.strings.str_contains, axis, level, **contains_kw)


@Substitution(
    axis=AXIS_DOC, level=LEVEL_DOC, pat=PAT_DOC, case=CASE_DOC, flags=FLAGS_DOC
)
class Match(LabelMask):
    """
    Select labels that match a regular expression.

    Parameters
    ----------
    %(pat)s
    %(case)s
    %(flags)s
    %(axis)s
    %(level)s

    See Also
    --------
    :meth:`pandas.Series.str.match`: Base implementation
    Contains: Analogous, but less strict, relying on :func:`re.search`
        instead of :func:`re.match`.

    Examples
    --------"
    >>> df = pd.DataFrame({"Mouse": [1], "dog": [1], "house and parrot": [1]})
    >>> df
       Mouse  dog  house and parrot
    0      1    1                 1
    >>> df[Match(".*og")]
       dog
    0    1
    >>> df[Match("house|dog")]
       dog  house and parrot
    0    1                 1
    """

    def __init__(
        self,
        pat: str,
        flags: int = 0,
        axis: Axis = "columns",
        level: Optional[int] = None,
    ):
        match_kw = {"pat": pat, "flags": flags, "na": False}
        super().__init__(pd.core.strings.str_match, axis, level, **match_kw)
