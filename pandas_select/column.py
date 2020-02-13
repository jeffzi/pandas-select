from typing import List, Optional, Union

import pandas as pd

from ._utils import to_set
from .label import LabelSelector


Dtypes = Union[str, List[str], type, List[type]]


class HasDtype(LabelSelector):
    """
    Select columns based on the column dtypes.

    Parameters
    ----------
    include, exclude: scalar or list-like
        A selection of dtypes or strings to be included/excluded. At least one of
        these parameters must be supplied.

    Raises
    ------
    ValueError
        If both of ``include`` and ``exclude`` are empty;
        if ``include`` and ``exclude`` have overlapping elements;
        if any kind of string dtype is passed in.

    Notes
    -----
    * To select all *numeric* types, use ``numpy.number`` or ``'number'``
      or :py:class:`AllNumber()`
    * To select strings you must use the ``object`` dtype, but note that
      this will return *all* object dtype columns
    * See the `numpy dtype hierarchy
      <http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html>`__
    * To select datetimes, use :class:`numpy.datetime64`, ``'datetime'`` or
      ``'datetime64'``
    * To select timedeltas, use :class:`numpy.timedelta64`, ``'timedelta'`` or
      ``'timedelta64'``
    * To select Pandas categorical dtypes, use ``'category'``
    * To select Pandas datetimetz dtypes, use ``'datetimetz'``  or ``'datetime64[ns, tz]'``

    See Also
    --------
    AllBool, AllNumber, AllNominal

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, 2],
    ...                    "b": [True, False],
    ...                    "c": [1.0, 2.0]})
    >>> df
       a      b    c
    0  1   True  1.0
    1  2  False  2.0
    >>> df[HasDtype("int")]
       a
    0  1
    1  2
    >>> import numpy as np
    >>> df[HasDtype(include=np.number, exclude=["int"])]
         c
    0  1.0
    1  2.0
    """

    def __init__(
        self, include: Optional[Dtypes] = None, exclude: Optional[Dtypes] = None
    ) -> None:
        super().__init__(axis="columns", level=None)
        self.include = include and to_set(include)
        self.exclude = exclude and to_set(exclude)

    def select(self, df: pd.DataFrame) -> pd.Index:
        df_row = df.iloc[:1]
        return df_row.select_dtypes(self.include, self.exclude).columns


class AllNumber(HasDtype):
    """
    Select numeric columns.

    See Also
    --------
    HasDtype

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, 2],
    ...                    "b": [True, False],
    ...                    "c": [1.0, 2.0]})
    >>> df
       a      b    c
    0  1   True  1.0
    1  2  False  2.0
    >>> df[AllNumber()]
       a    c
    0  1  1.0
    1  2  2.0
    """

    def __init__(self) -> None:
        super().__init__("number")


class AllBool(HasDtype):
    """
    Select numeric columns.

    See Also
    --------
    HasDtype

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, 2],
    ...                    "b": [True, False],
    ...                    "c": [1.0, 2.0]})
    >>> df
       a      b    c
    0  1   True  1.0
    1  2  False  2.0
    >>> df[AllBool()]
           b
    0   True
    1  False
    """

    def __init__(self) -> None:
        super().__init__("bool")


class AllCat(HasDtype):
    """
    Select nominal columns (`category`, `object` or `string` if pandas version >= 1.0.0)

    Parameters
    ----------
    ordered: default `None`
       Whether to filter ordered categorical, `None` to select all categorical columns.

    See Also
    --------
    HasDtype, AllNominal

    Examples
    --------
    >>> df = pd.DataFrame({"i": [1, 2],
    ...                    "cat":  pd.Categorical(["a", "b"], ordered=False),
    ...                    "ordered_cat": pd.Categorical(["a", "b"], ordered=True)})
    >>> df
       i cat ordered_cat
    0  1   a           a
    1  2   b           b
    >>> df[AllCat()]
      cat ordered_cat
    0   a           a
    1   b           b
    >>> df[AllCat(ordered=True)]
      ordered_cat
    0           a
    1           b
    """

    def __init__(self, *, ordered: bool = None) -> None:
        super().__init__("category")
        self.ordered = ordered

    def select(self, df: pd.DataFrame) -> pd.Index:
        cols = super().select(df)
        if self.ordered is not None:
            drop_cols = [col for col in cols if df[col].cat.ordered != self.ordered]
            cols = cols.drop(drop_cols)
        return cols


def _get_str_dtypes(strict: bool) -> List[str]:
    old_pandas = pd.__version__ < "1.0.0"

    if strict:
        if old_pandas:
            raise ValueError("strict=True is incompatible with pandas < 1.0.0")
        return ["string"]

    return ["object"] if old_pandas else ["string", "object"]


class AllStr(HasDtype):
    """
    Select nominal columns (`category`, `object` or `string` if pandas version >= 1.0.0)

    Parameters
    ----------
    strict: default `False`
       If True, Dtype `object` is not considered as a string. Ignored if pandas version < 1.0.0.

    Raises
    ------
    ValueError:
        If pandas version < 1.0.0 and ``strict = True``

    Notes
    -----
    Be aware that `strict=False` will select **all** ``object`` dtype columns. Columns
    with mixed types are stored with the object dtype !

    See Also
    --------
    HasDtype, AllNominal

    Examples
    --------
    >>> df = pd.DataFrame({"i": [1, 2],
    ...                    "o": ["a", 2],
    ...                    "obj_str": ["a", "b"],
    ...                    "str": ["a", "b"]})
    >>> try:
    ...     df = df.astype({"obj_str": "object", "str": "string"})
    ... except TypeError: # pandas.__version__ < '1.0.0'
    ...     df = df.astype({"obj_str": "object", "str": "object"})
    >>> df
       i  o obj_str str
    0  1  a       a   a
    1  2  2       b   b
    >>> df[AllStr()]
       o obj_str str
    0  a       a   a
    1  2       b   b
    >>> pd.__version__ >= '1.0.0'
    True
    >>> df[AllStr(strict=True)]
      str
    0   a
    1   b
    """

    def __init__(self, *, strict: bool = False):
        super().__init__(_get_str_dtypes(strict))


class AllNominal(HasDtype):
    """
    Select nominal columns (`category`, `object` or `string` if pandas version >= 1.0.0)

    See Also
    --------
    HasDtype, AllCat, AllStr

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, 2],
    ...                    "b": ["a", "b"],
    ...                    "c": ["a", "b"]})
    >>> df = df.astype({"a": "int", "b": "object", "c": "category"})
    >>> df
       a  b  c
    0  1  a  a
    1  2  b  b
    >>> df[AllNominal()]
       b  c
    0  a  a
    1  b  b
    """

    def __init__(self, *, strict: bool = False) -> None:
        super().__init__(["category", *_get_str_dtypes(strict)])
