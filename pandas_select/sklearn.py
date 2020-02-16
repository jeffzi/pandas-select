# -*- coding: utf-8 -*-

from typing import List

import pandas as pd

from pandas_select.base import PrettyPrinter
from pandas_select.label import LabelSelector


class ColumnSelector(PrettyPrinter):
    """Create a callable compatible with :class:`sklearn.compose.ColumnTransformer`.

    Parameters
    ----------
    selector:
        A label selector, i.e. a :func:`callable` that returns a list of strings.

    Raises
    ------
    ValueError:
        If `selector` is not a callable or doesn't target the "columns" axis.

    Examples
    --------
    >>> from pandas_select import AnyOf, AllBool, AllNominal, AllNumeric, ColumnSelector
    >>> from sklearn.compose import make_column_transformer
    >>> from sklearn.preprocessing import OneHotEncoder, StandardScaler
    >>> make_column_transformer(
    ...    (StandardScaler(), ColumnSelector(AllNumeric() & ~AnyOf("Generation"))),
    ...    (OneHotEncoder(), ColumnSelector(AllNominal() | AllBool() | "Generation"))
    ... )
    """

    def __init__(self, selector: LabelSelector):
        self.selector = selector

        if not callable(selector):
            raise ValueError(f"{selector} is not a callable.")

        try:
            if selector.axis not in {1, "columns"}:
                raise ValueError(
                    f"Cannot make a ColumnSelector from {selector}"
                    + ", which does not target the column axis."
                )
        except AttributeError:
            pass  # noqa: WPS420

    def __call__(self, df: pd.DataFrame) -> List[str]:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("ColumnSelector can only be applied to a DataFrame.")
        cols = self.selector(df)
        try:
            # LabelSelector may return a pandas.Index
            return cols.tolist()  # type: ignore
        except AttributeError:
            return list(cols)
