# -*- coding: utf-8 -*-

from typing import List

import pandas as pd

from pandas_select.label import LabelSelector


class ColumnSelector:
    """Create a callable compatible with :class:`sklearn.compose.ColumnTransformer`."""

    def __init__(self, selector: LabelSelector):
        self.selector = selector

        is_label_selector = not hasattr(selector, "axis")  # noqa: WPS421
        if is_label_selector or selector.axis not in {1, "columns"}:
            raise ValueError(
                f"Cannot make a ColumnSelector from {selector}"
                + ", which does not target the column axis."
            )

    def __call__(self, df: pd.DataFrame) -> List[str]:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("ColumnSelector can only be applied to a DataFrame.")
        cols = self.selector(df)
        try:
            # IndexerMixin returns a pandas.Index
            return cols.tolist()
        except AttributeError:
            return list(cols)
