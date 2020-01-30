from typing import List

import pandas as pd

from .index import IndexerMixin


class ColumnSelector:
    def __init__(self, selector: IndexerMixin):
        self.selector = selector

        if not hasattr(selector, "axis") or selector.axis not in [1, "columns"]:
            raise ValueError(
                f"Cannot make a ColumnSelector from {selector}, "
                "which does not target the column axis."
            )

    def __call__(self, df: pd.DataFrame) -> List[str]:
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                "ColumnSelector can only be applied to a pandas DataFrame."
            )
        cols = self.selector.select(df)
        try:
            # IndexerMixin returns a pandas.Index
            return cols.tolist()  # type: ignore
        except AttributeError:
            return list(cols)
