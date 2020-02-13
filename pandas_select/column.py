import pandas as pd

from ._utils import to_set
from .index import Indexer


class HasDtype(Indexer):
    def __init__(
        self, include: Optional[Dtypes] = None, exclude: Optional[Dtypes] = None
    ) -> None:
        super().__init__(axis="columns", level=None)
        self.include = include and to_set(include)
        self.exclude = exclude and to_set(exclude)

    def select(self, df: pd.DataFrame) -> pd.Index:
        df_row = df.iloc[:1]
        return df_row.select_dtypes(self.include, self.exclude).columns


class AllNumeric(HasDtype):
    def __init__(self) -> None:
        super().__init__(["number"])


class AllNominal(HasDtype):
    def __init__(self) -> None:
        dtypes = ["category", "object"]
        if pd.__version__ >= "1.0.0":
            dtypes.append("string")
        super().__init__(dtypes)
