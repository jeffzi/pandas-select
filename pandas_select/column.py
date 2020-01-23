import pandas as pd

from .index import IndexSelector
from .utils import to_list


class HasDtype(IndexSelector):
    def __init__(self, dtypes) -> None:  # type: ignore
        super().__init__(axis="columns", level=None)
        self.dtypes = to_list(dtypes)

    def select(self, df: pd.DataFrame) -> pd.Index:
        return df.select_dtypes(self.dtypes).columns


class AllNumeric(HasDtype):
    def __init__(self) -> None:
        super().__init__(["number"])


class AllNominal(HasDtype):
    def __init__(self) -> None:
        dtypes = ["category", "object"]
        if pd.__version__ >= "1.0.0":
            dtypes.append("string")
        super().__init__(dtypes)
