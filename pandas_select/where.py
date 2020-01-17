from typing import Callable, Iterable, List, Union

import pandas as pd

from .base import Selector


Cond = Callable[[pd.Series], Iterable[bool]]


class Where(Selector):
    def __init__(self, cond: Cond, columns: Union[str, List[str]] = None):
        self.cond = cond
        self.columns = columns

    def _join(self, df: pd.DataFrame) -> Iterable[bool]:
        raise NotImplementedError()

    def select(self, df: pd.DataFrame) -> Iterable[bool]:
        if self.columns is not None:
            df = df[self.columns]
        masks = df.apply(self.cond)
        return self._join(masks)


class Anywhere(Where):
    def __init__(self, cond: Cond, columns: Union[str, List[str]] = None):
        super().__init__(cond, columns)

    def _join(self, df: pd.DataFrame) -> Iterable[bool]:
        return df.any(axis="columns").values


class Everywhere(Where):
    def __init__(self, cond: Cond, columns: Union[str, List[str]] = None):
        super().__init__(cond, columns)

    def _join(self, df: pd.DataFrame) -> Iterable[bool]:
        return df.all(axis="columns").values
