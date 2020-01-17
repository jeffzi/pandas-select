import inspect

from abc import ABC, abstractmethod
from typing import Iterable, Union

import pandas as pd


IndexerValues = Union[pd.Index, Iterable[bool]]


class Selector(ABC):
    @abstractmethod
    def select(self, df: pd.DataFrame) -> IndexerValues:
        raise NotImplementedError()

    def __call__(self, df: pd.DataFrame) -> IndexerValues:
        return self.select(df)

    def __repr__(self):
        args = [
            f"{param}={repr(vars(self)[param])}"
            for param in inspect.signature(self.__init__).parameters
            if param in vars(self)  # param is a class attribute
        ]
        return f"{type(self).__name__}({', '.join(args)})"
