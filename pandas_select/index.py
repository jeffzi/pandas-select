from abc import abstractmethod
from typing import Any, List, Tuple, Union

import pandas as pd

from .base import Selector
from .utils import to_list


class IndexSelector(Selector):
    def __init__(self, axis: Union[int, str] = "columns", level: int = 0):
        self.axis = axis
        self.level = level

    @abstractmethod
    def select_index(self, index=pd.Index) -> List[Union[Any, Tuple]]:
        raise NotImplementedError()

    def select(self, df: pd.DataFrame) -> List[Union[Any, Tuple]]:
        index = df._get_axis(self.axis)
        level = index.get_level_values(self.level)
        return index[self.select_index(level)].tolist()

    def __call__(self, df: pd.DataFrame) -> List[str]:
        return self.select(df)


class OneOf(IndexSelector):
    def __init__(
        self, values: List[Any], axis: Union[int, str] = "columns", level: int = 0
    ):
        super().__init__(axis, level)
        self.values = to_list(values)

    def select_index(self, index=pd.Index) -> List[Union[Any, Tuple]]:
        return index.isin(self.values)
