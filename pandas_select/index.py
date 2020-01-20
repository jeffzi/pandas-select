from abc import abstractmethod
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .base import Selector
from .utils import to_list


IndexMaskValues = Union[
    Sequence[int], Sequence[bool], Sequence[str], Sequence[Tuple[str]]
]


class IndexSelector(Selector):
    def __init__(self, axis: Union[int, str] = "columns", level: int = 0):
        self.axis = axis
        self.level = level

    @abstractmethod
    def get_index_mask(self, index: pd.Index) -> IndexMaskValues:
        raise NotImplementedError()

    def select(self, df: pd.DataFrame) -> pd.Index:
        index = df._get_axis(self.axis)
        level = index.get_level_values(self.level)
        return index[self.get_index_mask(level)]


class OneOf(IndexSelector):
    def __init__(
        self, values: List[Any], axis: Union[int, str] = "columns", level: int = 0
    ):
        super().__init__(axis, level)
        self.values = to_list(values)

    def get_index_mask(self, index: pd.Index) -> np.ndarray:
        return index.isin(self.values)
