from abc import abstractmethod
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .base import Selector
from .utils import to_list


IndexMaskValues = Union[
    Sequence[int], Sequence[bool], Sequence[str], Sequence[Tuple[str]]
]


class IndexSelector(Selector):
    def __init__(self, axis: Union[int, str] = "columns", level: Optional[int] = None):
        self.axis = axis
        self.level = level

    @abstractmethod
    def _get_index_mask(self, index: pd.Index) -> IndexMaskValues:
        raise NotImplementedError()

    def select(self, df: pd.DataFrame) -> pd.Index:
        index = df._get_axis(self.axis)
        if self.level is not None:
            level = index.get_level_values(self.level)
        else:
            level = index
        return index[self._get_index_mask(level)]


class Exact(IndexSelector):
    def __init__(
        self,
        values: Union[str, List[str]],
        axis: Union[int, str] = "columns",
        level: int = 0,
    ):
        super().__init__(axis, level)
        self.values = to_list(values)

    def _get_index_mask_from_unique(self, index: pd.Index) -> np.ndarray:
        indexer = index.get_indexer(self.values)
        missing = np.asarray(self.values)[indexer == -1].tolist()
        if missing:
            raise KeyError(missing)
        return indexer

    def _get_index_mask(self, index: pd.Index) -> Union[List[int], np.ndarray]:
        if not index.has_duplicates:
            return self._get_index_mask_from_unique(index)

        locs = [index.get_loc(val) for val in self.values]

        if index.is_monotonic:
            # locs contains a mixture of slices and ints
            indexer = []
            for loc in locs:
                indices = loc
                if isinstance(loc, slice):
                    indices = np.arange(loc.start, loc.stop, loc.step)
                indexer.append(indices)
            locs = np.ravel(indexer)
        else:
            # locs contains a mixture of boolean arrays and ints
            masks = []
            for loc in locs:
                new_mask = loc
                if isinstance(loc, int):
                    new_mask = np.zeros(len(index), dtype=bool)
                    new_mask[loc] = True
                masks.append(new_mask)
            locs = np.logical_or.reduce(masks)

        return locs


class OneOf(IndexSelector):
    def __init__(
        self, values: List[Any], axis: Union[int, str] = "columns", level: int = 0
    ):
        super().__init__(axis, level)
        self.values = to_list(values)

    def _get_index_mask(self, index: pd.Index) -> np.ndarray:
        return index.isin(self.values)
