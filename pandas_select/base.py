from abc import ABC, abstractmethod
from inspect import signature
from typing import List, Union

import pandas as pd


class Selector(ABC):
    @abstractmethod
    def select(self, df: pd.DataFrame) -> Union[List[str], "np.ndarray[np.bool]"]:
        raise NotImplementedError()

    def __call__(self, df: pd.DataFrame) -> Union[List[str], "np.ndarray[np.bool]"]:
        return self.select(df)

    def __repr__(self):
        args = [
            f"{param}={str(vars(self)[param])}"
            for param in signature(self.__init__).parameters
        ]
        return f"{type(self).__name__}({', '.join(args)})"
