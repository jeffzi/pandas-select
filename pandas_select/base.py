import inspect

from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Union

import pandas as pd


IndexerValues = Union[pd.Index, Iterable[bool]]


class Selector(ABC):
    @abstractmethod
    def select(self, df: pd.DataFrame) -> IndexerValues:
        raise NotImplementedError()

    def __call__(self, df: pd.DataFrame) -> IndexerValues:
        return self.select(df)

    def _format(self, args: List[str] = None) -> str:
        pretty_cls = type(self).__name__
        pretty_args = f"({', '.join(args)})" if args else ""
        return pretty_cls + pretty_args

    def __repr__(self) -> str:
        args = [
            f"{param}={repr(vars(self)[param])}"
            for param in inspect.signature(self.__class__).parameters
            if param in vars(self)  # param is a class attribute
        ]
        return self._format(args)

    def __str__(self) -> str:
        args = []
        for param_name, param in inspect.signature(self.__class__).parameters.items():
            try:
                value = vars(self)[param_name]
                if param.default != value:
                    args.append(f"{param_name}={str(value)}")
            except KeyError:
                pass  # param is not a class attribute

        return self._format(args)


class BinarySelector(Selector, ABC):
    def __init__(
        self,
        left: Selector,
        right: Selector,
        op: Callable[[IndexerValues, IndexerValues], IndexerValues],
        op_name: str,
    ):
        self.op = op
        self.op_name = op_name
        self.left = left
        self.right = right

    def select(self, df: pd.DataFrame) -> IndexerValues:
        return self.op(self.left(df), self.right(df))

    def __repr__(self) -> str:
        return f"{self.left} {self.op_name} {self.right}"
