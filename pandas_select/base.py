from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Callable, List, Optional, Sequence

import pandas as pd


class Selector(ABC):
    @abstractmethod
    def select(self, df: pd.DataFrame) -> Sequence:
        raise NotImplementedError()

    def __call__(self, df: pd.DataFrame) -> Sequence:
        return self.select(df)

    def _format(self, args: Optional[List[str]] = None) -> str:
        pretty_args = ", ".join(args) if args else ""
        return f"{ type(self).__name__}({pretty_args})"

    def __repr__(self) -> str:
        args = [
            f"{param}={repr(vars(self)[param])}"
            for param in signature(self.__class__).parameters
            if param in vars(self)  # param is a class attribute
        ]
        return self._format(args)

    def __str__(self) -> str:
        """ Same as generated repr but ignore attributes set to default """
        args = []
        for param_name, param in signature(self.__class__).parameters.items():
            try:
                value = vars(self)[param_name]
                if param.default is param.empty:
                    args.append(str(value))
                elif param.default != value:
                    args.append(f"{param_name}={str(value)}")
            except KeyError:
                pass  # param is not a class attribute

        return self._format(args)


class LogicalOp(Selector, ABC):
    def __init__(
        self,
        op: Callable[[Sequence, Optional[Sequence]], Sequence],
        op_name: str,
        left: Selector,
        right: Optional[Selector] = None,
    ):
        self.op = op
        self.op_name = op_name
        self.left = left
        self.right = right

    def select(self, df: pd.DataFrame) -> Sequence:
        args = [self.left(df)]
        if self.right is not None:
            args.append(self.right(df))
        return self.op(*args)

    def _pretty_format(self, fmt: Callable[[Any], str]) -> str:
        if self.right is None:
            return f"{fmt(self.op_name)}{fmt(self.left)}"
        return f"{fmt(self.left)} {self.op_name} {fmt(self.right)}"

    def __repr__(self) -> str:
        return self._pretty_format(repr)

    def __str__(self) -> str:
        return self._pretty_format(str)
