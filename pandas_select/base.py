# -*- coding: utf-8 -*-

from inspect import signature
from typing import Any, Callable, Iterable, List, Optional

import pandas as pd


Selector = Callable[[pd.DataFrame], Iterable]


class PrettyPrinter:
    """Base class for all estimators in pandas-select.

    Notes
    -----
    All selectors should specify all the parameters that can be set at the class level
    in their ``__init__`` as explicit keyword arguments (no ``*args`` or ``**kwargs``).
    """

    def __repr__(self) -> str:
        params = signature(self.__class__).parameters
        args = [
            f"{param}={repr(self.__dict__[param])}"
            for param in params
            if param in self.__dict__  # param is a class attribute
        ]
        return self._format(args)

    def __str__(self) -> str:
        """Same as repr but more concise."""
        params = signature(self.__class__).parameters.items()
        args = []
        for name, param in params:
            if name not in self.__dict__:
                continue  # param is not a class attribute
            value = self.__dict__[name]
            if param.default is param.empty:
                args.append(str(value))
            elif param.default != value:
                args.append(f"{name}={str(value)}")

        return self._format(args)

    def _format(self, args: Optional[List[str]] = None) -> str:
        pretty_args = ", ".join(args) if args else ""
        return f"{type(self).__name__}({pretty_args})"


class LogicalOp:
    """Base class for logical operations in pandas-select."""

    def __init__(
        self,
        op: Callable[[Iterable, Optional[Iterable]], Iterable],
        op_name: str,
        left: Selector,
        right: Optional[Selector] = None,
    ):
        self.op = op
        self.op_name = op_name
        self.left = left
        self.right = right

    def __call__(self, df: pd.DataFrame) -> Iterable:
        operands = [self.left(df)]
        if self.right is not None:
            operands.append(self.right(df))
        return self.op(*operands)

    def __repr__(self) -> str:
        return self._pretty_format(repr)

    def __str__(self) -> str:
        return self._pretty_format(str)

    def _pretty_format(self, fmt: Callable[[Any], str]) -> str:
        pretty_left = fmt(self.left)
        if self.right is None:
            return self.op_name + pretty_left
        pretty_right = fmt(self.right)
        return f"{pretty_left} {self.op_name} {pretty_right}"
