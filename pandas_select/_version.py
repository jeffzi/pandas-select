# -*- coding: utf-8 -*-

try:
    from importlib.metadata import version  # type: ignore
except ImportError:
    from importlib_metadata import version

__version__ = version("pandas-select")
