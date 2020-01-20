from typing import Any, List

import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from pandas_select.base import Selector


def assert_col_indexer(df: pd.DataFrame, selector: Selector, expected: List[Any]):
    assert selector.select(df).tolist() == expected
    assert df.loc[:, selector].columns.tolist() == expected
    assert_frame_equal(df[selector], df[expected])


def assert_row_indexer(df: pd.DataFrame, selector: Selector, expected: List[Any]):
    assert selector.select(df).tolist() == expected
    assert df.loc[selector, :].index.tolist() == expected
    assert_frame_equal(df.loc[selector], df.loc[expected])


def pretty_param(*values, **kw):
    id = "-".join(map(str, values))
    return pytest.param(*values, id=id, **kw)
