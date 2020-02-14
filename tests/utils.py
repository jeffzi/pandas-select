# -*- coding: utf-8 -*-

import pytest

from pandas.testing import assert_frame_equal


def assert_col_indexer(df, selector, expected):
    print(f"{selector} selected:\n\t{selector(df)}")

    assert list(selector(df)) == expected
    assert df.loc[:, selector].columns.tolist() == expected
    assert_frame_equal(df[selector], df[expected])


def assert_row_indexer(df, selector, expected):
    print(f"{selector} selected:\n\t{selector(df)}")

    assert list(selector(df)) == expected
    assert df.loc[selector, :].index.tolist() == expected
    assert_frame_equal(df.loc[selector], df.loc[expected])


def pp_param(*values, **kw):
    id = kw.pop("id", "-".join(map(str, values)))
    return pytest.param(*values, id=id, **kw)
