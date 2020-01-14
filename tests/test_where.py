import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from pandas_select import Anywhere, Everywhere


@pytest.fixture
def df():
    """
       A  B
    0  1  1
    1 -1 -1
    2 -1  1
    """
    return pd.DataFrame({"A": [1, -1, -1], "B": [1, -1, 1]})


def test_anywhere(df):
    selector = Anywhere(lambda x: x < 0)
    expected = df.iloc[[1, 2]].copy()
    assert_frame_equal(df[selector], expected)


def test_anywhere_empty(df):
    selector = Anywhere(lambda x: x > 99)
    assert df[selector].empty


def test_everywhere(df):
    selector = Everywhere(lambda x: x > 0)
    expected = df.iloc[[0]].copy()
    assert_frame_equal(df[selector], expected)


def test_everywhere_empty(df):
    selector = Everywhere(lambda x: x > 99)
    assert df[selector].empty
