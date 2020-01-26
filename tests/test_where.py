import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from pandas_select import Anywhere, Everywhere


@pytest.fixture
def where_df():
    """
           A  B
    pos    1  1
    neg   -1 -1
    mixed -1  1
    """
    return pd.DataFrame(
        {"A": [1, -1, -1], "B": [1, -1, 1]}, index=["pos", "neg", "mixed"]
    )


def test_anywhere(where_df):
    selector = Anywhere(lambda x: x < 0)
    expected = where_df.loc[["neg", "mixed"]].copy()
    assert_frame_equal(where_df[selector], expected)


def test_anywhere_empty(where_df):
    selector = Anywhere(lambda x: x > 99)
    assert where_df[selector].empty


def test_everywhere(where_df):
    selector = Everywhere(lambda x: x > 0)
    expected = where_df.loc[["pos"]].copy()
    assert_frame_equal(where_df[selector], expected)


def test_everywhere_empty(where_df):
    selector = Everywhere(lambda x: x > 99)
    assert where_df[selector].empty


def test_where_and(where_df):
    a = Anywhere(lambda x: x == -1)
    b = Everywhere(lambda x: x < 0)
    assert_frame_equal(where_df.loc[["neg"]], where_df[a & b])
    assert_frame_equal(where_df.loc[["neg"]], where_df[b & a])


def test_where_or(where_df):
    a = Anywhere(lambda x: x == -1)
    b = Everywhere(lambda x: x < 0)
    assert_frame_equal(where_df.loc[["neg", "mixed"]], where_df[a | b])
    assert_frame_equal(where_df.loc[["neg", "mixed"]], where_df[b | a])


def test_where_xor(where_df):
    a = Anywhere(lambda x: x == -1)
    b = Everywhere(lambda x: x < 0)
    assert_frame_equal(where_df.loc[["mixed"]], where_df[a ^ b])
    assert_frame_equal(where_df.loc[["mixed"]], where_df[b ^ a])


def test_where_not(where_df):
    selector = Anywhere(lambda x: x > 99)
    assert_frame_equal(where_df, where_df[~selector])
    selector = Everywhere(lambda x: x > 99)
    assert_frame_equal(where_df, where_df[~selector])


def test_where_and_empty(where_df):
    a = Anywhere(lambda x: x > 99)
    b = Everywhere(lambda x: x > 0)
    assert where_df[a & b].empty
    assert where_df[b & a].empty


def test_where_or_empty(where_df):
    a = Anywhere(lambda x: x > 99)
    b = Everywhere(lambda x: x > 0)
    assert_frame_equal(where_df.loc[["pos"]], where_df[a | b])
    assert_frame_equal(where_df.loc[["pos"]], where_df[b | a])


def test_where_xor_empty(where_df):
    a = Anywhere(lambda x: x > 99)
    b = Everywhere(lambda x: x > 0)
    assert_frame_equal(where_df.loc[["pos"]], where_df[a ^ b])
    assert_frame_equal(where_df.loc[["pos"]], where_df[b ^ a])


def test_where_not_empty(where_df):
    selector = Anywhere(lambda x: x < 99)
    assert where_df[~selector].empty


def test_multiple_operators(where_df):
    all_pos = Everywhere(lambda x: x > 0)
    any_gt_99 = Everywhere(lambda x: x > 99)
    any_pos = Anywhere(lambda x: x == 1)

    actual = where_df[~all_pos & any_pos | any_gt_99]
    assert_frame_equal(where_df.loc[["mixed"]], actual)
