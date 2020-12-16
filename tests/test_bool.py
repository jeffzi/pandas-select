# -*- coding: utf-8 -*-

import operator

import numpy as np
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
    assert_frame_equal(where_df[selector], where_df.loc[["neg", "mixed"]])

    for cols in ["B", ["B"]]:
        selector = Anywhere(lambda x: x < 0, columns=cols)
        assert_frame_equal(where_df[selector], where_df.loc[["neg"]])


def test_anywhere_empty(where_df):
    selector = Anywhere(lambda x: x > 99)
    assert where_df[selector].empty


def test_everywhere(where_df):
    selector = Everywhere(lambda x: x > 0)
    assert_frame_equal(where_df[selector], where_df.loc[["pos"]])

    for cols in ["B", ["B"]]:
        selector = Everywhere(lambda x: x > 0, columns=cols)
        assert_frame_equal(where_df[selector], where_df.loc[["pos", "mixed"]])


def test_everywhere_empty(where_df):
    selector = Everywhere(lambda x: x > 99)
    assert where_df[selector].empty


@pytest.mark.parametrize(
    "op, expected",
    [
        (operator.and_, ["neg"]),
        (operator.or_, ["neg", "mixed"]),
        (operator.xor, ["mixed"]),
    ],
)
def test_where_binary_op(where_df, op, expected):
    left = Anywhere(lambda x: x == -1)
    right = Everywhere(lambda x: x < 0)
    assert_frame_equal(where_df.loc[expected], where_df[op(left, right)])
    assert_frame_equal(where_df.loc[expected], where_df[op(right, left)])


@pytest.mark.parametrize(
    "op, expected",
    [(operator.and_, []), (operator.or_, ["pos"]), (operator.xor, ["pos"])],
)
def test_where_binary_op_empty_operand(where_df, op, expected):
    left = Anywhere(lambda x: x > 99)
    right = Everywhere(lambda x: x > 0)
    assert_frame_equal(where_df.loc[expected], where_df[op(left, right)])
    assert_frame_equal(where_df.loc[expected], where_df[op(right, left)])


def test_where_not(where_df):
    selector = Anywhere(lambda x: x > 99)
    assert_frame_equal(where_df, where_df[~selector])
    selector = Everywhere(lambda x: x > 99)
    assert_frame_equal(where_df, where_df[~selector])


def test_where_not_empty_operand(where_df):
    selector = Anywhere(lambda x: x < 99)
    assert where_df[~selector].empty


def test_multiple_operators(where_df):
    all_pos = Everywhere(lambda x: x > 0)
    any_gt_99 = Everywhere(lambda x: x > 99)
    any_pos = Anywhere(lambda x: x == 1)

    actual = where_df[~all_pos & any_pos | any_gt_99]
    assert_frame_equal(where_df.loc[["mixed"]], actual)


@pytest.mark.parametrize(
    "op, expected",
    [
        (operator.and_, ["mixed"]),
        (operator.or_, ["pos", "mixed"]),
        (operator.xor, ["pos"]),
    ],
)
def test_where_ops_cast_array(where_df, op, expected):
    left = Anywhere(lambda x: x == 1)
    right = [False, False, True]
    assert_frame_equal(where_df.loc[expected], where_df[op(left, right)])

    right = np.asarray(right)
    assert_frame_equal(where_df.loc[expected], where_df[op(left, right)])


def test_where_invalid_ops(where_df):
    selector = Anywhere(lambda x: x > 99)

    with pytest.raises(TypeError):
        where_df["a" & selector]
    with pytest.raises(TypeError):
        "a" | selector
    with pytest.raises(TypeError):
        "a" ^ selector
