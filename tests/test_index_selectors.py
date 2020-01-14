import numpy as np
import pandas as pd
import pytest

from pandas.api.types import CategoricalDtype
from pandas.testing import assert_frame_equal

from pandas_select import OneOf

from .utils import assert_col_indexer
from .utils import assert_row_indexer


@pytest.fixture
def df():
    """
       int  float category ordinal string
    0    1    1.0        a       a      a
    1   -1   -1.0        b       b      b
    """
    return pd.DataFrame(
        {
            "int": [1, -1],
            "float": [1.0, -1.0],
            "category": ["a", "b"],
            "ordinal": ["a", "b"],
            "string": ["a", "b"],
        }
    ).astype(
        {
            "int": np.int64,
            "float": np.float64,
            "category": "category",
            "ordinal": CategoricalDtype(categories=["a", "b"], ordered=True),
            "string": "object",
        }
    )


@pytest.fixture
def df_mi():
    """
    data_type      int  float category
    ml_type     number number  nominal ordinal nominal
    idx_i idx_s
    1     A         -1   -1.0        a       a       a
          B          1    1.0        b       b       b
    """
    index = pd.MultiIndex.from_product([[0], ["A", "B"]], names=["idx_i", "idx_s"])
    columns = pd.MultiIndex.from_arrays(
        [
            ["int", "float", "category", "category", "string"],
            ["number", "number", "nominal", "ordinal", "nominal"],
        ],
        names=["data_type", "ml_type"],
    )
    data = [[-1, -1.0, "a", "a", "a"], [1, 1.0, "b", "b", "b"]]
    return pd.DataFrame(data, index=index, columns=columns)


@pytest.mark.parametrize(
    "cols, expected",
    [
        (["int"], ["int"]),
        (["int", "float"], ["int", "float"]),
        ("int", ["int"]),
        (["float", "int"], ["int", "float"]),  # assert preserve order
        ("invalid", []),
        (-99, []),
    ],
)
def test_one_of_col(df, cols, expected):
    assert_col_indexer(df, OneOf(cols), expected)


def test_one_of_col_duplicates(df):
    df = pd.DataFrame([[0, 0]], columns=["a", "a"])
    selector = OneOf("a")
    assert_frame_equal(df[selector], df[["a", "a"]])
    assert df.loc[:, selector].columns.tolist() == ["a", "a", "a", "a"]


def test_one_of_row_duplicates(df):
    df = pd.DataFrame([0, 0], columns=["col"], index=["a", "a"])
    selector = OneOf("a", axis=0)
    assert_frame_equal(df.loc[selector], df.loc[["a", "a"]])
    assert df.loc[selector].index.tolist() == ["a", "a", "a", "a"]


@pytest.mark.parametrize(
    "cols, expected",
    [
        ([0], [0]),
        ([0, 1], [0, 1]),
        (0, [0]),
        ([1, 0], [0, 1]),  # assert preserve order
        (-99, []),
        ("invalid", []),
    ],
)
def test_one_of_row(df, cols, expected):
    assert_row_indexer(df, OneOf(cols, axis=0), expected)


@pytest.mark.parametrize(
    "level, cols, expected",
    [
        (0, ["int"], [("int", "number")]),
        (0, ["int", "float"], [("int", "number"), ("float", "number")]),
        (0, ["float", "int"], [("int", "number"), ("float", "number")]),
        (0, [99], []),
        (
            1,
            ["ordinal", "nominal"],
            [("category", "nominal"), ("category", "ordinal"), ("string", "nominal")],
        ),
        (
            1,
            ["nominal", "ordinal"],
            [("category", "nominal"), ("category", "ordinal"), ("string", "nominal")],
        ),
        (1, [99], []),
    ],
)
def test_one_of_col_multi_index(df_mi, level, cols, expected):
    assert_col_indexer(df_mi, OneOf(cols, level=level), expected)


@pytest.mark.parametrize(
    "level, cols, expected",
    [
        (0, 0, [(0, "A"), (0, "B")]),
        (0, 99, []),
        (1, ["A", "B"], [(0, "A"), (0, "B")]),
        (1, ["B", "A"], [(0, "A"), (0, "B")]),
        (1, 99, []),
    ],
)
def test_one_of_row_multi_index(df_mi, level, cols, expected):
    assert_row_indexer(df_mi, OneOf(cols, axis=0, level=level), expected)
