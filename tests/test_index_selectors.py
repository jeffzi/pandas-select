import numpy as np
import pandas as pd
import pytest

from pandas.api.types import CategoricalDtype
from pandas.testing import assert_frame_equal

from pandas_select import OneOf

from .utils import assert_col_indexer, assert_row_indexer, pretty_param


# ##############################  FIXTURES  ##############################


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
            "int": np.int32,
            "float": np.float32,
            "category": "category",
            "ordinal": CategoricalDtype(categories=["a", "b"], ordered=True),
            "string": "object",
        }
    )


@pytest.fixture
def df_mi():
    """
    data_type      int  float category          string
    ml_type     number number  nominal ordinal nominal
    idx_s idx_i
    A     0         -1   -1.0        a       a       a
          1          1    1.0        b       b       b
    """
    index = pd.MultiIndex.from_product([["A"], [0, 1]], names=["idx_s", "idx_i"])
    columns = pd.MultiIndex.from_arrays(
        [
            ["int", "float", "category", "category", "string"],
            ["number", "number", "nominal", "ordinal", "nominal"],
        ],
        names=["data_type", "ml_type"],
    )
    data = [[-1, -1.0, "a", "a", "a"], [1, 1.0, "b", "b", "b"]]
    types = {
        ("int", "number"): np.int32,
        ("float", "number"): np.float32,
        ("category", "nominal"): "category",
        ("category", "ordinal"): CategoricalDtype(categories=["a", "b"], ordered=True),
        ("string", "nominal"): "object",
    }
    return pd.DataFrame(data, index=index, columns=columns).astype(types)


# ##############################  OneOf  ##############################


@pytest.mark.parametrize(
    "cols, expected",
    [
        pretty_param(["int"], ["int"]),
        pretty_param(["int", "float"], ["int", "float"]),
        pretty_param("int", ["int"]),
        pretty_param(["float", "int"], ["int", "float"]),
        pretty_param("invalid", []),
        pretty_param(-99, []),
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
        pretty_param([0], [0]),
        pretty_param([0, 1], [0, 1]),
        pretty_param(0, [0]),
        pretty_param([1, 0], [0, 1]),
        pretty_param(-99, []),
        pretty_param("invalid", []),
    ],
)
def test_one_of_row(df, cols, expected):
    assert_row_indexer(df, OneOf(cols, axis=0), expected)


@pytest.mark.parametrize(
    "level, cols, expected",
    [
        pretty_param(0, ["int"], [("int", "number")]),
        pretty_param(0, ["int", "float"], [("int", "number"), ("float", "number")]),
        pretty_param(0, ["float", "int"], [("int", "number"), ("float", "number")]),
        pretty_param(0, [99], []),
        pretty_param(
            1,
            ["ordinal", "nominal"],
            [("category", "nominal"), ("category", "ordinal"), ("string", "nominal")],
        ),
        pretty_param(
            1,
            ["nominal", "ordinal"],
            [("category", "nominal"), ("category", "ordinal"), ("string", "nominal")],
        ),
        pretty_param(1, [99], []),
    ],
)
def test_one_of_col_multi_index(df_mi, level, cols, expected):
    assert_col_indexer(df_mi, OneOf(cols, level=level), expected)


@pytest.mark.parametrize(
    "level, cols, expected",
    [
        pretty_param(0, "A", [("A", 0), ("A", 1)]),
        pretty_param(0, 99, []),
        pretty_param(1, [0, 1], [("A", 0), ("A", 1)]),
        pretty_param(1, [1, 0], [("A", 0), ("A", 1)]),
        pretty_param(1, 99, []),
    ],
)
def test_one_of_row_multi_index(df_mi, level, cols, expected):
    assert_row_indexer(df_mi, OneOf(cols, axis=0, level=level), expected)
