import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from pandas_select.index import Exact, OneOf

from .utils import assert_col_indexer, assert_row_indexer, pp_param


# ##############################  Exact  ##############################


@pytest.mark.parametrize(
    "cols, expected",
    [
        pp_param(["int"], ["int"]),
        pp_param(["int", "float"], ["int", "float"]),
        pp_param("int", ["int"]),
    ],
)
def test_exact_col(df, cols, expected):
    assert_col_indexer(df, Exact(cols), expected)


@pytest.mark.parametrize(
    "rows, expected", [pp_param([0], [0]), pp_param([0, 1], [0, 1]), pp_param(0, [0])],
)
def test_exact_row(df, rows, expected):
    assert_row_indexer(df, Exact(rows, axis=0), expected)


def test_exact_not_found(df_mi):
    for axis in [0, 1]:
        for level in [0, 1]:
            with pytest.raises(KeyError, match="invalid"):
                df_mi[Exact("invalid", axis=axis, level=level)]


def test_exact_row_duplicates(df):
    df = pd.DataFrame([0, 0], columns=["col"], index=["a", "a"])
    selector = Exact("a", axis=0)
    assert_frame_equal(df.loc[selector], df.loc[["a", "a"]])
    assert df.loc[selector].index.tolist() == ["a", "a", "a", "a"]


@pytest.mark.parametrize(
    "level, cols, expected",
    [
        pp_param(None, [("int", "number")], [("int", "number")]),
        pp_param(
            None,
            [("int", "number"), ("float", "number")],
            [("int", "number"), ("float", "number")],
        ),
        pp_param(0, ["int"], [("int", "number")]),
        pp_param(0, ["int", "float"], [("int", "number"), ("float", "number")]),
        pp_param(
            1,
            ["ordinal", "nominal"],
            [("category", "nominal"), ("category", "ordinal"), ("string", "nominal")],
        ),
    ],
)
def test_exact_col_multi_index(df_mi, level, cols, expected):
    assert_col_indexer(df_mi, Exact(cols, level=level), expected)


@pytest.mark.parametrize(
    "level, cols, expected",
    [
        pp_param(None, ("A", 0), [("A", 0)]),
        pp_param(0, "A", [("A", 0), ("A", 1)]),
        pp_param(1, [1, 0], [("A", 1), ("A", 0)]),
    ],
)
def test_exact_row_multi_index(df_mi, level, cols, expected):
    assert_row_indexer(df_mi, Exact(cols, axis=0, level=level), expected)


def test_exact_duplicate_values():
    with pytest.raises(ValueError):
        Exact(["A", "A"])


# ##############################  OneOf  ##############################


@pytest.mark.parametrize(
    "cols, expected",
    [
        pp_param(["int"], ["int"]),
        pp_param(["int", "float"], ["int", "float"]),
        pp_param("int", ["int"]),
        pp_param(["float", "int"], ["int", "float"]),
        pp_param("invalid", []),
        pp_param(-99, []),
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
        pp_param([0], [0]),
        pp_param([0, 1], [0, 1]),
        pp_param(0, [0]),
        pp_param([1, 0], [0, 1]),
        pp_param(-99, []),
        pp_param("invalid", []),
    ],
)
def test_one_of_row(df, cols, expected):
    assert_row_indexer(df, OneOf(cols, axis=0), expected)


@pytest.mark.parametrize(
    "level, cols, expected",
    [
        pp_param(
            None,
            [("int", "number"), ("float", "number")],
            [("int", "number"), ("float", "number")],
        ),
        pp_param(
            None,
            [("float", "number"), ("int", "number")],
            [("int", "number"), ("float", "number")],
        ),
        pp_param(0, ["int"], [("int", "number")]),
        pp_param(0, ["int", "float"], [("int", "number"), ("float", "number")]),
        pp_param(0, ["float", "int"], [("int", "number"), ("float", "number")]),
        pp_param(0, [99], []),
        pp_param(
            1,
            ["ordinal", "nominal"],
            [("category", "nominal"), ("category", "ordinal"), ("string", "nominal")],
        ),
        pp_param(
            1,
            ["nominal", "ordinal"],
            [("category", "nominal"), ("category", "ordinal"), ("string", "nominal")],
        ),
        pp_param(1, [99], []),
    ],
)
def test_one_of_col_multi_index(df_mi, level, cols, expected):
    assert_col_indexer(df_mi, OneOf(cols, level=level), expected)


@pytest.mark.parametrize(
    "level, cols, expected",
    [
        pp_param(None, ("A", 0), [("A", 0)]),
        pp_param(0, "A", [("A", 0), ("A", 1)]),
        pp_param(0, 99, []),
        pp_param(1, [0, 1], [("A", 0), ("A", 1)]),
        pp_param(1, [1, 0], [("A", 0), ("A", 1)]),
        pp_param(1, 99, []),
    ],
)
def test_one_of_row_multi_index(df_mi, level, cols, expected):
    assert_row_indexer(df_mi, OneOf(cols, axis=0, level=level), expected)
