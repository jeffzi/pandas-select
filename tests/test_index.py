import operator

import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from pandas_select.index import (
    Contains,
    EndsWith,
    Everything,
    Exact,
    Match,
    OneOf,
    StartsWith,
)

from .utils import assert_col_indexer, assert_row_indexer, pp_param


# ##############################  Fixtures  ##############################


@pytest.fixture
def df():
    """
       int  float category string
    0    1    1.0        a      a
    1   -1   -1.0        b      b
    """
    data = {
        "int": [1, -1],
        "float": [1.0, -1.0],
        "category": ["a", "b"],
        "string": ["a", "b"],
    }
    types = {"int": "int", "float": "float", "category": "category", "string": "object"}
    return pd.DataFrame(data).astype(types)


@pytest.fixture
def df_mi():
    """
    data_type      int  float category  string
    ml_type     number number  nominal nominal
    idx_s idx_i
    A     0         -1   -1.0        a       a
          1          1    1.0        b       b
    """
    index = pd.MultiIndex.from_product([["A"], [0, 1]], names=["idx_s", "idx_i"])
    columns = pd.MultiIndex.from_arrays(
        [
            ["int", "float", "category", "string"],
            ["number", "number", "nominal", "nominal"],
        ],
        names=["data_type", "ml_type"],
    )
    data = [[-1, -1.0, "a", "a"], [1, 1.0, "b", "b"]]
    types = {
        ("int", "number"): "int",
        ("float", "number"): "float",
        ("category", "nominal"): "category",
        ("string", "nominal"): "object",
    }
    return pd.DataFrame(data, index=index, columns=columns).astype(types)


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
        pp_param(1, ["nominal"], [("category", "nominal"), ("string", "nominal")],),
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


# ##############################  Logical operations  ##############################


@pytest.mark.parametrize(
    "op, left, right, expected",
    [
        # and
        pp_param(operator.and_, "int", "int", ["int"]),
        pp_param(operator.and_, "int", "float", []),
        pp_param(operator.and_, ["int", "float"], "int", ["int"]),
        pp_param(operator.and_, ["int", "float"], ["float", "int"], ["int", "float"]),
        pp_param(operator.and_, ["float", "int"], ["int", "float"], ["float", "int"]),
        # or
        pp_param(operator.or_, "int", "int", ["int"]),
        pp_param(operator.or_, "int", "float", ["int", "float"]),
        pp_param(operator.or_, ["int", "float"], "int", ["int", "float"]),
        pp_param(operator.or_, ["int", "float"], ["float", "int"], ["int", "float"]),
        pp_param(operator.or_, ["float", "int"], ["int", "float"], ["float", "int"]),
        # difference
        pp_param(operator.sub, "int", "int", []),
        pp_param(operator.sub, "int", "float", ["int"]),
        pp_param(operator.sub, ["int", "float"], "int", ["float"]),
        pp_param(operator.sub, ["int", "float"], ["float", "int"], []),
        pp_param(operator.sub, ["float", "int"], ["int", "float"], []),
        # xor
        pp_param(operator.xor, "int", "int", []),
        pp_param(operator.xor, "int", "float", ["int", "float"]),
        pp_param(operator.xor, ["int", "float"], "int", ["float"]),
        pp_param(
            operator.xor, ["int", "string"], ["float", "int"], ["string", "float"]
        ),
        pp_param(
            operator.xor, ["float", "int"], ["int", "string"], ["float", "string"]
        ),
    ],
)
def test_col_binary_op(df, op, left, right, expected):
    assert_col_indexer(df, op(Exact(left), Exact(right)), expected)
    assert_col_indexer(df, op(left, Exact(right)), expected)
    assert_col_indexer(df, op(Exact(left), right), expected)


def test_col_not_op(df):
    assert_col_indexer(df, ~Exact("int"), ["float", "category", "string"])


@pytest.mark.parametrize(
    "op, left_sel, right_sel, expected",
    [
        pp_param(
            operator.and_,
            Exact("A", axis=0, level=0),
            Exact(1, axis=0, level=1),
            [("A", 1)],
        ),
        pp_param(
            operator.or_,
            Exact(1, axis=0, level=1),
            Exact("A", axis=0, level=0),
            [("A", 1), ("A", 0)],
        ),
        pp_param(
            operator.xor,
            Exact("A", axis=0, level=0),
            Exact(1, axis=0, level=1),
            [("A", 0)],
        ),
    ],
)
def test_binary_op_mixed_levels(df_mi, op, left_sel, right_sel, expected):
    assert_row_indexer(df_mi, op(left_sel, right_sel), expected)


@pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor])
def test_incompatible_axis(df_mi, op):
    with pytest.raises(ValueError):
        op(Exact("A", axis=0), Exact("A", axis=1))
    with pytest.raises(ValueError):
        op(Exact("A", axis=1), Exact("A", axis=0))


def test_multiple_col_operators(df_mi):
    select_str_number = Exact("string", level=0) | Exact("number", level=1)
    drop_int = ~Exact("int", level=0)
    actual = (select_str_number & drop_int).select(df_mi).tolist()

    expected = [("string", "nominal"), ("float", "number")]
    assert actual == expected


def test_multiple_row_operators(df_mi):
    drop_1 = ~Exact(1, axis=0, level=1)
    select_1 = Exact("A", axis=0, level=0) ^ Exact(1, axis=0, level=1)

    assert (drop_1 | select_1).select(df_mi).tolist() == [("A", 0)]


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
        pp_param(1, ["nominal"], [("category", "nominal"), ("string", "nominal")],),
        pp_param(1, ["nominal"], [("category", "nominal"), ("string", "nominal")],),
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


# ##############################  Everything  ##############################


def test_everything(df_mi):
    assert_row_indexer(df_mi, Everything(axis=0), [("A", 0), ("A", 1)])
    assert_col_indexer(
        df_mi,
        Everything(axis=1),
        [
            ("int", "number"),
            ("float", "number"),
            ("category", "nominal"),
            ("string", "nominal"),
        ],
    )


# ##############################  Pandas str  ##############################


@pytest.fixture
def df_pattern_lower():
    return pd.DataFrame({"a": [1, 2], "b": [1, 2], "a__b_d1": [1, 2]})


@pytest.fixture
def df_pattern_upper():
    return pd.DataFrame({"A": [1, 2], "B": [1, 2], "A__b_d1": [1, 2]})


def test_startswith(df_pattern_lower):
    assert_col_indexer(df_pattern_lower, StartsWith("a"), ["a", "a__b_d1"])


def test_startswith_case(df_pattern_lower, df_pattern_upper):
    assert_col_indexer(
        df_pattern_lower, StartsWith("A", ignore_case=True), ["a", "a__b_d1"]
    )
    assert_col_indexer(
        df_pattern_upper, StartsWith("a", ignore_case=True), ["A", "A__b_d1"]
    )


def test_endswith(df_pattern_lower):
    assert_col_indexer(df_pattern_lower, EndsWith("1"), ["a__b_d1"])


def test_endswith_case(df_pattern_lower, df_pattern_upper):
    assert_col_indexer(df_pattern_lower, EndsWith("B", ignore_case=True), ["b"])
    assert_col_indexer(df_pattern_upper, EndsWith("b", ignore_case=True), ["B"])


def test_contains(df_pattern_lower):
    assert_col_indexer(df_pattern_lower, Contains("b"), ["b", "a__b_d1"])


def test_contains_case(df_pattern_lower, df_pattern_upper):
    assert_col_indexer(
        df_pattern_lower, Contains("B", ignore_case=True), ["b", "a__b_d1"]
    )
    assert_col_indexer(
        df_pattern_upper, Contains("b", ignore_case=True), ["B", "A__b_d1"]
    )


def test_match(df_pattern_lower):
    assert_col_indexer(df_pattern_lower, Match(".*_d[0-9]{1}"), ["a__b_d1"])


def test_match_case(df_pattern_lower, df_pattern_upper):
    assert_col_indexer(
        df_pattern_lower, Match(".*_D[0-9]{1}", ignore_case=True), ["a__b_d1"]
    )
    assert_col_indexer(
        df_pattern_upper, Match(".*_d[0-9]{1}", ignore_case=True), ["A__b_d1"]
    )
