import operator

import pytest

from pandas_select.index import Exact

from .utils import assert_col_indexer, assert_row_indexer, pp_param


@pytest.mark.parametrize(
    "op, left, right, expected",
    [
        # and
        pp_param(operator.and_, "int", "int", ["int"]),
        pp_param(operator.and_, "int", "float", []),
        pp_param(operator.and_, ["int", "float"], "int", ["int"]),
        pp_param(operator.and_, ["int", "float"], "float", ["float"]),
        pp_param(operator.and_, ["int", "float"], ["float", "int"], ["int", "float"]),
        pp_param(operator.and_, ["float", "int"], ["int", "float"], ["float", "int"]),
        # or
        pp_param(operator.or_, "int", "int", ["int"]),
        pp_param(operator.or_, "int", "float", ["int", "float"]),
        pp_param(operator.or_, ["int", "float"], "int", ["int", "float"]),
        pp_param(operator.or_, ["int", "float"], ["float", "int"], ["int", "float"]),
        pp_param(operator.or_, ["float", "int"], ["int", "float"], ["float", "int"]),
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
    assert_col_indexer(df, op(Exact(left), right), expected)
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
