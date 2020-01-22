import pytest

from pandas_select.column import AllNominal, AllNumeric, HasDtype

from .utils import assert_col_indexer, pp_param


@pytest.mark.parametrize(
    "dtypes, expected",
    [
        pp_param("int", ["int"]),
        pp_param("category", ["category"]),
        pp_param("number", ["int", "float"]),
        pp_param(["object", "float"], ["float", "string"]),
    ],
)
def test_has_dtype(df, dtypes, expected):
    assert_col_indexer(df, HasDtype(dtypes), expected)


def test_all_numeric(df):
    assert_col_indexer(df, AllNumeric(), ["int", "float"])


def test_all_nominal(df):
    assert_col_indexer(df, AllNominal(), ["category", "string"])
