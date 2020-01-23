import pandas as pd
import pytest

from pandas_select.column import AllNominal, AllNumeric, HasDtype

from .utils import assert_col_indexer, pp_param


@pytest.fixture
def df():
    """
       int  float category string
    0    1    1.0        a      a
    1   -1   -1.0        b      b
    """
    data = {
        "int": [1],
        "float": [1.0],
        "category": ["a"],
        "object": ["a"],
    }
    types = {
        "int": "int",
        "float": "float",
        "category": "category",
        "object": "object",
    }
    df = pd.DataFrame(data).astype(types)
    if pd.__version__ >= "1.0.0":
        df["string"] = pd.Series(["a"], dtype="string")
    return df


@pytest.mark.parametrize(
    "dtypes, expected",
    [
        pp_param("int", ["int"]),
        pp_param("category", ["category"]),
        pp_param("number", ["int", "float"]),
        pp_param(["object", "float"], ["float", "object"]),
    ],
)
def test_has_dtype(df, dtypes, expected):
    assert_col_indexer(df, HasDtype(dtypes), expected)


def test_all_numeric(df):
    assert_col_indexer(df, AllNumeric(), ["int", "float"])


def test_all_nominal(df):
    expected = ["category", "object"]
    if pd.__version__ >= "1.0.0":
        expected.append("string")
    assert_col_indexer(df, AllNominal(), expected)
