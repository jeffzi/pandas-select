# -*- coding: utf-8 -*-

import pandas as pd
import pytest

from pandas_select.column import (
    AllBool,
    AllCat,
    AllNominal,
    AllNumber,
    AllStr,
    HasDtype,
)

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
        "ordered_category": pd.Categorical(["a"], categories=["a", "b"], ordered=True),
        "object": ["a"],
        "bool": [True],
    }
    types = {
        "int": "int",
        "float": "float",
        "category": "category",
        "object": "object",
        "bool": "bool",
    }
    df = pd.DataFrame(data).astype(types)
    if pd.__version__ >= "1.0.0":
        df["string"] = pd.Series(["a"], dtype="string")
    return df


@pytest.mark.parametrize(
    "dtypes, expected",
    [
        pp_param("int", ["int"]),
        pp_param("category", ["category", "ordered_category"]),
        pp_param("number", ["int", "float"]),
        pp_param(["object", "float"], ["float", "object"]),
        pp_param(["bool"], ["bool"]),
    ],
)
def test_has_dtype(df, dtypes, expected):
    assert_col_indexer(df, HasDtype(dtypes), expected)


def test_all_numeric(df):
    assert_col_indexer(df, AllNumber(), ["int", "float"])


def test_all_bool(df):
    assert_col_indexer(df, AllBool(), ["bool"])


def test_all_str(df):
    expected = ["object", "string"] if pd.__version__ >= "1.0.0" else ["object"]
    assert_col_indexer(df, AllStr(strict=False), expected)

    if pd.__version__ >= "1.0.0":
        assert_col_indexer(df, AllStr(strict=True), ["string"])
    else:
        with pytest.raises(ValueError):
            AllStr(strict=True)


@pytest.mark.parametrize(
    "ordered, expected",
    [
        pp_param(None, ["category", "ordered_category"]),
        pp_param(False, ["category"]),
        pp_param(True, ["ordered_category"]),
    ],
)
def test_all_cat(df, ordered, expected):
    assert_col_indexer(df, AllCat(ordered=ordered), expected)


def test_all_nominal(df):
    expected = ["category", "ordered_category", "object"]
    if pd.__version__ >= "1.0.0":
        expected.append("string")
    assert_col_indexer(df, AllNominal(strict=False), expected)

    if pd.__version__ >= "1.0.0":
        expected.remove("object")
        assert_col_indexer(df, AllNominal(strict=True), expected)
    else:
        with pytest.raises(ValueError):
            AllNominal(strict=True)
