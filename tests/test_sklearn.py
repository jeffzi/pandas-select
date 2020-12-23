# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pandas_select import AllNumeric, AnyOf, ColumnSelector, StartsWith


def test_column_selector():
    X = pd.DataFrame(
        {
            "country": ["GB", "GB", "FR", "US"],
            "city": ["London", "London", "Paris", "Sallisaw"],
            "int": [5, 3, 4, 5],
        }
    )
    ct = make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include=np.number)),
        (OneHotEncoder(), make_column_selector("city")),
    )
    expected = ct.fit_transform(X)

    ct = make_column_transformer(
        (StandardScaler(), ColumnSelector(AllNumeric())),
        (OneHotEncoder(), ColumnSelector(StartsWith("c") & ~AnyOf("country"))),
    )
    actual = ct.fit_transform(X)

    assert_array_equal(actual, expected)


def test_invalid_column_selector():
    for selector in ["a", StartsWith("a", axis=0)]:
        with pytest.raises(ValueError):
            ColumnSelector(selector)


def test_invalid_input():
    with pytest.raises(ValueError):
        ColumnSelector(lambda x: ["A"])([[1, 2]])


def test_lambda_selector():
    df = pd.DataFrame([[0, 1]], columns=["A", "B"])
    assert ColumnSelector(lambda x: ["A"])(df) == ["A"]
