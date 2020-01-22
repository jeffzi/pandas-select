import pandas as pd
import pytest

from pandas.api.types import CategoricalDtype


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
            "int": "int",
            "float": "float",
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
        ("int", "number"): "int",
        ("float", "number"): "float",
        ("category", "nominal"): "category",
        ("category", "ordinal"): CategoricalDtype(categories=["a", "b"], ordered=True),
        ("string", "nominal"): "object",
    }
    return pd.DataFrame(data, index=index, columns=columns).astype(types)
