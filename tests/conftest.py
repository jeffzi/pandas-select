import pandas as pd
import pytest


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
