import pandas as pd
import pandera as pa
import pytest

from pandas_select import SchemaSelector

from .utils import assert_col_indexer, pp_param


@pytest.fixture
def df() -> pd.DataFrame:
    """
       a  abc  b
    0  1    2  3
    """
    return pd.DataFrame(data=[[1, 2, 3]], columns=["a", "abc", "b"])


@pytest.fixture
def df_mi():
    """
    data_type    int  float category  string
    ml_type   number number  nominal nominal
    0             -1   -1.0        a       a
    1              1    1.0        b       b
    """
    return pd.DataFrame(
        data=[[-1, -1.0, "a", "a"], [1, 1.0, "b", "b"]],
        columns=pd.MultiIndex.from_arrays(
            [
                ["int", "float", "category", "string"],
                ["number", "number", "nominal", "nominal"],
            ],
            names=["data_type", "ml_type"],
        ),
    )


@pytest.mark.parametrize(
    "attrs, expected",
    [
        pp_param({"required": False}, ["b"]),
        pp_param({"nullable": False}, ["a", "abc"]),
    ],
)
def test_schema_selector(df, attrs, expected):
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(int, regex=True, nullable=False),
            "b": pa.Column(int, required=False, nullable=True),
        }
    )
    df = schema.validate(df)
    selector = SchemaSelector(**attrs)
    assert_col_indexer(df, selector, expected)


@pytest.mark.parametrize(
    "attrs, expected",
    [
        pp_param(
            {"nullable": True},
            [("int", "number"), ("float", "number"), ("string", "nominal")],
        ),
        pp_param({"nullable": True, "required": False}, [("string", "nominal")]),
    ],
)
def test_schema_selector_multi_index(df_mi, attrs, expected):
    schema = pa.DataFrameSchema(
        {
            ("int", "number"): pa.Column(int, nullable=True),
            ("float", "number"): pa.Column(float, nullable=True),
            ("category", "nominal"): pa.Column(str, required=False),
            ("string", "nominal"): pa.Column(str, required=False, nullable=True),
        }
    )
    df = schema.validate(df_mi)
    selector = SchemaSelector(**attrs)
    assert_col_indexer(df, selector, expected)


def test_no_schema(df):
    with pytest.raises(
        ValueError, match="A schema is not associated with the DataFrame."
    ):
        df[SchemaSelector()]
