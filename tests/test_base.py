import pytest

from pandas_select.base import LogicalOp, Selector

from .utils import pp_param


class DummySelector(Selector):
    def __init__(self, b, a=1, optional=None, ignored=None):
        self.a = a
        self.b = b
        self.optional = optional

    def select(self, df):
        return df.columns.tolist()


@pytest.mark.parametrize(
    "func, kw, expected",
    [
        pp_param(repr, {"b": 0}, "DummySelector(b=0, a=1, optional=None)"),
        pp_param(str, {"b": 0}, "DummySelector(0)"),
        pp_param(str, {"a": 1, "b": 0}, "DummySelector(0)"),
        pp_param(str, {"a": 2, "b": 0}, "DummySelector(0, a=2)"),
    ],
)
def test_selector_fmt(func, kw, expected):
    actual = func(DummySelector(**kw))
    print(f"actual: {actual}")
    assert actual == expected


@pytest.mark.parametrize(
    "func, expected",
    [
        (
            repr,
            "DummySelector(b='left', a=1, optional=None)"
            + " >_< "
            + "DummySelector(b='right', a=1, optional=None)",
        ),
        (str, "DummySelector(left) >_< DummySelector(right)"),
    ],
)
def test_logical_op_str(func, expected):
    op = LogicalOp(lambda x, y: x, ">_<", DummySelector("left"), DummySelector("right"))
    actual = func(op)
    print(f"actual: {actual}")
    assert func(op) == expected
