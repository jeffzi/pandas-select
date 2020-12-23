from typing import Any, Iterable, List, Optional

import pandas as pd
from pandas.util import Substitution

from pandas_select.label import LEVEL_DOC, AnyOf, Level, Match

try:
    import pandera as pa  # noqa: WPS433
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Support for schemas requires pandera. \n"
        + "You can install pandas-select together with the schema dependencies with: \n"
        + "pip install pandas-select[schema]\n"
    ) from exc


@Substitution(level=LEVEL_DOC)
class SchemaSelector(AnyOf):
    """
    Select columns based on the column attributes of the
    :class:`~pandera.schemas.DataFrameSchema` associated with the
    :class:`~pandas.DataFrame`.

    Parameters
    ----------
    attrs: Dictionary of columns attributes to filter on.

    %(level)s

    Raises
    ------
    ValueError:
        If a :class:`~pandera.schemas.DataFrameSchema`is not associated with the
        class:`~pandas.DataFrame`.

    Notes
    -----
    A :class:`~pandera.schemas.DataFrameSchema` is automatically added to a
    :class:`~pandas.DataFrame` after calling
    :meth:`pandera.schemas.DataFrameSchema.validate`.

    Examples
    --------
    >>> df = pd.DataFrame(data=[[1, 2, 3]], columns=["a", "abc", "b"])
    >>> df
       a  abc  b
    0  1    2  3
    >>> import pandera as pa
    >>> schema = pa.DataFrameSchema({"a": pa.Column(int, regex=True, required=False)})
    >>> df = df.pandera.add_schema(schema)
    >>> df[SchemaSelector(required=False)]
       a  abc
    0  1    2
    """

    def __init__(
        self,
        level: Optional[Level] = None,
        **attrs: Any,
    ):
        super().__init__(values=None, axis="columns", level=level)
        self.attrs = attrs

    def __call__(self, df: pd.DataFrame) -> Iterable:
        schema = df.pandera.schema
        if not schema:
            raise ValueError("A schema is not associated with the DataFrame.")

        self.values = self._filter_schema(schema, df, **self.attrs)  # type: ignore
        selection = super().__call__(df)
        self.values = None  # type: ignore
        return selection

    def _filter_schema(
        self,
        schema: pa.DataFrameSchema,
        df: pd.DataFrame,
        **attrs: Any,
    ) -> List[str]:
        names: List[str] = []
        for col in schema.columns.values():
            if any(  # noqa: WPS221, WPS337
                getattr(col, attr) != value for attr, value in attrs.items()
            ):
                continue

            if getattr(col, "regex", False):
                selection = Match(col.name, axis=self.axis, level=self.level)(df)
            else:
                selection = AnyOf(col.name, axis=self.axis, level=self.level)(df)
            names.extend(selection)
        return names
