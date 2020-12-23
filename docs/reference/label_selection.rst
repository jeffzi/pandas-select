.. _label-selection:

===================
Selection by labels
===================

Core
----

.. autosummary::
    :toctree: api/

    ~pandas_select.label.Exact
    ~pandas_select.label.Everything
    ~pandas_select.label.LabelMask

List selectors
--------------

.. autosummary::
    :toctree: api/

    ~pandas_select.label.AnyOf
    ~pandas_select.label.AllOf

String selectors
----------------

.. autosummary::
    :toctree: api/

    ~pandas_select.label.StartsWith
    ~pandas_select.label.EndsWith
    ~pandas_select.label.Contains
    ~pandas_select.label.Match

Data type selectors
-------------------

.. autosummary::
    :toctree: api/

    ~pandas_select.column.HasDtype
    ~pandas_select.column.AllBool
    ~pandas_select.column.AllNumeric
    ~pandas_select.column.AllCat
    ~pandas_select.column.AllStr
    ~pandas_select.column.AllNominal

.. _label_logical_operators:

Logical operators
-----------------

All label selectors implement the following operators:

.. list-table::
    :align: center
    :header-rows: 1
    :widths: 1 6

    * - Operator
      - Description

    * - ``~s``
      - Inverse the selection.

    * - ``s & t``
      - Select elements in both selectors.

    * - ``s | t``
      - Select elements in the left side but not in the right side.

    * - ``s ^ t``
      - Select elements in the left side but not in the right side.

    * - ``s - t``
      - Select elements in the left side but not in the right side.

For all operators, if one operand is incompatible, it will be wrapped with
:class:`~pandas_select.label.Exact` first. In that case, the ``axis`` and ``level``
arguments are inferred from the other operand.

.. ipython:: python

    from pandas_select import AnyOf

    AnyOf("A", axis="index", level=2) & "B"
    ["A", "B"] | AnyOf("B")

Duplicates
----------

Label selectors return a :class:`pandas.Index`, which is interpreted by
:class:`~pandas.DataFrame` ``[]`` and :obj:`~pandas.DataFrame.loc` as a sequence
of strings.

.. warning::

    ``pandas_select`` will raise a :exc:`RuntimeError` when the selection contains
    duplicates. This is because selecting duplicates is probably not what you want. In
    this case, Pandas gives you a :class:`~pandas.DataFrame` that contains all columns
    with that name, for each column name you selected.

    .. ipython:: python

        import pandas as pd

        df = pd.DataFrame([[2, 1], [1, 2]], columns=["A", "A"], index=["a", "a"])
        df
        df[["A", "A"]]
        df.loc[["a", "a"]]
        from pandas_select import AnyOf

        try:
            df[AnyOf("A")]
        except RuntimeError as e:
            print(e)
