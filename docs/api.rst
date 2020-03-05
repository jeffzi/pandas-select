.. _api:

API Reference
=============

``pandas_select`` relies on Pandas' mechanism of `selection by callable <https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#selection-by-callable>`_
to achieve `selection by labels <https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-label>`_
and `boolean indexing <https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#boolean-indexing>`_.
It defines selectors as classes implementing the magic method `__call__`.

What follows is the API explanation, if you'd like a more hands-on introduction,
have a look at `examples`.

`pandas-select` distinguishes between two types of selectors:

- Label selectors: :ref:`label-selectors`.
- Boolean indexers: :ref:`boolean-indexing`

.. _label-selectors:

Selection by labels
-------------------

Label selectors return a :py:class:`pandas.Index`, which is interpreted by
:py:class:`~pandas.DataFrame` ``[]`` and :py:obj:`~pandas.DataFrame.loc` as a sequence
of strings.

.. warning::

    If the columns or index contain duplicates, ``pandas_select`` will raise a
    :py:exc:`RuntimeError` if the selection contains one of the duplicates. This is
    because selecting duplicates is probably not what you want. In this case,
    Pandas gives you a :py:class:`~pandas.DataFrame` that contains all columns with that
    name for each column name you selected.

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

Core
^^^^

.. autosummary::
    ~pandas_select.label.Exact
    ~pandas_select.label.Everything
    ~pandas_select.label.LabelMask

List selectors
^^^^^^^^^^^^^^

.. autosummary::
    ~pandas_select.label.AnyOf
    ~pandas_select.label.AllOf

String selectors
^^^^^^^^^^^^^^^^

.. autosummary::
    ~pandas_select.label.StartsWith
    ~pandas_select.label.EndsWith
    ~pandas_select.label.Contains
    ~pandas_select.label.Match

Data type selectors
^^^^^^^^^^^^^^^^^^^

.. autosummary::
    ~pandas_select.column.HasDtype
    ~pandas_select.column.AllBool
    ~pandas_select.column.AllNumeric
    ~pandas_select.column.AllCat
    ~pandas_select.column.AllStr
    ~pandas_select.column.AllNominal

Logical operators
^^^^^^^^^^^^^^^^^

All label selectors implement the following operators:

.. list-table::
    :align: center
    :header-rows: 1
    :widths: 1 6

    * - Operator
      - Method

    * - ``s & t``
      - .. autofunction:: pandas_select.label.LabelSelector.intersection

    * - ``s | t``
      - .. autofunction:: pandas_select.label.LabelSelector.union

    * - ``s ^ t``
      - .. autofunction:: pandas_select.label.LabelSelector.symmetric_difference

    * - ``s - t``
      - .. autofunction:: pandas_select.label.LabelSelector.difference

For all operators, if one operand is incompatible, it will be wrapped with
:py:class:`~pandas_select.label.Exact` first. In that case, the ``axis`` and ``level``
arguments are inferred from the other operand.

.. ipython:: python

    from pandas_select import AnyOf

    AnyOf("A", axis="index", level=2) & "B"
    ["A", "B"] | AnyOf("B")

.. _boolean-indexing:

Boolean indexing
----------------

A common operation is the use of boolean vectors to filter the data.

Filters
^^^^^^^

.. autosummary::
    ~pandas_select.bool.Anywhere
    ~pandas_select.label.Everywhere

Logical operators
^^^^^^^^^^^^^^^^^

All boolean indexers implement the following operators:

.. list-table::
    :align: center
    :header-rows: 1
    :widths: 1 6

    * - Operator
      - Method

    * - ``s & t``
      - .. autofunction:: pandas_select.bool.BoolIndexer.intersection

    * - ``s | t``
      - .. autofunction:: pandas_select.bool.BoolIndexer.union

    * - ``s ^ t``
      - .. autofunction:: pandas_select.bool.BoolIndexer.symmetric_difference

One operand is allowed to be a `boolean array-like`:

.. ipython:: python

    from pandas_select import Anywhere, Everywhere
    def is_even(x):
        return (x % 2) == 0
    Anywhere(is_even) & [True, False]
    [False, False] | Everywhere(is_even)

Scikit-learn integration
------------------------

.. autosummary::
    ~pandas_select.sklearn.ColumnSelector

Definitions
-----------

.. autoclass:: pandas_select.label.Exact

.. autoclass:: pandas_select.label.Everything
.. autoclass:: pandas_select.label.LabelMask

.. autoclass:: pandas_select.label.AnyOf
.. autoclass:: pandas_select.label.AllOf

.. autoclass:: pandas_select.label.StartsWith
.. autoclass:: pandas_select.label.EndsWith
.. autoclass:: pandas_select.label.Contains
.. autoclass:: pandas_select.label.Match

.. autoclass:: pandas_select.column.HasDtype
.. autoclass:: pandas_select.column.AllBool
.. autoclass:: pandas_select.column.AllNumeric
.. autoclass:: pandas_select.column.AllCat
.. autoclass:: pandas_select.column.AllStr
.. autoclass:: pandas_select.column.AllNominal

.. autoclass:: pandas_select.bool.Anywhere
.. autoclass:: pandas_select.label.Everywhere

.. autoclass:: pandas_select.sklearn.ColumnSelector
