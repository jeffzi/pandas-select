.. _getting-started:

Getting started
===============

.. include:: ../README.rst
  :start-after: installation-start
  :end-before: installation-end

Dataset
-------

The introduction to `pandas-select` is based on the
`Pokemon with stats <https://www.kaggle.com/abcsds/pokemon>`_.
This dataset is a listing of all Pokemon species as of mid-2016, containing data about
their types and statistics.

.. ipython:: python

    import pandas as pd
    pd.set_option('display.max_rows', 5)

    df = pd.read_csv(
        "https://raw.githubusercontent.com/jeffzi/pokemonData/master/Pokemon.csv"
    )
    df = df.set_index("Name")
    df

Basics
------

.. ipython:: python

    from pandas_select import *

The simplest possible usage is:

.. tabbed:: pandas-select

    .. ipython:: python

        df[StartsWith("Type")]

.. tabbed:: vanilla

    .. ipython:: python

        df.loc[:, df.columns.str.startswith("Type")]

Index can be selected as well. :meth:`~pandas.DataFrame.loc` must be used, on top of
``axis="index"``, to tell pandas we are selecting on :attr:`~pandas.DataFrame.index`.

.. tabbed:: pandas-select

    .. ipython:: python

        df.loc[Contains("chu", axis="index")]

.. tabbed:: vanilla

    .. ipython:: python

        df.loc[df.index.str.contains("chu")]

Logical operators
-----------------

``&``, ``|``, ``^``, ``~`` operators are supported on selectors. See :ref:`api`.

.. tabbed:: pandas-select

    .. ipython:: python

        df[~AllNumeric()]  # same as df[HasDtype(exclude="number")]
        df[StartsWith("Type") | "Legendary"]

.. tabbed:: vanilla

    .. ipython:: python

        cols = df.select_dtypes(exclude="number").columns
        df[cols]
        df.loc[:, df.columns.str.startswith("Type") | (df.columns == "Legendary")]

Filters
-------

``pandas-select`` is also helpful to filter row values.

For example, let's find out which are the strongest legendary pokemons. We define strong
as being Legendary and having at least one stat above 100.

.. tabbed:: pandas-select

    .. ipython:: python

        stats = AllNumeric() & ~AnyOf("Total")
        has_strong_stat = Anywhere(lambda stat: stat > 100, columns=stats)
        df.loc[has_strong_stat & df["Legendary"]]

.. tabbed:: vanilla

    .. ipython:: python

        stats = [col for col in df.select_dtypes("number").columns if col != "Total"]
        df_stats = df[stats]
        has_strong_stat = df_stats.where(df_stats > 100).notnull().any(axis="columns")
        df.loc[has_strong_stat & df["Legendary"]]

Hierarchical indexing
---------------------

In vanilla pandas `indexing with hierarchical index <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced-indexing-with-hierarchical-index>`_
is challenging as soon as you step out of slicers or exact selection.

:ref:`label selectors <label-selection>` are compatible with :class:`~pandas.MultiIndex` out of the box.
They also have a ``level`` argument to target a specific level in the hierarchy.

First, we'll set a :class:`~pandas.MultiIndex` on the pokemon dataset:

.. ipython:: python

    df_mi = df.reset_index().set_index(["Generation", "Number", "Name"])
    df_mi


Now, we can select pokemon with a name containing "eon" and part of the 1st or 6th
 generation.

.. tabbed:: pandas-select

    .. ipython:: python

        eon_mask = Contains("eon", axis="index", level="Name")
        df_mi.loc[eon_mask]

    .. ipython:: python

        gen_mask = AnyOf([1, 6], axis="index", level="Generation")
        df_mi.loc[eon_mask & gen_mask]

.. tabbed:: vanilla

    .. ipython:: python

        eon_mask = df_mi.index.get_level_values("Name").str.contains("eon")
        df_mi.loc[eon_mask]

    .. ipython:: python

        df_mi_copy = df_mi.reset_index()
        gen_mask = df_mi_copy["Generation"].isin([1, 6])
        df_mi_copy[eon_mask & gen_mask].set_index(["Generation", "Number", "Name"])

Scikit-learn integration
------------------------

:class:`sklearn.compose.ColumnTransformer` was added to `scikit-learn <https://scikit-learn.org/stable/>`_
in version 0.20. It allows combining the outputs of multiple transformer objects used
on column subsets of the data into a single feature space.

There is also a helper :func:`sklearn.compose.make_column_selector` to map columns
based on datatype or the column names with a regex.

We clean up the dataset:

 .. ipython:: python

    from sklearn.compose import make_column_selector, make_column_transformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = df.reset_index()
    df.loc[:, StartsWith("Type")] = df.loc[:, StartsWith("Type")].fillna("")
    df

Similarly to :func:`sklearn.compose.make_column_selector`, ``pandas-select`` selectors
can be fed to :class:`sklearn.compose.ColumnTransformer` via the wrapper
:class:`~pandas_select.sklearn.ColumnSelector`.

.. tabbed:: pandas-select

    .. ipython:: python

        ct = make_column_transformer(
            (StandardScaler(), ColumnSelector(AllNumeric() & ~AnyOf("Generation"))),
            (OneHotEncoder(), ColumnSelector(AllNominal() | AllBool() | "Generation")),
        )
        ct.fit_transform(df).shape

.. tabbed:: vanilla

    .. ipython:: python

        to_encode = ["object", "bool"]
        if pd.__version__ >= "1.0.0":
            to_encode.append("string")

        ct = make_column_transformer(
            (
                StandardScaler(),
                make_column_selector(r"^(?!Generation).*$", dtype_include=["number"]),
            ),
            (OneHotEncoder(), make_column_selector(dtype_include=to_encode)),
            (OneHotEncoder(), make_column_selector("Generation")),
        )
        ct.fit_transform(df).shape

``pandas-select`` makes the intent clearer and enables more complex selection.

.. _order:

Order
-----

Selectors preserve the column order found in the DataFrame, except
for the :class:`~pandas_select.label.Exact` selector.

.. ipython:: python

    df
    df[["Type2", "Type1"]]
    df[AnyOf(["Type2", "Type1"])]

Logical operators can be used to force a particular order.

.. ipython:: python

    df[AllNumeric()]
    df["Generation" | AllNumeric()]
