Why pandas-select?
==================

``pandas-select`` is a collection of DataFrame selectors that facilitates indexing
and selecting data. The main goal is to bring the power of `tidyselect <https://tidyselect.r-lib.org/reference/language.html>`_
to pandas.

.. panels::
    :container: container pb-4
    :column: col-md-12 p-2

    Fully compatible with :class:`~pandas.DataFrame` ``[]``
    and :meth:`~pandas.DataFrame.loc` accessors.
    ---
    Emphasise readability and conciseness by cutting boilerplate:

    .. code-block:: python

       # pandas-select
       df[AllNumeric()]
       # vanilla
       df.select_dtypes("number").columns

       # pandas-select
       df[StartsWith("Type") | "Legendary"]
       # vanilla
       df.loc[:, df.columns.str.startswith("Type") | (df.columns == "Legendary")]

    ---
    Ease the challenges of
    `indexing with hierarchical index <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced-indexing-with-hierarchical-index>`_
    and offers an alternative to
    `slicers <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced-mi-slicers>`_
    when the labels cannot be listed manually.

    .. code-block:: python

       # pandas-select
       df_mi.loc[Contains("Jeff", axis="index", level="Name")]

       # vanilla
       df_mi.loc[df_mi.index.get_level_values("Name").str.contains("Jeff")]

    ---
    Allow *deferred selection* when the DataFrame's columns are not known in advance,
    for example in automated machine learning applications. ``pandas_select`` offers
    integration with `sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.compose.`make_column_selector.html>`_.

    .. code-block:: python

       from pandas_select import AnyOf, AllBool, AllNominal, AllNumeric, ColumnSelector
       from sklearn.compose import make_column_transformer
       from sklearn.preprocessing import OneHotEncoder, StandardScaler

       ct = make_column_transformer(
          (StandardScaler(), ColumnSelector(AllNumeric() & ~AnyOf("Generation"))),
          (OneHotEncoder(), ColumnSelector(AllNominal() | AllBool() | "Generation")),
       )
       ct.fit_transform(df)
