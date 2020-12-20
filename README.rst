==================================================
``pandas-select``: Supercharged DataFrame indexing
==================================================

.. image:: https://github.com/jeffzi/pandas-select/workflows/tests/badge.svg
   :target: https://github.com/jeffzi/pandas-select/actions
   :alt: Github Actions status

.. image:: https://codecov.io/gh/jeffzi/pandas-select/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/jeffzi/pandas-select
   :alt: Coverage

.. image:: https://readthedocs.org/projects/project-template-python/badge/?version=latest
   :target: https://pandas-select.readthedocs.io/
   :alt: Documentation status

.. image:: https://img.shields.io/pypi/v/pandas-select.svg
   :target: https://pypi.org/project/pandas-select/
   :alt: Latest PyPI version

.. image:: https://img.shields.io/pypi/pyversions/pandas-select.svg
   :target: https://pypi.org/project/pandas-select/
   :alt: Python versions supported

.. image:: https://img.shields.io/pypi/l/pandas-select.svg
   :target: https://pypi.python.org/pypi/pandas-select/
   :alt: License

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

.. image:: https://img.shields.io/badge/style-wemake-000000.svg
   :target: https://github.com/wemake-services/wemake-python-styleguide

``pandas-select`` is a collection of DataFrame selectors that facilitates indexing
and selecting data, fully compatible with pandas vanilla indexing.

The selector functions can choose variables based on their name, data type, arbitrary
conditions, or any combination of these.

``pandas-select`` is inspired by two R libraries: `tidyselect <https://tidyselect.r-lib.org/reference/select_helpers.html>`_
and `recipe <https://tidymodels.github.io/recipes/reference/selections.html>`_.

Installation
------------

``pandas-select`` is a Python-only package `hosted on PyPI <https://pypi.org/project/pandas-select/>`_.
The recommended installation method is `pip <https://pip.pypa.io/en/stable/>`_-installing
into a `virtualenv <https://hynek.me/articles/virtualenv-lives/>`_:

.. code-block:: console

   $ pip install pandas-select


Design goals
------------

.. why-begin

* Fully compatible with `pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
  ``[]`` and `pandas.DataFrame.loc <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html?highlight=loc#pandas.DataFrame.loc>`_
  accessors.

* Emphasise readability and conciseness by cutting boilerplate:

.. code-block:: python


    df[AllNumeric()] # pandas-select
    df.select_dtypes(exclude="number").columns # vanilla

    df[StartsWith("Type") | "Legendary"] # pandas-select

     # vanilla
    df[df.select_dtypes(exclude="number").columns]
    cond = lambda col: col.startswith("Type") or col == "Legendary"
    cols = [col for col in df.columns if cond(col)]
    df[cols]

* Ease the challenges of `indexing with hierarchical index <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced-indexing-with-hierarchical-index>`_
  and offers an alternative to `slicers <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced-mi-slicers>`_
  when the labels cannot be listed manually.

.. code-block:: python

    # pandas-select
    selector = Contains("Jeff", axis="index", level="Name")
    df_mi.loc[selector]

    # vanilla
    selector = df_mi.index.get_level_values("Name").str.contains("Jeff")
    df_mi.loc[selector]

* Allow *deferred selection* when the DataFrame's columns are not known in advance,
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

Project Information
-------------------

``pandas-select`` is released under the `BS3 <https://choosealicense.com/licenses/bsd-3-clause/>`_ license,
its documentation lives at `Read the Docs <https://pandas-select.readthedocs.io/>`_,
the code on `GitHub <https://github.com/jeffzi/pandas-select>`_,
and the latest release on `PyPI <https://pypi.org/project/pandas-select/>`_.
It is tested on Python 3.6+.
