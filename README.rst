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

The selector functions can choose variables based on their
`name <https://pandas-select.readthedocs.io/en/latest/reference/label_selectors.html>`_,
`data type <https://pandas-select.readthedocs.io/en/latest/reference/label_selection.html#data-type-selectors>`_,
`arbitrary conditions <https://pandas-select.readthedocs.io/en/latest/reference/api/pandas_select.label.LabelMask.htmlk>`_,
or any `combination of these <https://pandas-select.readthedocs.io/en/latest/reference/label_selection.html#logical-operators>`_.

``pandas-select`` is inspired by the excellent R library `tidyselect <https://tidyselect.r-lib.org/reference/language.html>`_.

.. installation-start

Installation
------------

``pandas-select`` is a Python-only package `hosted on PyPI <https://pypi.org/project/pandas-select/>`_.
It can be installed via `pip <https://pip.pypa.io/en/stable/>`_:

.. code-block:: console

   pip install pandas-select

.. installation-end

Design goals
------------

* Fully compatible with the
  `pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
  ``[]`` operator and the
  `pandas.DataFrame.loc <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html?highlight=loc#pandas.DataFrame.loc>`_
  accessor.

* Emphasise readability and conciseness by cutting boilerplate:

.. code-block:: python

   # pandas-select
   df[AllNumeric()]
   # vanilla
   df.select_dtypes("number").columns

   # pandas-select
   df[StartsWith("Type") | "Legendary"]
   # vanilla
   df.loc[:, df.columns.str.startswith("Type") | (df.columns == "Legendary")]

* Ease the challenges of `indexing with hierarchical index <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced-indexing-with-hierarchical-index>`_
  and offers an alternative to `slicers <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced-mi-slicers>`_
  when the labels cannot be listed manually.

.. code-block:: python

    # pandas-select
    df_mi.loc[Contains("Jeff", axis="index", level="Name")]

    # vanilla
    df_mi.loc[df_mi.index.get_level_values("Name").str.contains("Jeff")]

* Play well with machine learning applications.

   * Respect the columns :ref:`order <order>`.
   * Allow *deferred selection* when the DataFrame's columns are not known in advance,
   for example in automated machine learning applications.
   * Offer integration with :ref:`sklearn`.

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
