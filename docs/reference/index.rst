.. _api:

=============
API reference
=============

.. toctree::
   :maxdepth: 4
   :hidden:

   label_selection
   boolean_indexing
   sklearn
   pandera

``pandas_select`` relies on Pandas' mechanism of
`selection by callable <https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#selection-by-callable>`_
to achieve `selection by labels <https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-label>`_
and `boolean indexing <https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#boolean-indexing>`_.
It defines selectors as classes implementing the magic method :meth:`~object.__call__`.

This section describes the API, if you'd like a more hands-on introduction,
have a look at :ref:`getting-started`.

Following pandas'semantic, `pandas-select` distinguishes between two types of selectors:

- Label selectors: :ref:`label-selection`.
- Boolean indexers: :ref:`boolean-indexing`
