.. _sources:

================================
Data Sources (``sdoml.sources.DataSource``)
================================

.. toctree::
   :maxdepth: 1

One of the core classes in ``sdoml`` is a `DataSource`. An ``sdoml`` DataSource
object provides information on the location and storage of the data.
These DataSource objects are subclasses of the :py:meth:`~sdoml.sources.data_base.GenericDataSource`
and are created using the DataSource factory, :py:meth:`~sdoml.sources.DataSource`.

``sdoml`` currently supports a small number data sources, see :ref:`available-datasource-classes` for a list of them.

.. warning::
   The API is not stable, and is subject to change.

.. _available-datasource-classes:

Available ``DataSource`` classes
--------------------------------

The available set of DataSource objects that can be instantiated by
:py:meth:`~sdoml.sources.DataSource` are as described below,
with an accompanying inheritance diagram.

.. automodapi:: sdoml.sources.sdoml_gcs
   :no-heading:
   :inheritance-diagram:
   :no-main-docstr:

.. note::

   Instrument-specific subclasses should not be instantiated. Instead, call
   :py:meth:`~sdoml.sources.DataSource` with the appropriate arguments,
   and let the appropriate subclass be instantiated!


Example
-------

If the user wishes to load HMI (Bx, By, Bz), AIA (94 Å, 131 Å), and EVE (O V, Fe XI),
:py:meth:`~sdoml.sources.DataSource` should be called as follows for each instrument.

The ``DataSource`` objects should be assigned in a `List`, e.g. ``datasource_arr``.
This ``List[DataSource]`` can be passed to the ``sdoml.dataloader.SDOMLDataset`` class, as discussed in
:doc:`dataloader`.

   >>> from sdoml.sources import DataSource
   >>> data_to_load = {
   ...    "HMI": {
   ...          "storage_location": "gcs",
   ...          "root": "fdl-sdoml-v2/sdomlv2_hmi_small.zarr/",
   ...          "channels": ["Bx", "By", "Bz"],
   ...    },
   ...    "AIA": {
   ...          "storage_location": "gcs",
   ...          "root": "fdl-sdoml-v2/sdomlv2_small.zarr/",
   ...          "channels": ["94A", "131A"],
   ...    },
   ...    "EVE": {
   ...          "storage_location": "gcs",
   ...          "root": "fdl-sdoml-v2/sdomlv2_eve.zarr/",
   ...          "channels": ["O V", "Fe XI"],
   ...    },
   ... }
   >>> datasource_arr = [DataSource(instrument=k, meta=v) for k, v in data_to_load.items()]

Writing a new DataSource class
------------------------------

A subclass of :py:meth:`~sdoml.sources.data_base.GenericDataSource` with the
``datasource`` method, will be registered with the
:py:meth:`~sdoml.sources.DataSource` factory. An example of the `datasource`
method, for :py:meth:`~sdoml.sources.sdoml_gcs.SDOML_AIA_GCS` is shown below:

.. code-block:: python

   @classmethod
   def datasource(cls, instrument: str, meta: Dict) -> bool:
      """
      Determines if the combination of ``instrument``, ``storage_location``,
      and filename (extracted from ``root``) should lead to the instantiation
      of this child class
      """
      return (
         instrument.lower() == "aia"
         and str(meta["storage_location"]).lower() == "gcs"
         and Path(str(meta["root"])).name == "sdomlv2_small.zarr"
      )

where upon instantiation of the :py:meth:`~sdoml.sources.DataSource`,
if the instrument name is  ``aia``, the storage location ``gcs``, and the
``.zarr`` file, ``sdomlv2_small.zarr``, :py:meth:`~sdoml.sources.sdoml_gcs.SDOML_AIA_GCS`
will be instantiated.

In addition to the ``datasource`` method, the following attributes
should be set in the subclass:

.. code-block:: python

   self._time_format : str                 # e.g. %Y-%m-%dT%H:%M:%S.%fZ

   # as a result of the abstractmethod ``self._get_years_channels()``
   self._available_years : List[str]       # e.g. ['2010','2011', ...]
   self._available_channels : List[str]    # e.g. ['94A', '131A', ...]

   # as a result of the abstractmethod ``self.load_data_meta()``
   self._data_by_year : List
   self._meta_by_year : List
   self._time_by_year : np.ndarray
