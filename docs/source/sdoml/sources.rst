.. _sources:

================================
Data Sources (``sdoml.sources``)
================================

.. toctree::
   :maxdepth: 1

Currently, ``sdoml`` contains a small number of available data sources,
prefixed with ``SDOML``, that inherit from
:py:meth:`~sdoml.sources.data_base.GenericDataSource`.

Here, :py:meth:`~sdoml.sources.data_base.GenericDataSource` is a generic ``DataSource``
class from which all other ``DataSource`` classes inherit from.
Each of the children have a `datasource` method,
e.g. :py:meth:`~sdoml.sources.sdoml_gcs.SDOML_AIA_GCS.datasource`,
which provides information on what input parameters will lead to class instantiation
e.g. for :py:meth:`~sdoml.sources.sdoml_gcs.SDOML_AIA_GCS`, this is:

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

.. code-block:: python

   from sdoml.sources import DataSource

   data_to_load = {
      "HMI": {
            "storage_location": "gcs",
            "root": "fdl-sdoml-v2/sdomlv2_hmi_small.zarr/",
            "channels": ["Bx", "By", "Bz"],
      },
      "AIA": {
            "storage_location": "gcs",
            "root": "fdl-sdoml-v2/sdomlv2_small.zarr/",
            "channels": ["94A", "131A"],
      },
      "EVE": {
            "storage_location": "gcs",
            "root": "fdl-sdoml-v2/sdomlv2_eve.zarr/",
            "channels": ["O V", "Fe XI"],
      },
   }

   years = ["2010", "2011"]
   cache_max_size = 1 * 512 * 512 * 2048

   datasource_arr = [DataSource(instrument=k, meta=v) for k, v in data_to_load.items()]

.. automodapi:: sdoml.sources
   :include-all-objects:
   :inherited-members:
   :no-main-docstr:
