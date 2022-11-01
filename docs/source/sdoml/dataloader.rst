.. _dataloader:

SDOML Dataloader (``sdoml.dataloader``)
=======================================

.. toctree::
   :maxdepth: 1

The initial implementation of ``sdoml`` provides a single dataloader/Dataset
class for the SDOML v2+ data located on the Google Cloud Platform.

The :py:meth:`~sdoml.dataloader.SDOMLDataset` accepts `DataSource` objects.
.. The :py:meth:`~sdoml.sources.GenericDataSource` object is a factory for generating

For example, to load HMI (Bx, By, Bz), AIA (94A, 131A), and EVE (O V, Fe XI) for
2010, located in Google Cloud Storage (``gcs``)
:py:meth:`~sdoml.dataloader.SDOMLDataset` can be called as follows:

.. code-block:: python

   sdomlds = SDOMLDataset(
      cache_max_size=1 * 512 * 512 * 4096,
      years=["2010"],
      data_to_load={
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
      },
   )


.. automodapi:: sdoml.dataloader
   :include-all-objects:
   :inherited-members:
   :no-inheritance-diagram:
