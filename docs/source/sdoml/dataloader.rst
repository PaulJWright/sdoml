.. _dataloader:

SDOML Dataloader (``sdoml.dataloader``)
=======================================

.. toctree::
   :maxdepth: 1

The initial implementation of ``sdoml`` provides a single dataloader
class for the SDOML v2+ data located on the Google Cloud Platform.

The :py:meth:`~sdoml.dataloader.SDOMLDataset` instantiates ``DataSource``
objects (see :ref:`sources`). When called,
:py:meth:`~sdoml.sources.dataset_factory.DataSource` utilises the
:py:meth:`~sdoml.sources.dataset_factory.DataSourceFactory` to instantiate one
of the registered types which match the arguments provided. If the match
exists, that type is used. If there are ``0`` or ``>=1`` matches,
:py:meth:`~sunpy.map.map_factory.NoMatchError` or
:py:meth:`~sunpy.map.map_factory.MultipleMatchError` will be raised.

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

.. warning::
   The API is not stable, and is subject to change.

.. automodapi:: sdoml.dataloader
   :include-all-objects:
   :inherited-members:
   :no-inheritance-diagram:
