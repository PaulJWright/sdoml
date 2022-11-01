.. _dataloader:

=================================
Dataloader (``sdoml.dataloader``)
=================================

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

For example, to load HMI (Bx, By, Bz), AIA (94 Å, 131 Å), and EVE (O V, Fe XI) for
2010, located in Google Cloud Storage (``gcs``)
:py:meth:`~sdoml.dataloader.SDOMLDataset` can be called as follows, where the
creation of ``datasource_arr`` was described in :doc:`sources`.

.. code-block:: python

   from sdoml import SDOMLDataset

   sdomlds = SDOMLDataset(
      cache_max_size=1 * 512 * 512 * 4096,
      years=["2010"],
      data_to_load=datasource_arr, # this is a List[``sdo.sources.DataSource``]
   )

``sdomlds`` can then be used as any other dataloader. If caching is implemented,
the second request on an index will be quicker. e.g.:

.. code-block:: python

   >>> first ``sdomlds.__getitem__(0)`` request took 69.02 seconds
   >>> second ``sdomlds.__getitem__(0)`` request took 0.54 seconds

.. warning::
   The API is not stable, and is subject to change.

.. automodapi:: sdoml.dataloader
   :include-all-objects:
   :inherited-members:
   :no-inheritance-diagram:
