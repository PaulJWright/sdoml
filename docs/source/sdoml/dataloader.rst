.. _dataloader:

=================================
Dataloader (``sdoml.SDOMLDataset``)
=================================

.. toctree::
   :maxdepth: 1

This initial implementation of ``sdoml`` provides a single torch dataloader
class for the SDOML v2+ data located on the Google Cloud Platform.

The :py:meth:`~sdoml.dataloader.SDOMLDataset` accepts a List of ``DataSource``
objects (see :ref:`sources`).

.. warning::
   The API is not stable, and is subject to change.

.. automodapi:: sdoml.dataloader
   :no-heading:
   :include-all-objects:
   :inherited-members:
   :no-inheritance-diagram:


Example
-------

If the user wishes to load HMI (Bx, By, Bz), AIA (94 Å, 131 Å), and EVE (O V, Fe XI) from the year 2010,
:py:meth:`~sdoml.dataloader.SDOMLDataset` can be called as follows, where the
creation of ``datasource_arr`` was described in :doc:`sources` (but reproduced minimally using SDO/EVE for clarity)

.. code-block:: python

   >>> import torch
   >>> from sdoml import SDOMLDataset
   >>> from sdoml.sources import DataSource
   >>> datasource = DataSource(
   ...     instrument="EVE",
   ...     meta={
   ...         "storage_location": "aws",
   ...         "root": "s3://gov-nasa-hdrl-data1/contrib/fdl-sdoml/fdl-sdoml-v2/sdomlv2_eve.zarr/",
   ...         "channels": ["O V", "Fe XI"],
   ...     }
   ... )
   >>> sdomlds = SDOMLDataset(
   ...    cache_max_size=1 * 512 * 512 * 4096,
   ...    years=["2010"],
   ...    data_to_load=[datasource], # this is a List[``sdo.sources.DataSource``]
   ... )
   >>> dataloader = torch.utils.data.DataLoader(
   ...    sdomlds,
   ...    batch_size=1,
   ...    shuffle=False,
   ... )

For more examples see, :doc:`examples/index`.

.. note::

   If caching is implemented, the second request on an index will be quicker. e.g.:

.. code-block:: python

   >>> for i in ["first", "second"]:  # doctest: +SKIP
   ...    start = timeit.default_timer()
   ...    _ = sdomlds.__getitem__(0)
   ...    end = timeit.default_timer()
   ...    logger.info(
   ...       f"{i} ``sdomlds.__getitem__(0)`` request took {end-start} seconds"
   ...    )
   first ``sdomlds.__getitem__(0)`` request took 69.02 seconds  # doctest: +SKIP
   second ``sdomlds.__getitem__(0)`` request took 0.54 seconds  # doctest: +SKIP
