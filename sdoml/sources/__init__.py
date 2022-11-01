"""
Data sources are seperated into their own files with at least one class defined.

As demonstrated in :py:meth:`~sdoml.sources.sdoml_gcs`, these should be children
of the :py:meth:`~sdoml.sources.data_base.GenericDataSource` parent class. While
:py:meth:`~sdoml.sources.data_base.GenericDataSource` isn't instantiated
directly, this class described the attributes that should be set in the child
class. e.g.

.. code-block:: python

    self._time_format : str                 # e.g. %Y-%m-%dT%H:%M:%S.%fZ

    # as a result of the abstractmethod ``self._get_years_channels()``
    self._available_years : List[str]       # e.g. ['2010','2011', ...]
    self._available_channels : List[str]    # e.g. ['94A', '131A', ...]

    # as a result of the abstractmethod ``self.load_data_meta()``
    self._data_by_year : List
    self._meta_by_year : List
    self._time_by_year : np.ndarray

For examples surrounding the SDOML v2+ dataset on Google Cloud Storage,
see :py:meth:`~sdoml.sources.sdoml_gcs`.
"""

from .data_base import *
from .sdoml_gcs import *
