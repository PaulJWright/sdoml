"""
Data sources are seperated into their own files with at least one class defined.

Here, :py:meth:`~sdoml.sources.data_base.GenericDataSource` is a generic ``DataSource``
class from which all other ``DataSource`` classes inherit from. While
:py:meth:`~sdoml.sources.data_base.GenericDataSource` isn't instantiated
directly, this class described the attributes that should be set in the child
class. e.g.:

.. code-block:: python

    self._time_format : str                 # e.g. %Y-%m-%dT%H:%M:%S.%fZ

    # as a result of the abstractmethod ``self._get_years_channels()``
    self._available_years : List[str]       # e.g. ['2010','2011', ...]
    self._available_channels : List[str]    # e.g. ['94A', '131A', ...]

    # as a result of the abstractmethod ``self.load_data_meta()``
    self._data_by_year : List
    self._meta_by_year : List
    self._time_by_year : np.ndarray

For examples surrounding the SDOML v2+ dataset on Amazon Web Services,
see :py:meth:`~sdoml.sources.sdoml_aws`.
"""

from .data_base import *
from .dataset_factory import *
from .sdoml_cloud import *
