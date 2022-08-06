"""
GenericDataSource is a generic DataSource class from which all other DataSource classes inherit from.
"""

import os
import logging
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from sdoml.utils.utils import inspect_single_gcs_zarr
from tqdm.autonotebook import tqdm
from typing import Dict, List

__all__ = ["GenericDataSource"]


class GenericDataSource(ABC):
    """
    Generic DataSource Class to be inherited by specific DataSource classes.
    """

    # initialise the ``_registry`` with an empty dict()
    _registry = dict()

    def __init_subclass__(cls, **kwargs):
        """
        This hook initialises the subclasses of the GenericDataSource class.
        This block of code is called for each subclass, and is used to register
        each subclass in a dict that has the ``datasource`` attribute.
        This is passed into the DataSource Factory for registration.
        """
        super().__init_subclass__(**kwargs)

        if hasattr(cls, "datasource"):
            cls._registry[cls] = cls.datasource

    def __init__(self, instrument, meta, years, cache_size, **kwargs):
        self._instrument = instrument
        self._meta = meta
        self._years = sorted(years)
        self._cache_size = cache_size

        # The following need to be overwritten as part of the implementation
        self.years = None
        self.channels = None

    def __repr__(self):
        """
        Function to show the attrs of self
        """
        return ", \n ".join((f"{k}: {v}" for k, v in vars(self).items()))

    def find_remove_missing(
        self, series, selected_index, gg, timedelta: str = "3m"
    ) -> List[int]:
        """
        find and remove missing indices within a specified ``timedelta``
        """

        missing_index = np.where(
            np.abs(series[selected_index] - gg) > pd.Timedelta(timedelta)
        )[0].tolist()
        for midx in missing_index:
            # if there is a missing_index, set to NaN
            selected_index[midx] = pd.NA

        return selected_index

    def find_selected_indices(
        self, pdseries: pd.DataFrame, selected_times
    ) -> List:
        """
        Find the set of indices of ``pdseries`` that are closest to ``selected_times``
        """
        return [np.argmin(abs(time - pdseries)) for time in selected_times]

    @abstractmethod
    def _get_years_channels(self, yrs, chnnls) -> Dict:
        """
        Function for determining cotemporal data

        set ``self.years`` and ``self.channels``
        """
        pass

    @abstractmethod
    def load_data_meta(self) -> None:
        """
        Function to load the data
        """
        pass

    @abstractmethod
    def get_cotemporal_indices(self) -> None:
        """
        Function for determining cotemporal data
        """
        pass
