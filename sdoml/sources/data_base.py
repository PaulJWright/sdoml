"""
GenericDataSource is a generic DataSource class from which all other DataSource classes inherit from.
"""

from abc import ABC, abstractmethod
from tqdm.autonotebook import tqdm
from sdoml.utils.utils import inspect_single_gcs_zarr
import os

import logging
import numpy as np
import pandas as pd
from typing import Dict, List

__all__ = ["GenericDataSource"]


class GenericDataSource(ABC):
    """..."""

    # initialise the ``_registry`` with an empty dict()
    _registry = dict()

    def __init_subclass__(cls, **kwargs):
        """
        This hook initialises the subclasses of the GenericData class.
        This block of code is called for each subclass, and is used to register
        each subclass in a dict that has the ``datasource`` attribute.
        This is passed into the Map Factory for registration.
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

    def __getitem__(self):
        raise NotImplementedError

    @abstractmethod
    def _get_years_channels(self, yrs, chnnls) -> Dict:
        """
        Function for determining cotemporal data
        """
        yc_dict = {}

        # go through years, and channels ensuring we can read the data
        for key, values in chnnls.items():
            for year in self._years:
                for i, channel in enumerate(values):
                    # print(key, year, channel, self.zarr_root[key])
                    # print(os.path.join(self.zarr_root[key], year, channel))
                    try:
                        _ = inspect_single_gcs_zarr(
                            os.path.join(self.zarr_root[key], year, channel)
                        )

                        if i == 0:
                            try:
                                yc_dict[year]
                            except KeyError:
                                yc_dict[year] = {}

                            yc_dict[year][key] = []

                        yc_dict[year][key].append(channel)
                    except Exception:
                        logging.warning(
                            f"Cannot find ``{os.path.join(self.zarr_root[key], year, channel)}``"
                        )
        # check the data has the correct channels for all years
        if not yc_dict:
            logging.error("Empty yc_dict")

        return yc_dict

    def __repr__(self):
        """
        Function to show the attrs of self
        """
        return ", ".join((f"{k}: {v}" for k, v in vars(self).items()))

    @abstractmethod
    def load_data_meta(self) -> None:
        """
        Function to load the data

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def get_cotemporal_indices(self) -> None:
        """
        Function for determining cotemporal data
        """
        pass

    def find_remove_missing(
        self, series, selected_index, gg, timedelta
    ) -> List[int]:
        """
        find and remove missing indices within a specified timedelta
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
        """..."""
        return [np.argmin(abs(time - pdseries)) for time in selected_times]
