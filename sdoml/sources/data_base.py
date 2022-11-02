"""
``GenericDataSource`` is a generic ``DataSource`` class from which
all other DataSource classes inherit from.
"""

import logging
import os
from abc import ABC, abstractmethod
from itertools import repeat
from multiprocessing import get_context
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# from Python 3.9+, use dict, list, tuple; see PEP 585


__all__ = ["GenericDataSource"]


class GenericDataSource(ABC):
    """
    Generic DataSource Class to be inherited by specific DataSource classes.

    The following attributes should be set in the subclass:

    .. code-block:: python

        self._time_format : str                 # e.g. %Y-%m-%dT%H:%M:%S.%fZ

        # as a result of the abstractmethod ``self._get_years_channels()``
        self._available_years : List[str]       # e.g. ['2010','2011', ...]
        self._available_channels : List[str]    # e.g. ['94A', '131A', ...]

        # as a result of the abstractmethod ``self.load_data_meta()``
        self._data_by_year : List
        self._meta_by_year : List
        self._time_by_year : np.ndarray

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

    def __init__(
        self,
        instrument: str,
        meta: Dict,
        # years: List,
        # cache_size: int,
        **kwargs,
    ) -> None:

        # want to do checks here...
        self._instrument = instrument
        self._meta = meta

        # !TODO I feel like we can address these later on;
        # ... they can just be set by the dataloader
        self._requested_years = None  # sorted(years)
        self._cache_size = None  # cache_size

        #  -- Set some attributes that need to be updated in the subclasses
        #
        self._time_format: str
        #
        # the following can be an output of ``self._get_years_channels()``
        self._available_years: List[str]
        self._available_channels: List[str]
        #
        # the following can be an output of ``self.load_data_meta()``
        self._data_by_year: List
        self._meta_by_year: List
        self._time_by_year: np.ndarray
        # --

    def __repr__(self):
        """
        Return the class attributes
        """
        return ", \n".join((f"{k}: {v}" for k, v in vars(self).items()))

    def _find_remove_missing(
        self,
        series,
        selected_index: pd.DataFrame,  # pd.core.indexes.datetimes.DatetimeIndex
        selected_times: pd.DataFrame,
        timedelta: str,
    ) -> List[int]:
        """
        Find and remove missing indices within a specified ``timedelta``

        Parameters
        ----------

        series:
            ``pd.DataFrame`` of cotemporal observations ???

        selected_index:
            indices (from ``self._find_selected_indices``) of the ????
            that are closest to the selected_times

        selected_times:
            times that have been requested from the original data

        timedelta: str
            A string representing the frequency at which data should be obtained.
            See [here](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases)
            for a list of frequency aliases.
        """

        missing_index = np.where(
            np.abs(series[selected_index] - selected_times)
            > pd.Timedelta(timedelta)
        )[0].tolist()
        for midx in missing_index:
            # if there is a missing_index, set to NaN
            selected_index[midx] = pd.NA

        return selected_index

    def _find_selected_indices(
        self,
        pdseries: pd.DataFrame,
        selected_times: pd.DataFrame,
    ) -> List:
        """
        Find the set of indices of ``pdseries`` that are closest to ``selected_times``
        """

        return [np.argmin(abs(time - pdseries)) for time in selected_times]

    @abstractmethod
    def set_years_channels(self) -> None:
        """
        Determine the available years and channels.

        This method should set:

            self._available_years : List[str]
            self._available_channels : List[str]

        """
        raise NotImplementedError

    @abstractmethod
    def load_data_meta(self) -> None:
        """
        Load the data.

        This method should set:

            self._data_by_year : List
            self._meta_by_year : List
            self._time_by_year : np.ndarray

        """

        if self._available_years is None or self._available_channels is None:
            msg = "self._available_years or self._available_years is None. "
            "Run ``self.set_years_channels()`` first."
            raise ValueError(msg)

    def get_cotemporal_indices(
        self,
        original_df: pd.DataFrame,
        column_name: str,
        time_delta: Optional[str] = "3m",
    ) -> pd.DataFrame:
        """
        Function for determining cotemporal data

        Parameters
        ----------

        original_df: pd.DataFrame
            the ``pd.DataFrame`` that will be filled. Must contain
            a column with values that are datetime64[ns] and are associated with
            a time-zone (tz; must not be tz-naive).

        column_name: str
            the name of the column corresponding the requested times.

        time_delta: str: optional
            A string representing the frequency at which data should be obtained.
            See [here](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases)
            for a list of frequency aliases. By default this is '3m' (3 minutes)


        Returns
        -------

        og_df_copy: pd.DataFrame
            A copy of ``original_df``, populated with additional columns
            that correspond to the ``self._available_channels``.
            If there is no match within ``time_delta``, these are set to ``pd.NA``
        """

        # ensure ``self._data_by_year exists``
        if not hasattr(self, "_data_by_year"):
            raise ValueError(
                "``self._data_by_year`` is None. "
                "First run ``self.load_data_meta()``"
            )
            # check if aiapy does something like this... that requires methods to be called in order

        # ``time_delta``
        if not isinstance(time_delta, str):
            raise TypeError(f"{time_delta} is not a ``str``")
        try:
            pd.tseries.frequencies.to_offset(time_delta)
        except ValueError as e:
            raise ValueError(f"{time_delta} is not a valid time_delta.") from e

        # ``original_df; column_name``
        if not isinstance(original_df, pd.DataFrame):
            raise TypeError("``original_df`` is not a ``pd.DataFrame``")
        elif not isinstance(column_name, str):
            raise TypeError(f"{column_name} is not a ``str``")
        try:
            select_times = original_df[column_name]
            if not (
                pd.api.types.is_datetime64_ns_dtype(select_times)
                and pd.api.types.is_datetime64tz_dtype(select_times)
            ):
                raise TypeError(
                    f"``original_df[{column_name}] is not of type"
                    "``datetime64[ns]`` and ``DatetimeTZDtype``"
                )
                # This checks if the data is tz-naive (which may be an issue?).
                # !TODO look further into this. May want: datetime64[ns, UTC]
        except KeyError as e:
            raise KeyError(
                f"a column with name ``{column_name}`` does not exist in ``original_df``"
            ) from e

        # create a copy of the ``pd.DataFrame``
        og_df_copy = original_df.copy()

        logging.info(">>> Multiprocessing")
        # Uses 75% of CPU cores
        # !TODO may want to make this a variable
        with get_context("spawn").Pool(int(os.cpu_count() * 0.75)) as p:
            channel_indices = p.starmap(
                self._get_cotemporal_indices_singular,  # function
                zip(
                    list(range(len(self._data_by_year[0]))),
                    repeat(original_df[column_name]),
                    repeat(time_delta),
                ),
            )
        logging.info(">>> End of Multi-processing")

        for item in channel_indices:
            logging.info(
                f">>> inserting column ``{self._available_channels[item[0]]}``"
                " into the dataframe"
            )
            og_df_copy.insert(
                len(og_df_copy.columns),
                self._available_channels[item[0]],
                item[1],
            )

        return og_df_copy

    def _get_cotemporal_indices_singular(
        self,
        idx: int,
        select_times: pd.DataFrame,
        t_delta: str,
    ) -> Tuple[int, List[int]]:
        """
        Return a set of indices in the loaded data that correspond to
        observations at a set of chosen (``select_times``).

        This method calls ``self._find_selected_indices`` to obtain the set of
        indices in the loaded data that correspond to the desired times
        (``select_index``). This is later passed to ``self._find_remove_missing``
        to limit those to matches within a given ``t_delta``.

        Returns
        -------
        (idx, select_index_removed_missing): Tuple[int, List[?]]
        """

        if not isinstance(idx, int):
            raise TypeError
        if self.time_format is None:
            raise ValueError

        data_years = self._time_by_year[idx]

        # create a series based on the years of the loaded data
        pd_series = pd.to_datetime(
            data_years, format=self.time_format, utc=True
        )
        # find indices of loaded data that match the desired times
        select_index = self._find_selected_indices(
            pdseries=pd_series, selected_times=select_times
        )
        # remove rows where there is not a match between loaded data and
        # requested times to within ``t_delta``
        select_index_removed_missing = self._find_remove_missing(
            series=pd_series,
            selected_index=select_index,
            selected_times=select_times,
            timedelta=t_delta,
        )

        return (idx, select_index_removed_missing)

    @property
    def data_meta_time(self) -> Tuple[List, List, List]:
        """
        return the data/metadata/time by_year
        """
        return (self._data_by_year, self._meta_by_year, self._time_by_year)

    @property
    def time_format(self) -> str:
        """
        return the *provided* time format of the data
        """
        return self._time_format

    @property
    def available_years(self) -> List[str]:
        """
        return the list of available years in the data
        """
        return self._available_years

    @property
    def available_channels(self) -> List[str]:
        """
        return the list of channels that are available in all ``available_years``
        """
        return self._available_channels

    @classmethod
    @abstractmethod
    def datasource(cls) -> bool:
        """
        Determines if the subclass should be instantiated
        """
        return NotImplementedError
