"""
Subclass for the SDOML Dataset (v2+) hosted on gcs
"""

from collections import defaultdict
import os
from typing import List, Tuple, Dict
from pprint import pformat
from functools import reduce

import logging

import pandas as pd

from multiprocessing import Pool
from sdoml.utils.utils import load_single_gcs_zarr
from sdoml.sources.data_base import GenericDataSource
from sdoml.utils.utils import inspect_single_gcs_zarr

__all__ = ["SDOAIA_gcs", "SDOHMI_gcs", "SDOEVE_gcs"]


class SDOAIA_gcs(GenericDataSource):
    """
    Data class for SDO/AIA located on GCS under the ``fdl-sdoml-v2`` bucket.

    The data is stored as a ``zarr.hierarchy.Group``, e.g.:

    ```
    /
    └── 2010
        ├── 131A (47116, 512, 512) float32
        ├── 1600A (47972, 512, 512) float32
        ├── 1700A (46858, 512, 512) float32
        ├── 171A (47186, 512, 512) float32
        ├── 193A (47134, 512, 512) float32
        ├── 211A (47186, 512, 512) float32
        ├── 304A (47131, 512, 512) float32
        ├── 335A (47187, 512, 512) float32
        └── 94A (46930, 512, 512) float32
    ```

    """

    def __init__(self, instrument, meta, years, cache_size, **kwargs):
        super().__init__(instrument, meta, years, cache_size, **kwargs)

        # set the time_format
        self.time_format = "%Y-%m-%dT%H:%M:%S.%fZ"

        # define the cache size
        self._cache_size = cache_size

        # obtain the years and channels available from the
        # provided _years / _channels
        yr_channel_dict = self._get_years_channels()
        self.years = list(yr_channel_dict.keys())
        # check that each year has the same channels
        self.channels = list(
            reduce(
                lambda i, j: i & j,
                (set(n) for n in list(yr_channel_dict.values())),
            )
        )

        # print the attributes of the class
        # logging.info(f'class attributes: {pformat(self.__dict__)}')

    def _get_years_channels(self) -> Dict[str, List[str]]:
        """
        Function for determining the available years and channels

        Returns
        -------

        yc_dict: Dict[str, List[str]]
            a dictionary with the keys being the years available, and the values
        being the available channels for that year.

        """
        yc_dict = {}

        # go through years, and channels ensuring we can read the data
        for i, channel in enumerate(self._meta["channels"]):
            for year in self._years:
                try:
                    _ = inspect_single_gcs_zarr(
                        os.path.join(self._meta["root"], year, channel)
                    )
                    if i == 0:
                        try:
                            yc_dict[year]
                        except KeyError:
                            yc_dict[year] = []  # Fix this
                        # yc_dict[year] = []
                    yc_dict[year].append(channel)
                except Exception:
                    logging.warning(
                        f"Cannot find ``{os.path.join(self._meta['root'], year, channel)}``"
                    )

        # check the data has the correct channels for all years
        if not yc_dict:
            logging.error("Empty yc_dict")

        return yc_dict

    def load_data_meta(self) -> Tuple[List, List]:
        """
        Method to load SDO/AIA data from the .zarr file on GCS

        Returns
        """

        # __init__ is enforcing this
        if self.years is None or self.channels is None:
            raise ValueError

        by_year = []
        meta_yr = []
        for yr in self.years:
            data = [
                load_single_gcs_zarr(
                    os.path.join(self._meta["root"], yr, ch),
                    cache_max_single_size=self._cache_size,
                )
                for ch in self.channels
            ]
            meta = [d.attrs for d in data]
            by_year.append(data)
            meta_yr.append(meta)

        return (by_year, meta_yr)

    def get_cotemporal_indices(
        self, original_df: pd.DataFrame, data_byyear: List, select_times
    ) -> pd.DataFrame:

        """
        Method to obtain co-temporal AIA data using a dataframe with a set of
        ``datetime`` indices

        Returns
        -------

        """

        # setting for multiprocessing
        self.__data_byyear = data_byyear
        self.__select_times = select_times

        # create a copy of the ``pd.DataFrame``
        og_df_copy = original_df.copy()

        logging.info(">>> Multiprocessing")
        with Pool(int(os.cpu_count() * 0.75)) as p:
            x_ = p.map(
                self.get_cotemporal_indices_singular,
                list(range(len(self.__data_byyear[0]))),
            )
        logging.info(">>> End of Multi-processing")

        for item in x_:
            logging.info(
                f">>> inserting column ``{self.channels[item[0]]}`` into the dataframe"
            )
            og_df_copy.insert(
                len(og_df_copy.columns), self.channels[item[0]], item[1]
            )

        return og_df_copy

    def get_cotemporal_indices_singular(self, idx) -> Tuple:
        """
        function to get the co-temporal data for one channel.
        This is called from the multiprocessing within
        ``self.get_cotemporal_indices``
        """

        arr = []

        # combine the seperate years
        for j in range(len(self.__data_byyear)):
            arr.extend(self.__data_byyear[j][idx].attrs["T_OBS"])

        pd_series = pd.to_datetime(arr, format=self.time_format, utc=True)
        select_index = self.find_selected_indices(
            pd_series, self.__select_times
        )
        select_index_removed_missing = self.find_remove_missing(
            pd_series, select_index, self.__select_times, "3m"
        )

        return (idx, select_index_removed_missing)

    @classmethod
    def datasource(cls, instrument: str, meta: Dict) -> bool:
        """
        Determines if the combination of ``instrument``, ``location``, and
        ``root`` should lead to the instantiation of this child class
        """

        return (
            instrument.lower() == "aia"
            and str(meta["storage_location"]).lower() == "gcs"
            and str(meta["root"]).startswith("fdl-sdoml-v2")
        )


class SDOHMI_gcs(SDOAIA_gcs):
    """
    Data class for SDO/HMI located on GCS under the ``fdl-sdoml-v2`` bucket.
    As ``SDOAIA_gcs`` with ``self.time_format`` where the data is stored in
    the time format: ``%Y.%m.%d_%H:%M:%S_TAI``

    As with SDOAIA_gcs, the data is stored as a ``zarr.hierarchy.Group``, e.g.:

    ```
    /
    └── 2010
        ├── Bx (25540, 512, 512) float32
        ├── By (25540, 512, 512) float32
        └── Bz (25540, 512, 512) float32
    ```

    """

    def __init__(self, instrument, meta, years, cache_size, **kwargs):
        super().__init__(instrument, meta, years, cache_size, **kwargs)

        # Main difference between AIA and HMI data is the time format.
        self.time_format = "%Y.%m.%d_%H:%M:%S_TAI"

    @classmethod
    def datasource(cls, instrument: str, meta: Dict) -> bool:
        """
        Determines if the combination of ``instrument``, ``location``, and
        ``root`` should lead to the instantiation of this child class
        """
        return (
            instrument.lower() == "hmi"
            and str(meta["storage_location"]).lower() == "gcs"
            and str(meta["root"]).startswith("fdl-sdoml-v2")
        )


class SDOEVE_gcs(GenericDataSource):
    """
    Data class for SDO/EVE(MEGS-A) located on GCS under the ``fdl-sdoml-v2`` bucket.
    As ``SDOAIA_gcs`` with ``self.time_format`` where the data is stored in
    the time format: ``%Y.%m.%d_%H:%M:%S_TAI``

    The data is stored as a ``zarr.hierarchy.Group``:

    ```
    /
    └── MEGS-A
        ├── C III (2137380,) float32
        ├── Fe IX (2137380,) float32
        ├── Fe VIII (2137380,) float32
        ├── Fe X (2137380,) float32
        ├── Fe XI (2137380,) float32
        ├── Fe XII (2137380,) float32
        ├── Fe XIII (2137380,) float32
        ├── Fe XIV (2137380,) float32
        ├── Fe XIX (2137380,) float32
        ├── Fe XV (2137380,) float32
        ├── Fe XVI (2137380,) float32
        ├── Fe XVIII (2137380,) float32
        ├── Fe XVI_2 (2137380,) float32
        ├── Fe XX (2137380,) float32
        ├── Fe XX_2 (2137380,) float32
        ├── Fe XX_3 (2137380,) float32
        ├── H I (2137380,) float32
        ├── H I_2 (2137380,) float32
        ├── H I_3 (2137380,) float32
        ├── He I (2137380,) float32
        ├── He II (2137380,) float32
        ├── He II_2 (2137380,) float32
        ├── He I_2 (2137380,) float32
        ├── Mg IX (2137380,) float32
        ├── Mg X (2137380,) float32
        ├── Mg X_2 (2137380,) float32
        ├── Ne VII (2137380,) float32
        ├── Ne VIII (2137380,) float32
        ├── O II (2137380,) float32
        ├── O III (2137380,) float32
        ├── O III_2 (2137380,) float32
        ├── O II_2 (2137380,) float32
        ├── O IV (2137380,) float32
        ├── O IV_2 (2137380,) float32
        ├── O V (2137380,) float32
        ├── O VI (2137380,) float32
        ├── S XIV (2137380,) float32
        ├── Si XII (2137380,) float32
        ├── Si XII_2 (2137380,) float32
        └── Time (2137380,) <U23
    ```

    """

    def __init__(self, instrument, meta, years, cache_size=None, **kwargs):
        super().__init__(instrument, meta, years, cache_size=None, **kwargs)

        # set the time_format
        self.time_format = "%Y-%m-%d %H:%M:%S.%f"

        # define the cache size
        self._cache_size = cache_size

        # obtain the years and channels available from the
        # provided _years / _channels
        yr_channel_dict = self._get_years_channels()
        # data is for all years
        self.channels = yr_channel_dict["all"]

        # print the attributes of the class
        # logging.info(f'class attributes: {pformat(self.__dict__)}')

    def _get_years_channels(self) -> Dict[str, List[str]]:
        """
        Function to return the available channels in the data.

        """
        # The data is not stored in a per-year format.
        yc_dict = {"all": []}

        for channel in self._meta["channels"]:
            try:
                _ = inspect_single_gcs_zarr(
                    os.path.join(self._meta["root"], "MEGS-A", channel)
                )
                yc_dict["all"].append(channel)
            except Exception:
                logging.warning(
                    f"Cannot find ``{os.path.join(self._meta['root'], 'MEGS-A', channel)}``"
                )

        # check the data has the correct channels for all years
        if not yc_dict:
            logging.error("Empty year/channel dictionary")

        return yc_dict

    def load_data_meta(self) -> List:
        """
        Method to load SDO/AIA data from the .zarr file on GCS

        Returns
        -------

        loaded_data:
        """

        loaded_data = [
            load_single_gcs_zarr(
                os.path.join(self._meta["root"], "MEGS-A", ch),
                cache_max_single_size=self._cache_size,
            )
            for ch in self.channels
        ]
        # all data share the same Time information. Add this to the end of the list.
        meta = [d.attrs for d in loaded_data]

        # by default the data contains one set of keys
        # to be consistent with other sdoml data,
        # create a list for each key of data length

        ddict_arr = []
        for x in meta:
            ddict = defaultdict(list)
            for k, v in x.items():
                ddict[k] = [v] * len(loaded_data[0])
            ddict_arr.append(ddict)

        loaded_data.append(
            load_single_gcs_zarr(
                os.path.join(self._meta["root"], "MEGS-A", "Time"),
                cache_max_single_size=self._cache_size,
            )
        )

        return [loaded_data], [ddict_arr]

    def get_cotemporal_indices(
        self, original_df: pd.DataFrame, data_byyear: List, select_times
    ) -> pd.DataFrame:

        """
        Method to obtain co-temporal AIA data using a dataframe with a set of
        ``datetime`` indices
        """

        # make a copy of the original dataframe
        og_df_copy = original_df.copy()

        # obtain co-temporal data
        x_ = self.get_cotemporal_indices_singular(data_byyear, select_times)

        # insert data into the df
        for channel in self.channels:
            og_df_copy.insert(len(og_df_copy.columns), channel, x_)

        return og_df_copy

    def get_cotemporal_indices_singular(self, dby, select_t) -> Tuple:
        """
        ...

        """
        pd_series = pd.to_datetime(
            dby[0][-1], format=self.time_format, utc=True
        )

        # all data for EVE is stored as a single array covering 4 years.
        # Downsample ``pd_series`` to only the relevant years
        itms = [i.year in [int(yr) for yr in self._years] for i in pd_series]
        pd_series = pd_series[itms]

        select_index = self.find_selected_indices(pd_series, select_t)
        select_index = self.find_remove_missing(
            pd_series, select_index, select_t, "3m"
        )

        return select_index

    @classmethod
    def datasource(cls, instrument: str, meta: Dict) -> bool:
        """
        Determines if the combination of ``instrument``, ``location``, and
        ``root`` should lead to the instantiation of this child class
        """
        return (
            instrument.lower() == "eve"
            and str(meta["storage_location"]).lower() == "gcs"
            and str(meta["root"]).startswith("fdl-sdoml-v2")
        )
