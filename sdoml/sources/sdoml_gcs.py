"""
Subclass for the SDOML Dataset (v2+) hosted on gcs
"""

import logging
import os
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sdoml.sources.data_base import GenericDataSource
from sdoml.utils.utils import (
    inspect_single_gcs_zarr,
    load_single_gcs_zarr,
    solve_list,
)

# from Python 3.9+, use dict, list, tuple; see PEP 585


__all__ = ["SDOAIA_gcs", "SDOHMI_gcs", "SDOEVE_gcs"]

# MEGS-A is depreacated; these years are a constant.
EVE_MEGSA_YEARS = ["2010", "2011", "2012", "2014"]


class SDOAIA_gcs(GenericDataSource):
    """
    Data class for SDO/AIA located on GCS under the ``fdl-sdoml-v2`` bucket.

    The data is stored as a ``zarr.hierarchy.Group``, e.g.:

    .. code-block::

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

    """

    def __init__(self, instrument, meta, years, cache_size, **kwargs):
        super().__init__(instrument, meta, years, cache_size, **kwargs)

        # Set the time format of the data
        self._time_format = "%Y-%m-%dT%H:%M:%S.%fZ"

    def set_years_channels(self) -> None:
        """
        Function for determining the available years and channels.

        The method should set:

            - self._available_years : List[str]
                List of available years

            - self._available_channels : List[str]
                List of available channels (over all years)

        """
        # Returns
        # -------

        # yc_dict: Dict[str, List[str]]
        #     - Keys are the subset of years available from ``yrs``
        #     - Values are the subset of channels available (from ``chnnls``)
        #     for the given key (year)

        # e.g.

        # ```
        # yc_dict = {
        #     '2010': ['94A', '131A', '171A', ...],
        #     '2011': ['94A', '131A', '171A', ...],
        #     }
        # ```

        yc_dict = {}

        # go through years, and channels ensuring we can read the data
        for i, channel in enumerate(self._meta["channels"]):
            for year in self._requested_years:
                path_to_data = os.path.join(self._meta["root"], year, channel)

                try:
                    _ = inspect_single_gcs_zarr(path_to_data)

                    if i == 0:
                        try:
                            yc_dict[year]
                        except KeyError:
                            yc_dict[year] = []

                    yc_dict[year].append(channel)

                except Exception:
                    logging.warning(f"Cannot find ``{path_to_data}``")

        # check the data has the correct channels for all years
        if not yc_dict:
            logging.error("Empty yc_dict")

        # setting self._available_years
        # setting self._available_channels
        #
        self._available_years = list(yc_dict.keys())
        # check that each year has the same channels
        self._available_channels = list(
            reduce(
                lambda i, j: i & j,
                (set(n) for n in list(yc_dict.values())),
            )
        )
        self._available_channels = solve_list(
            self._available_channels, self._meta["channels"]
        )  # setting to the same order as the items provided

    def load_data_meta(self) -> None:
        """
        Method to load SDO/AIA data from the ``.zarr`` file on GCS

        This method should set:

            - self._data_by_year: List:
                contains the data loaded per year, and per channel,
                e.g. ``by_year[year_index][channel_index]``

            - self._meta_by_year: List:
                contains the data loaded per year, and per channel,
                e.g. ``by_year[year_index][channel_index]``

            - self._time_by_year: np.ndarray:
                contains observation time per year, and per channel,
                e.g. ``time_yr[year_index][channel_index]``
        """
        super().load_data_meta()

        by_year, meta_yr = [], []
        for yr in self._available_years:
            data = [
                load_single_gcs_zarr(
                    path_to_zarr=os.path.join(self._meta["root"], yr, ch),
                    cache_max_single_size=self._cache_size,
                )
                for ch in self._available_channels
            ]
            meta = [d.attrs for d in data]
            by_year.append(data)
            meta_yr.append(meta)

        time_yr = []
        for idx in range(len(meta_yr[0])):
            data_years = []
            # combine the seperate years of the data
            for item in meta_yr:
                data_years.extend(item[idx]["T_OBS"])
            time_yr.append(np.array(data_years))

        time_yr = np.array(time_yr, dtype="object")

        self._data_by_year = by_year
        self._meta_by_year = meta_yr
        self._time_by_year = time_yr

    @classmethod
    def datasource(cls, instrument: str, meta: Dict) -> bool:
        """
        Determines if the combination of ``instrument``, ``storage_location``,
        and filename (extracted from ``root``) should lead to the instantiation
        of this child class
        """
        return (
            instrument.lower() == "aia"
            and str(meta["storage_location"]).lower() == "gcs"
            and Path(str(meta["root"])).name == "sdomlv2_small.zarr"
        )


class SDOHMI_gcs(SDOAIA_gcs):
    """
    Data class for SDO/HMI located on GCS under the ``fdl-sdoml-v2`` bucket.
    As ``SDOAIA_gcs`` with ``self._time_format`` where the data is stored in
    the time format: ``%Y.%m.%d_%H:%M:%S_TAI``

    As with SDOAIA_gcs, the data is stored as a ``zarr.hierarchy.Group``, e.g.:

    .. code-block:: bash

        /
        └── 2010
            ├── Bx (25540, 512, 512) float32
            ├── By (25540, 512, 512) float32
            └── Bz (25540, 512, 512) float32
    """

    def __init__(self, instrument, meta, years, cache_size, **kwargs):
        super().__init__(instrument, meta, years, cache_size, **kwargs)

        # Main difference between AIA and HMI data is the time format.
        self._time_format = "%Y.%m.%d_%H:%M:%S_TAI"

    @classmethod
    def datasource(cls, instrument: str, meta: Dict) -> bool:
        """
        Determines if the combination of ``instrument``, ``storage_location``,
        and filename (extracted from ``root``) should lead to the instantiation
        of this child class
        """
        return (
            instrument.lower() == "hmi"
            and str(meta["storage_location"]).lower() == "gcs"
            and Path(str(meta["root"])).name == "sdomlv2_hmi_small.zarr"
        )


class SDOEVE_gcs(GenericDataSource):
    """
    Data class for SDO/EVE(MEGS-A) located on GCS under the ``fdl-sdoml-v2``
    bucket. As ``SDOAIA_gcs`` with ``self._time_format`` where the data is
    stored in the time format: ``%Y.%m.%d_%H:%M:%S_TAI``

    The data is stored as a ``zarr.hierarchy.Group``:

    .. code-block:: bash

        /
        └── MEGS-A
            ├── C III (2137380,) float32
            ├── Fe IX (2137380,) float32
            ⋮
            ├── Si XII_2 (2137380,) float32
            └── Time (2137380,) <U23
    """

    def __init__(self, instrument, meta, years, cache_size=None, **kwargs):
        super().__init__(instrument, meta, years, cache_size=None, **kwargs)

        # set the time_format
        self._time_format = "%Y-%m-%d %H:%M:%S.%f"
        self._available_years = EVE_MEGSA_YEARS

    def set_years_channels(self) -> Dict[str, List[str]]:
        """
        Determine the available years and channels.

        As EVE only has data for 2010 - 2014, this method only
        sets self._available_channels; self._available_years is
        set to ``EVE_MEGSA_YEARS`` in __init__

        ```
        """
        # Returns
        # -------

        # yc_dict: Dict[str, List[str]]
        #     - Keys are the subset of years available from ``yrs``.
        #     For EVE, this is just set to 'all'.
        #     - Values are the subset of channels available (from ``chnnls``)
        #     for the given key (year)

        # e.g.
        # ```
        # yc_dict = {
        #     'all': ['C III', 'Fe IX', ..., 'Si XII_2'],
        #     }

        # The data is not stored per-year, but instead already combined
        yc_dict = {"all": []}

        for channel in self._meta["channels"]:
            path_to_data = os.path.join(self._meta["root"], "MEGS-A", channel)
            try:
                _ = inspect_single_gcs_zarr(path_to_data)  # check if exists
                yc_dict["all"].append(channel)
            except Exception:
                logging.warning(f"Cannot find ``{path_to_data}``")

        if not yc_dict:
            logging.error("Empty year/channel dictionary")

        self._available_channels = solve_list(  # 3)
            yc_dict["all"], self._meta["channels"]
        )

    def load_data_meta(self) -> None:
        """
        Method to load SDO/EVE data from the ``.zarr`` file on GCS, return
        the ``loaded_data``, the metadata as array of dictionaries ``dict_arr``,
        and a ``np.ndarray`` of the time information for each cahnnel.

        Returns
        -------
        (loaded_data, dict_arr, time_yr): Tuple(List, List):

            - loaded_data: List:
                contains the data loaded per year, and per channel,
                e.g. ``by_year[year_index][channel_index]``

            - dict_arr: List:
                contains the data loaded per year, and per channel,

            - time_yr: np.ndarray:
                contains observation time per year, and per channel,
                e.g. ``time_yr[year_index][channel_index]``
        """
        super().load_data_meta()

        loaded_data = [
            load_single_gcs_zarr(
                os.path.join(self._meta["root"], "MEGS-A", ch),
                cache_max_single_size=self._cache_size,
            )
            for ch in self._available_channels
        ]

        meta = [d.attrs for d in loaded_data]

        # by default the data contains one set of keys
        # to be consistent with other sdoml data,
        # create a list for each key of data length
        dict_arr = []
        for x in meta:
            ddict = defaultdict(list)
            for k, v in x.items():
                ddict[k] = [v] * len(loaded_data[0])
            dict_arr.append(ddict)

            # !TODO add T_OBS to metadata

        time_yr = np.array(
            [
                load_single_gcs_zarr(
                    os.path.join(self._meta["root"], "MEGS-A", "Time"),
                    cache_max_single_size=self._cache_size,
                )
            ]
            * len(self.available_channels)
        )

        self._data_by_year = [loaded_data]  # 4)
        self._meta_by_year = [dict_arr]  # 5)
        self._time_by_year = time_yr  # 6)

        # return ([loaded_data], [dict_arr], time_yr)

    def get_cotemporal_indices(
        self,
        original_df: pd.DataFrame,
        column_name: str,
        time_delta: str = "3m",
    ) -> pd.DataFrame:

        """
        Method to return a ``pd.DataFrame`` with co-temporal EVE data for
        ``self._available_channels``.

        This method replaces ``get_cotemporal_indices`` in the parent class
        with a more efficient implementation (EVE has co-temporal data between
        different ions: we only need to match to ``original_df`` once).

        Returns
        -------

        og_df_copy: pd.DataFrame
            Copy of the ``original_df`` with additional columns corresponding
            to ``self._available_channels``
        """
        # as eve is already co-temporal, only need to run on a single index
        EVE_INDEX = 0

        # make a copy of the original dataframe
        og_df_copy = original_df.copy()
        # obtain co-temporal data
        channel_indices = self._get_cotemporal_indices_singular(
            EVE_INDEX, original_df[column_name], time_delta
        )
        # insert data into the df
        for channel in self._available_channels:
            og_df_copy.insert(
                loc=len(og_df_copy.columns),
                column=channel,
                value=channel_indices[1],
            )

        return og_df_copy

    @classmethod
    def datasource(cls, instrument: str, meta: Dict) -> bool:
        """
        Determines if the combination of ``instrument``, ``storage_location``,
        and filename (extracted from ``root``) should lead to the instantiation
        of this child class
        """
        return (
            instrument.lower() == "eve"
            and str(meta["storage_location"]).lower() == "gcs"
            and Path(str(meta["root"])).name == "sdomlv2_eve.zarr"
        )
