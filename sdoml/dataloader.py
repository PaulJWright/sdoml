import os
import sys

import logging
import torch

import dask.array as da
import numpy as np
import pandas as pd

from collections import defaultdict
from datetime import date
from pprint import pformat
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from tqdm.autonotebook import tqdm

from sdoml.sources.dataset_factory import DataSource

# -- Setting up logging
logger = logging.getLogger(__name__)
logging.getLogger("sdoml").addHandler(logging.NullHandler())


class SDOMLDataset(Dataset):
    """
    Dataset class for the SDOML v2.+ (`.zarr`) data.

    Parameters
    ----------

    cache_max_size: int, Nonetype, optional
        The maximum size that the ``zarr`` cache may grow to,
        in number of bytes. By default this variable is 1 * 512 * 512 * 2048.
        If the variable is set to ``None``, the cache will have unlimited size.
        see https://zarr.readthedocs.io/en/stable/api/storage.html#zarr.storage.LRUStoreCache

    years: List[str], Nonetype, optional
        A list of years (from 2010 to present) to include. By default this
        variable is ``2010`` which will return data for 2010 only.

    data_to_load: Dict[str, Dict[str]]
        A dictionary of instruments to include.

        Options :
            - ``AIA`` : SDO Atomospheric Imaging Assembly
            - ``HMI`` : SDO Helioseismic and Magnetic Imager
            - ``EVE`` : Extreme UltraViolet Variability Experiment

        Each instrument should be a dictionary with the following keys:

        - storage_location: str
            Storage location of the file.

            Options :
                - ``gcs`` : Google Cloud Storage


        - root: str
            Location of the root ``.zarr`` file within the ``storage_location``.
            By default this is ``fdl-sdoml-v2/sdomlv2_small.zarr/`` (which is
            located on Google Cloud Storage ``storage_location == gcs``).


        - channels: List[str]
            A list of channels to include from each instrument.


    Example
    -------

    ```
    sdomlds = SDOMLDataset(
        cache_max_size=1 * 512 * 512 * 4096,
        years=["2010", "2011"],
        data_to_load={
                "HMI": {
                    "storage_location": "gcs",
                    "root": "fdl-sdoml-v2/sdomlv2_hmi_small.zarr/",
                    "channels": ["Bx", "By", "Bz"],
                    },
                "AIA": {
                    "storage_location": "gcs",
                    "root": "fdl-sdoml-v2/sdomlv2_small.zarr/",
                    "channels": ["94A", "131A", "171A", "193A", "211A", "335A"],
                    },
                "EVE": {
                    "storage_location": "gcs",
                    "root": "fdl-sdoml-v2/sdomlv2_eve.zarr/",
                    "channels": ["O V", "Mg X", "Fe XI"],
                },
            },
        )
    ```
    """

    def __init__(
        self,
        cache_max_size: Optional[int] = 1 * 512 * 512 * 2048,
        years: Optional[List[str]] = None,
        data_to_load: Optional[List[str]] = None,
        freq="6T",
    ):

        # !TODO implement passing of ``selected_times`` and ``required_keys``
        selected_times = None
        # required_keys = None

        if years is None:
            years = ["2010"]

        self._cache_max_size = cache_max_size
        self._single_cache_max_size = self._cache_max_size / len(data_to_load)
        self._years = years

        # instantiate the appropriate classes
        data_arr = [
            DataSource(k, v, self._years, self._single_cache_max_size)
            for k, v in data_to_load.items()
        ]

        # !TODO rearrange data for cadence (lowest to highest)

        if selected_times:
            # check the provided selected times are in agreement with the data
            # self.selected_times = self._check_selected_times(selected_times)
            raise NotImplementedError
        else:
            self.selected_times = self._select_times(freq)

        # Create a ``pd.DataFrame`` for the set of selected_times
        df = pd.DataFrame(
            self.selected_times,
            index=np.arange(np.shape(self.selected_times)[0]),
            columns=["selected_times"],
        )

        self.loaded_data, self.loaded_meta = zip(
            *[darr.load_data_meta() for darr in data_arr]
        )

        self.channels = [darr.channels for darr in data_arr]

        # Go through the time component of the loaded_data,
        # match to ``df.selected_times``, and delete any rows that have NaN.
        # Doing this (especially when data is ordered by cadence)
        # reduces the number of potential matches for the next set of data.
        for i, darr in enumerate(data_arr):
            if i == 0:
                self.df = darr.get_cotemporal_indices(
                    df, self.loaded_data[i], self.selected_times
                )
            else:
                self.df = darr.get_cotemporal_indices(
                    self.df, self.loaded_data[i], self.df["selected_times"]
                )

            self.df.dropna(inplace=True)
            self.df.reset_index(drop=True, inplace=True)

        # utilise cotemporal indices to get cotemporal data / metadata
        self.all_data = self.get_cotemporal_data()
        self.all_meta = self.get_cotemporal_meta()

    def get_cotemporal_meta(self) -> List[Dict]:
        """
        Function to return co-temporal metadata

        Returns
        -------
        dnr_arr: List[Dict]
            A list of dictionaries for each channel containing
            the respective metadata for that instrument
        """
        dictionaries = []

        for meta, channel_name in zip(self.loaded_meta, self.channels):
            # iterate through years
            dictionaries_ = []
            for idx in range(len(channel_name)):
                dd = defaultdict(list)

                # go through years
                for idy in range(len(meta)):
                    for key, value in meta[idy][idx].items():
                        dd[key].extend(value)

                for k, v in dd.items():
                    dd[k] = np.array(v, dtype="object")[
                        list(self.df[channel_name[idx]])
                    ]

                dictionaries_.append(dd)
            dictionaries.append(dictionaries_)

        # quick and dirty hack to get this to work for multiple instruments
        # required_keys = list(
        #     set().union(
        #         *[
        #             d[0].keys() for d in dictionaries
        #         ]
        #     )
        # )

        dnr_arr = []
        for sad, dicn in zip(self.all_data, dictionaries):

            # different keys
            required_keys = dicn[0].keys()

            dnr = [{k: [] for k in required_keys} for _ in range(sad.shape[0])]
            for i in range(sad.shape[0]):  # items in dataset
                # need to test, but should put NaN where the values
                # aren't shared between two instruments

                for key in required_keys:
                    for d_ in dicn:
                        try:
                            val = d_[key][i]
                        except Exception:
                            val = pd.NA

                        dnr[i][key].append(val)

            dnr_arr.append(dnr)

        return dnr_arr

    def get_cotemporal_data(self) -> List[da.array]:
        """
        Function to return co-temporal data

        Returns
        -------
        dnr_arr: List[da.array]
            A list of dask arrays for each channel containing
            the respective metadata for that instrument

        """
        data_all_inst = []
        for inst, channel_name in zip(self.loaded_data, self.channels):
            concat_data = []
            # iterate through years
            for idx in range(len(channel_name)):

                im_ = da.concatenate(
                    [inst[j][idx] for j in range(len(inst))], axis=0
                )

                concat_data.append(
                    im_[list(self.df[channel_name[idx]].to_numpy())]
                )
            data_all_inst.append(da.stack(concat_data, axis=1))

        return data_all_inst

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx) -> List:
        try:
            # This will take a while the first time a chunk is accessed;
            # this will then be cached upto the max_cache_size
            data_items = [
                torch.from_numpy(np.array(d[idx])).unsqueeze(dim=0)
                for d in self.all_data
            ]

            meta_items = [d[idx] for d in self.all_meta]

            return data_items, meta_items

        except Exception as error:
            logging.error(error)

    def _select_times(self, freq: str) -> pd.date_range:
        """
        Generate ``pd.date_range`` based on the provided years, ``self._years``
        """
        # !TODO modify this for sitatuions where years aren't contiguous
        return pd.date_range(
            start=date(int(self._years[0]), 1, 1),
            end=date(int(self._years[-1]), 12, 31),
            freq=freq,
            tz="utc",
        )

    def _check_selected_times(self):
        raise NotImplementedError


if __name__ == "__main__":

    import timeit

    s = timeit.default_timer()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s  %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    start = timeit.default_timer()

    sdomlds = SDOMLDataset(
        cache_max_size=1 * 512 * 512 * 4096,
        years=["2010", "2011"],
        data_to_load={
            "HMI": {
                "storage_location": "gcs",
                "root": "fdl-sdoml-v2/sdomlv2_hmi.zarr/",
                "channels": ["Bx", "By", "Bz"],
            },  # 12 minute cadence
            "AIA": {
                "storage_location": "gcs",
                "root": "fdl-sdoml-v2/sdomlv2_small.zarr/",
                "channels": ["94A", "131A", "171A", "193A", "211A", "335A"],
            },  # 6 minute cadence
            "EVE": {
                "storage_location": "gcs",
                "root": "fdl-sdoml-v2/sdomlv2_eve.zarr/",
                "channels": ["O V", "Mg X", "Fe XI"],
            },  # 1 minute candece
        },
    )
    end = timeit.default_timer()

    logging.info(f" pd.DataFrame \n {pformat(sdomlds.df)} \n ------------ ")
    # -- Logging
    logging.info(f"time taken to run: {end-start} seconds")
    logging.info(f"Dataset length, ``sdomlds.__len__()``: {sdomlds.__len__()}")

    # Second time requesting an this item will be quicker due to the cache
    for i in ["first", "second"]:
        start = timeit.default_timer()
        _ = sdomlds.__getitem__(0)[0]
        end = timeit.default_timer()
        logger.info(
            f"{i} ``sdomlds.__getitem__(0)`` request took {end-start} seconds"
        )

    logging.info(f"``sdomlds.channels``: {sdomlds.channels}")

    logging.info(
        f"``Shape of a single item: sdomlds.__getitem__(0)[0]``: {[sdomlds.__getitem__(0)[0][q].shape for q in range(len(sdomlds.__getitem__(0)[0]))]}"
    )

    logging.info(
        f"``Number of keys: sdomlds.__getitem__(0)[1]``: {[len(sdomlds.__getitem__(0)[1][q]) for q in range(len(sdomlds.__getitem__(0)[1]))]}"
    )
