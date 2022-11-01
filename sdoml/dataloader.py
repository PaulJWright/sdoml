import logging
import sys
from collections import defaultdict
from datetime import date
from pprint import pformat
from typing import Dict, List, Optional

import dask.array as da
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sdoml.sources.dataset_factory import DataSource

__all__ = ["SDOMLDataset"]

# -- Setting up logging
logger = logging.getLogger(__name__)
logging.getLogger("sdoml").addHandler(logging.NullHandler())


class SDOMLDataset(Dataset):
    """
    Dataset class for the SDOML v2.+ (`.zarr`) data.

    Parameters
    ----------

    cache_max_size : Optional[Union[int, None]]
        The maximum size that the ``zarr`` cache may grow to,
        in number of bytes. By default this variable is 1 * 512 * 512 * 2048.
        If the variable is set to ``None``, the cache will have unlimited size.
        see https://zarr.readthedocs.io/en/stable/api/storage.html#zarr.storage.LRUStoreCache

    years : Optional[Union[List[str], None]]
        A list of years to include. By default this
        variable is ``2010`` which will return data for 2010 only.

    freq : Optional[Union[str, None]]
        A string representing the frequency at which data should be obtained.
        See [here](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases)
        for a list of frequency aliases. By default this is ``120T`` (120 minutes)

    data_to_load : List[``~sdoml.sources.dataset_factory.DataSource``]
        A list of ``~sdoml.sources.dataset_factory.DataSource``

    Examples
    --------

    .. code-block:: python

        sdomlds = SDOMLDataset(
            cache_max_size=1 * 512 * 512 * 4096,
            years=["2010", "2011"],
            data_to_load=[...]
            )
    """

    def __init__(
        self,
        cache_max_size: Optional[int] = 1 * 512 * 512 * 2048,
        years: Optional[List[str]] = None,
        freq: str = "120T",
        data_to_load: List[DataSource] = None,
    ):

        # !TODO implement passing of ``selected_times`` and ``required_keys``
        selected_times = None
        # required_keys = None

        if years is None:
            years = ["2010"]

        self._cache_max_size = cache_max_size
        self._single_cache_max_size = self._cache_max_size / len(data_to_load)
        self._years = years
        self._meta = data_to_load

        # instantiate the appropriate classes
        data_arr = data_to_load  # !TODO remove before MR
        # data_arr = [
        #     DataSource(k, v, self._years, self._single_cache_max_size)
        #     for k, v in data_to_load.items()
        # ]

        # !TODO rearrange data for cadence (lowest to highest)

        if selected_times:
            # check the provided selected times are in agreement with the data
            # self.selected_times = self._check_selected_times(selected_times)
            raise NotImplementedError
        else:
            self._selected_times = self._select_times(freq)

        # Create a ``pd.DataFrame`` for the set of selected_times
        _df = pd.DataFrame(
            self._selected_times,
            index=np.arange(np.shape(self._selected_times)[0]),
            columns=["selected_times"],
        )

        for darr in data_arr:
            darr.set_years_channels()
            darr.load_data_meta()

        self._loaded_data, self._loaded_meta, _ = zip(
            *[darr.data_meta_time for darr in data_arr]
        )

        self._channels = [darr.available_channels for darr in data_arr]

        # Go through the time component of the loaded_data,
        # match to ``_df.selected_times``, and delete any rows that have NaN.
        # Doing this (especially when data is ordered by cadence) reduces
        # the number of potential matches for the next set of data.
        for i, darr in enumerate(data_arr):
            if i == 0:
                self._df = darr.get_cotemporal_indices(_df, "selected_times")
            else:
                self._df = darr.get_cotemporal_indices(
                    self._df, "selected_times"
                )

            self._df.dropna(inplace=True)
            self._df.reset_index(drop=True, inplace=True)

        # utilise cotemporal indices to get cotemporal data / metadata
        self._all_data = self._get_cotemporal_data()
        self._all_meta = self._get_cotemporal_meta()

    def _get_cotemporal_meta(self) -> List[Dict]:
        """
        Function to return co-temporal metadata

        Returns
        -------
        dnr_arr: List[Dict]
            A list of dictionaries for each channel containing
            the respective metadata for that instrument
        """
        dictionaries = []

        for meta, channel_name in zip(self._loaded_meta, self._channels):
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
                        list(self._df[channel_name[idx]])
                    ]

                dictionaries_.append(dd)
            dictionaries.append(dictionaries_)

        dnr_arr_all = []

        for sad, dicn in zip(self._all_data, dictionaries):
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

                # dnr_arr.append(dnr)
            dnr_arr = [str(dnr_i) for dnr_i in dnr]
            dnr_arr_all.append(dnr_arr)

        return dnr_arr_all

    def _get_cotemporal_data(self) -> List:
        """
        Function to return co-temporal data

        Returns
        -------
        dnr_arr: List[da.array]
            A list of dask arrays for each channel containing
            the respective metadata for that instrument

        """
        data_all_inst = []
        for inst, channel_name in zip(self._loaded_data, self._channels):
            concat_data = []
            # iterate through years
            for idx in range(len(channel_name)):

                im_ = da.concatenate(
                    [inst[j][idx] for j in range(len(inst))], axis=0
                )

                concat_data.append(
                    im_[list(self._df[channel_name[idx]].to_numpy())]
                )
            data_all_inst.append(da.stack(concat_data, axis=1))

        return data_all_inst

    def __len__(self):
        return len(self._df.index)

    def __getitem__(self, idx) -> List:
        try:
            # This will take a while the first time a chunk is accessed;
            # this will then be cached upto the max_cache_size
            data_items = [
                torch.from_numpy(np.array(d[idx]))
                # .unsqueeze(dim=0) to convert to 1 x H x W, to be in
                # compatible torchvision format
                for d in self._all_data
            ]

            data_items_dict = dict(zip(self._meta.keys(), data_items))

            meta_items = [d[idx] for d in self._all_meta]
            meta_items_dict = dict(zip(self._meta.keys(), meta_items))

            return {"data": data_items_dict, "meta": meta_items_dict}

        except Exception as error:
            logging.error(error)

    def _select_times(self, freq: str):
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

    @property
    def selected_times(self):
        """
        return the list of (selected) times that have been requested
        """
        return self._selected_times

    @property
    def dataframe(self):
        """
        return the final ``pd.DataFrame`` that is used
        """
        return self._df

    @property
    def all_data(self):
        """
        return the data for each channel.
        ``.__getitem__(index)`` will return a single index of ``all_data``
        """
        return self._all_data

    @property
    def all_meta(self):
        """
        return the metadata for each channel.
        ``.__getitem__(index)`` will return a single index of ``all_meta``
        """
        return self._all_meta

    @property
    def available_channels(self):
        """
        return the list of available channels in the data
        """
        return self._channels


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
                "root": "fdl-sdoml-v2/sdomlv2_hmi_small.zarr/",
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

    logging.info(f" sdomlds.dataframe \n {pformat(sdomlds.dataframe)} \n \n")
    logging.info(f"time taken to run: {end-start} seconds")
    logging.info(f"Dataset length, ``sdomlds.__len__()``: {sdomlds.__len__()}")

    # If caching is implemented, the second request on an index will be quicker.
    for i in ["first", "second"]:
        start = timeit.default_timer()
        _ = sdomlds.__getitem__(0)
        end = timeit.default_timer()
        logger.info(
            f"{i} ``sdomlds.__getitem__(0)`` request took {end-start} seconds"
        )
