import contextlib
import os
import sys

import gcsfs
import logging
import timeit
import torch
import zarr

import dask.array as da
import numpy as np
import pandas as pd

from collections import defaultdict
from datetime import date
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List, Optional

# -- Setting up logging
logger = logging.getLogger(__name__)
logging.getLogger("sdoml").addHandler(logging.NullHandler())


def is_str_list(val: List[object]):  # -> Boolean():
    """Determines whether all objects in the list are strings"""
    return all(isinstance(x, str) for x in val)


def get_minvalue(inputlist):
    """Function to return min. value and corresponding index"""
    # get the minimum value in the list
    min_value = min(inputlist)
    # return the index of minimum value
    min_index = inputlist.index(min_value)
    return min_value, min_index


def get_aia_channel_name(inputzarr):
    """Function to return aia channel name as string from zarr array"""
    return str(inputzarr).split("/")[2].split("'")[0]


# def catch(func, handle=lambda e : e, *args, **kwargs):
#     try:
#         return func(*args, **kwargs)
#     except Exception as e:
#         return handle(e)


class SDOMLDataset(Dataset):
    """
    Dataset class for the SDOML v2.+ (`.zarr`) data.

    Parameters
    ----------
    storage_location : str, optional
        Storage location of the root ``.zarr`` file`. This variable is set
        to ``gcs`` by default

        Options :
            - ``gcs`` : Google Cloud Storage

    zarr_root : str, optional
        Location of the root ``.zarr`` file within the ``storage_location``.
        By default this is ``fdl-sdoml-v2/sdomlv2_small.zarr/`` (which is
        located on Google Cloud Storage ``storage_location == gcs``).

    cache_max_size: int, Nonetype, optional
        The maximum size that the ``zarr`` cache may grow to,
        in number of bytes. By default this variable is 1 * 512 * 512 * 2048.
        If the variable is set to ``None``, the cache will have unlimited size.
        see https://zarr.readthedocs.io/en/stable/api/storage.html#zarr.storage.LRUStoreCache

    years: List[str], Nonetype, optional
        A list of years (from 2010 to present) to include. By default this
        variable is ``None`` which will return all years.

    instruments: List[str], optional
        A list of instruments to include. By default this
        variable is ``[``AIA``]`` which will provide data for AIA.

        Options :
            - ``AIA`` : SDO Atomospheric Imaging Assembly
            - ``HMI`` : SDO Helioseismic and Magnetic Imager
            - ``EVE`` : Extreme UltraViolet Variability Experiment

    channels: List[str], Nonetype, optional
        A list of SDO/AIA channels to include from
        ["94A", "131A", "211A", "304A", "335A", "1600A", "1700A"].
        By default this variable is ``None``, and will return all channels.

    required_keys: List[str], optional
        A list of metadata keys to include. By default this variable includes
        ["T_OBS", "EXPTIME",  "WAVELNTH", "WAVEUNIT", "DEG_COR"].

    selected_times: ...
        ...

    """

    def __init__(
        self,
        storage_location: str = "gcs",
        zarr_root: str = "fdl-sdoml-v2/sdomlv2_small.zarr/",
        cache_max_size: Optional[int] = 1 * 512 * 512 * 2048,
        years: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
        required_keys: Optional[List[str]] = None,
        selected_times=None,
    ):

        # !TODO allow variations on these
        if (
            storage_location != "gcs"
            or len(instruments) != 1
            or zarr_root != "fdl-sdoml-v2/sdomlv2_small.zarr/"
            or instruments is None
            or required_keys
        ):
            raise NotImplementedError

        # setting variables
        self.zarr_root = zarr_root
        self._cache_max_size = cache_max_size
        self._storage_location = storage_location

        # extract the set of available years/channels from the provided data
        yr_channel_dict = self._get_years_channels(years, channels)
        self.years = list(yr_channel_dict.keys())
        # for now assume that dictionary keys are of the same size; take zeroth
        self.channels = list(yr_channel_dict.values())[0]
        # The below is good, but doesn't preserve order.
        # self.channels = list(set.intersection(*[set(x) for x in yr_channel_dict.values()]))

        if selected_times:
            # check the provided selected times are in agreement with the data
            self.selected_times = self._check_selected_times(selected_times)
        else:
            self.selected_times = self._select_times()

        # Use the LRUStoreCache, on a channel/year basis...
        self.chunked_list = self._load_data()
        self.df = self._get_cotemporal_data()

        # data used for dataloader
        self.all_images = self._get_all_images()
        self.attrs = self._get_dictins()

    def show_params(self):
        pass

    def _get_years_channels(self, yrs, chnnls):

        # check the data has the correct years
        sorted_yrs = sorted(yrs)
        # create year/channel dictionary
        yc_dict = {}

        # go through years, and channels ensuring we can read the data
        for year in sorted_yrs:
            for i, channel in enumerate(chnnls):
                with contextlib.suppress(Exception):
                    store = gcsfs.GCSMap(
                        os.path.join(zr, year, channel),
                        gcs=gcsfs.GCSFileSystem(access="read_only"),
                        check=False,
                    )

                    # cache = zarr.LRUStoreCache(store, max_size=None)
                    # using ``zarr.LRUStoreCache`` is useful, but slows down the code,
                    # when we just want to check what the groups are (not access arrays)

                    _ = zarr.open(store, mode="r")  # cache,

                    if i == 0:
                        yc_dict[year] = []

                    yc_dict[year].append(channel)
        # check the data has the correct channels for all years
        return yc_dict

    def _check_selected_times(self, select_t):

        # Only checking the start and end times align with the data
        # as people may request e.g. 2010, 2012, 2014

        # check the min/max of the years are available (a quick sanity check)
        assert select_t[0].year == int(self.years[0])
        assert select_t[-1].year == int(self.years[-1])

        return select_t

    def _check_required_keys(self):
        pass

    def _load_data(self):

        by_year = []
        for c in self.channels:
            ch = []
            for y in self.years:
                store = gcsfs.GCSMap(
                    os.path.join(self.zarr_root, y, c),
                    gcs=gcsfs.GCSFileSystem(access="read_only"),
                    check=False,
                )

                cache = zarr.LRUStoreCache(
                    store,
                    max_size=(
                        self._cache_max_size
                        / (len(self.channels) * len(self.years))
                    ),
                )

                ch.append(zarr.open(cache, mode="r"))
            by_year.append(ch)

        return by_year

    def _get_cotemporal_data(
        self,
        # selected_times: pd.date_range,
        timedelta: str = "3m",
    ) -> pd.DataFrame():
        """
        Function to return co-temporal data across channels

        Parameters
        ----------
        selected_times : ``pd.date_range``
            ``pd.date_range`` object.

            ```
            selected_times = pd.date_range(
                start='2010-08-01T00:00:00',
                end='2010-08-01T23:59:59',
                freq="6T",
            )
            ```

        td : str, Optional:
            the maximum "time delta" that indicates if a given observation time
            matches the selected time. By default this is 3 minutes ("3m").
            The SDO/AIA data has been generated at "6m" cadence.

        Returns
        -------

        df : ``pd.DataFrame`` with cotemporal observations across all
            channels in the range of selected_times.


        """

        # Initialise ``pd.Dataframe`` with the times we requre observations for
        df = pd.DataFrame(
            self.selected_times,
            index=np.arange(np.shape(self.selected_times)[0]),
            columns=["selected_times"],
        )

        # iterate through channels, finding indices that match the times
        # selected. This is required as the SDOML v2.+ data doesn't necessarily
        # have every channel for each timestep.

        for i, _ in enumerate(
            tqdm(
                self.chunked_list,
                desc="iterating through channels",
                leave=None,
            )
        ):  # self.data):
            # extract the 'T_OBS' from the data that exists.
            arr = []
            for j in tqdm(
                range(len(self.chunked_list[0])),
                desc="combining seperate years",
                leave=None,
            ):
                arr.extend(self.chunked_list[i][j].attrs["T_OBS"])

            pd_series = pd.to_datetime(arr)

            selected_index = [
                np.argmin(abs(time - pd_series))
                for time in tqdm(
                    self.selected_times,
                    desc="finding matching indices",
                    leave=None,
                )
            ]

            # for all matches, flag missing where the offset is > ``timedelta``
            # and set to NaN
            missing_index = np.where(
                np.abs(pd_series[selected_index] - self.selected_times)
                > pd.Timedelta(timedelta)
            )[0].tolist()
            for midx in tqdm(
                missing_index, desc="removing missing indices", leave=None
            ):
                # if there is a missing_index, set to NaN
                selected_index[midx] = pd.NA

            # insert a new row into the main ``pd.DataFrame`` for the channel
            df.insert(i + 1, self.channels[i], selected_index)

        # drop all rows with a NaN, and reset the index
        df.dropna(inplace=True)
        df.reset_index(inplace=True)

        # ``pd.DataFrame`` with cotemporal observations across all channels in
        # the range of selected_times.
        return df

    def __len__(self):
        return self.all_images.shape[0]

    def __getitem__(self, idx):
        try:
            # obtain shape of [idx, channels, dim_1, dim_2]
            # This will take a while the first time a chunk is accessed;
            # this will then be cached upto the max_cache_size
            item = torch.from_numpy(
                np.array(self.all_images[idx, :, :, :])
            ).unsqueeze(dim=0)
            meta = self.attrs[idx]
            return item, meta

        except Exception as error:
            logging.error(error)

    def _get_all_images(self):
        images = []
        # iterate through channel, i; year, j
        for i in range(len(self.chunked_list)):  # channels [94, 131]
            im_ = da.concatenate(
                [
                    self.chunked_list[i][j]
                    for j in range(len(self.chunked_list[0]))
                ],
                axis=0,
            )

            zarr_imgs = im_[list(self.df[self.channels[i]].to_numpy())]
            images.append(zarr_imgs)

        return da.stack(images, axis=1)

    def _get_dictins(self):
        dictins = []
        for channel in range(
            len(self.chunked_list)
        ):  # looping through channels
            dd = defaultdict(list)
            for d in self.chunked_list[
                channel
            ]:  # you can list as many input dicts as you want here
                for key, value in d.attrs.items():
                    dd[key].extend(value)  # this should be for all years
            dictins.append(dd)

        for i in range(len(dictins)):
            for key, value in dictins[i].items():
                dictins[i][key] = np.array(value, dtype="object")[
                    list(self.df[self.channels[i]])
                ]

        required_keys = list(self.chunked_list[0][0].attrs.keys())

        dnr = [
            {k: [] for k in required_keys}
            for _ in range(self.all_images.shape[0])
        ]
        for i in range(self.all_images.shape[0]):  # items in dataset
            for d_ in dictins:  # channels
                for key, value in d_.items():
                    dnr[i][key].append(value[i])

        return dnr

    def _select_times(self):
        freq = "12T"  # 4 hours

        s = date(int(self.years[0]), 1, 1)
        e = date(int(self.years[-1]), 12, 31)
        pd_dr = pd.date_range(
            start=s,
            end=e,
            freq=freq,
            tz="utc",
        )

        logging.info(
            f"selected times range from {s} to {e} at a frequency of {freq}"
        )

        return pd_dr


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s  %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    start = timeit.default_timer()
    zr = "fdl-sdoml-v2/sdomlv2_small.zarr/"
    sdomlds = SDOMLDataset(
        storage_location="gcs",
        zarr_root=zr,
        cache_max_size=1 * 512 * 512 * 4096,
        years=[
            "2009",
            "2010",
            "2011",
            "2012",
        ],  # 2009 doesn't exist in this data
        channels=[
            "94A",
            "131A",
            "171A",
            "193A",
            "211A",
            "335A",
        ],  # 312 doesn't exist as an SDO channel
        instruments=["AIA"],
    )

    end = timeit.default_timer()
    logging.info(f"time taken to run {zr} TOTAL {end-start}")

    # -- Logging
    logging.info(f"Dataset length, ``sdomlds.__len__()``: {sdomlds.__len__()}")

    for i in ["first", "second"]:
        start = timeit.default_timer()
        logging.info(
            f"``Shape of a single item: sdomlds.__getitem__(0)[0].shape``: {sdomlds.__getitem__(0)[0].shape}"
        )
        end = timeit.default_timer()
        logger.info(
            f"{i} ``sdomlds.__getitem__(0)`` request took {end-start} seconds"
        )

    logging.info(
        f"``Number of keys: len(sdomlds.__getitem__(0)[1])``: {len(sdomlds.__getitem__(0)[1])}"
    )
