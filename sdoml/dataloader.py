import sys
import timeit

import logging
import zarr
import torch

import dask.array as da
import numpy as np
import pandas as pd

from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from typing import List, Set, Dict, Tuple, Optional

# -- Setting up logging
logger = logging.getLogger(__name__)
logging.getLogger("sdoml").addHandler(logging.NullHandler())


def is_str_list(val: List[object]):  # -> Boolean():
    """Determines whether all objects in the list are strings"""
    return all(isinstance(x, str) for x in val)


def get_minvalue(inputlist):

    # get the minimum value in the list
    min_value = min(inputlist)

    # return the index of minimum value
    min_index = inputlist.index(min_value)
    return min_value, min_index


def common_entries(*dcts):
    if not dcts:
        return
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)


class SDOMLDataset(Dataset):
    """
    Dataset class for the SDOML v2.+ (`.zarr`) data.

    Parameters
    -------------
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
        channels: Optional[List[str]] = None,
        required_keys: Optional[List[str]] = [
            "T_OBS",
            "EXPTIME",
            "WAVELNTH",
            "WAVEUNIT",
            "DEG_COR",
        ],
        selected_times=None,
    ):

        if storage_location == "gcs":
            import gcsfs

            store = gcsfs.GCSMap(
                zarr_root,
                gcs=gcsfs.GCSFileSystem(access="read_only"),
                check=False,
            )
        else:
            raise NotImplementedError

        # !TODO understand if we need to cache this...
        cache = zarr.LRUStoreCache(store, max_size=cache_max_size)
        self.root = zarr.open(store=cache, mode="r")

        if years is not None:
            if is_str_list(channels):
                # TODO this can return None
                by_year = [self.root.get(y) for y in years]
            else:
                raise ValueError()
        else:
            by_year = [group for _, group in self.root.groups()]

        if channels is not None:
            if is_str_list(channels):
                self.data = [
                    group.get(channel)
                    for group in by_year
                    for channel in channels
                ]
            else:
                raise ValueError()
        else:
            self.data = [g for y in by_year for _, g in y.arrays()]

        # one channel may have less images than another
        self._min_val, self._min_index = get_minvalue(
            [d.shape[0] for d in self.data]
        )

        if selected_times is None:
            # take times from the array with the least number of obs.
            t_obs_new = self.data[self._min_index].attrs["T_OBS"]

            # set start/end as limits of sorted arrays, and take values
            # at a frequency of 6 minutes
            selected_times = pd.date_range(
                start=sorted(t_obs_new)[0],
                end=sorted(t_obs_new)[-1],
                freq="6T",
            )

        df = self._get_cotemporal_data(selected_times)

        images = []
        # -- Obtain the image data
        # --
        for zarray in self.data:
            name = df[str(zarray).split("/")[2].split("'")[0]]
            zarr_imgs = da.from_array(zarray)[list(name.to_numpy())]
            print(
                f"zarray.shape, {zarray.shape} len zarr_imgs, {len(zarr_imgs)}"
            )
            images.append(zarr_imgs)
        self.all_images = da.stack(images, axis=1)

        # -- Obtain the image keys in a similar format
        # !TODO Figure out a better way of doing this
        att_arr = []
        for j, _ in enumerate(zarr_imgs):
            # create an empty dictionary
            dnr = {k: [] for k in required_keys}
            for zarray in self.data:
                name = str(zarray).split("/")[2].split("'")[0]
                # fill dictionary with keys from each channel of data
                [
                    dnr[k].append(zarray.attrs[k][df[name][j]])
                    for k in required_keys
                ]
            # append the observation-time dictionary to the final array
            att_arr.append(dnr)
        self.attrs = att_arr

        self.data_len = len(self.all_images)
        logging.info(f"There are {len(self.all_images)} observations")

    def _get_cotemporal_data(self, selected_times):
        """
        Function to return co-temporal data across channels
        """

        # Initialise ``pd.Dataframe`` with
        # the times we requre observations for
        df = pd.DataFrame(
            selected_times,
            index=np.arange(np.shape(selected_times)[0]),
            columns=["selected_times"],
        )

        # iterate through channels, finding
        for i, channel in enumerate(self.data):
            selected_index = []

            pd_df = pd.to_datetime(self.data[i].attrs["T_OBS"])

            for time in selected_times:
                selected_index.append(np.argmin(abs(time - pd_df)))

            missing_index = np.where(
                np.abs(pd_df[selected_index] - selected_times)
                > pd.Timedelta("3m")
            )[0].tolist()
            for midx in missing_index:
                selected_index[midx] = pd.NA

            name = str(channel).split("/")[2].split("'")[0]
            df.insert(i + 1, name, selected_index)

        df.dropna(inplace=True)

        return df

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        try:
            # obtain shape of [idx, channels, dim_1, dim_2]
            item = torch.from_numpy(
                np.array(self.all_images[idx, :, :, :])
            ).unsqueeze(dim=0)
            meta = self.attrs[idx]
            return item, meta

        except Exception as error:
            logging.error(error)


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s  %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    sdomlds = SDOMLDataset(
        storage_location="gcs",
        zarr_root="fdl-sdoml-v2/sdomlv2_small.zarr/",
        cache_max_size=None,
        years=["2010"],
        channels=["131A", "193A", "94A", "171A", "211A"],
    )

    # -- Logging
    logging.info(
        f"The `.zarr` directory structure is: \n {sdomlds.root.tree()}"
    )
    logging.info(f"Dataset length, ``sdomlds.__len__()``: {sdomlds.__len__()}")
    logging.info(
        f"``Shape of a single item: sdomlds.__getitem__(0)[0].shape``: {sdomlds.__getitem__(0)[0].shape}"
    )
    logging.info(
        f"``Number of keys: len(sdomlds.__getitem__(0)[1])``: {len(sdomlds.__getitem__(0)[1])}"
    )
