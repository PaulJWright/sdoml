import copy
from email.policy import default
import sys

import logging
import timeit
import torch
import zarr

import dask.array as da
import numpy as np
import pandas as pd

from collections import defaultdict
from torch.utils.data import Dataset
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
        instruments: Optional[List[str]] = ["AIA"],
        channels: Optional[List[str]] = None,
        required_keys: Optional[List[str]] = None,
        selected_times=None,
    ):

        if len(instruments) != 1 and instruments[0] != "AIA":
            raise ValueError()

        # if channels is not None:
        #     for channel in channels:

        #     self.channels
        # else:
        #     self.channels = None

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
        logging.info(">>> setting up cache")
        cache = zarr.LRUStoreCache(store, max_size=cache_max_size)

        logging.info(">>> opening zarr")
        start = timeit.default_timer()
        self.root = zarr.open(store=cache, mode="r") # this takes a while....
        end = timeit.default_timer()
        logging.info(f">>> time to zarr.open: {end-start}")


        # Reduce data based on years
        if years is not None:
            if is_str_list(years):
                # get the ``zarr.heirachy.Group`` for each year, 
                # and remove years that don't exist (returned None)
                # import timeit
                # start = timeit.default_timer()
                # zarr_yr = [self.root.get(y) for y in years] 
                # by_year = list(filter(None, zarr_yr))
                # end = timeit.default_timer()
                # print(end-start) 
                # 74.6 s 
                # for zarr_yr = [None,
                #                <zarr.hierarchy.Group '/2010' read-only>, 
                #                <zarr.hierarchy.Group '/2011' read-only>]

                start = timeit.default_timer()
                # for the same command, the following takes 0.0004 s
                zarr_yr = [self.root.get(y) for y in years]
                # find locations using string
                zarr_yr_str = [str(i) for i in zarr_yr]
                index_to_keep = [x for x, _ in enumerate(zarr_yr_str) if _ != 'None']
                by_year = [zarr_yr[i] for i in index_to_keep]
                end = timeit.default_timer()
                logging.info(f"time to downsample yr (~0.0004s ??): {end-start}")

                logger.info('list of channels provided')
                if len(by_year) != len(years):
                    logger.warning(f'Not all of {years} are avaiable from {zarr_root}...')
                    logger.warning(f'... returning {by_year}')
            else:
                raise ValueError()
        else:
            by_year = [group for _, group in self.root.groups()]
            logger.info('list of channels not provided')


        # print('by_year:', by_year)

        # Reduce data based on channels
        if channels is not None:
            if is_str_list(channels):
                data = [
                    group.get(channel)
                    for group in by_year
                    for channel in channels
                ]
                
                self.data = list(filter(None, data)) # why is this so slow?

                if len(self.data) != (len(channels)*len(by_year)):
                    logger.warning(f'Not all of {channels} are avaiable from {zarr_root}...')
                    logger.warning(f'...returning {self.data}')
            else:
                raise ValueError()
        else:
            self.data = [g for y in by_year for _, g in y.arrays()]

        # print('self.data', self.data)

        if selected_times is None:
            # one of the selected channels may have less images than another
            self._min_val, self._min_index = get_minvalue(
                [d.shape[0] for d in self.data]
            )

            # print("min, index", self._min_val, self._min_index)
            # take times from the array with the least number of obs.

            if len(by_year) == 1:
                t_obs_new = self.data[self._min_index].attrs["T_OBS"]

                selected_times = pd.date_range(
                    start=sorted(t_obs_new)[0].replace('TAI','Z').replace('_', 'T').replace(':60T',':59T'),
                    end=sorted(t_obs_new)[-1].replace('TAI','Z').replace('_', 'T').replace(':60T',':59T'),
                    freq="6T",
                )

                # print(sorted(t_obs_new)[0], sorted(t_obs_new)[-1])

                self.chunked_list = [self.data]
            else:
                # this is probably insanely inefficient
                chunked_list = list()
                chunk_size = len(channels)
                for i in range(0, len(self.data), chunk_size):
                    chunked_list.append(data[i:i+chunk_size])
                
                if instruments == ["AIA"]:
                    selected_times = pd.date_range(
                        # start=sorted(chunked_list[0][0].attrs["T_OBS"])[0],
                        # end=sorted(chunked_list[-1][0].attrs["T_OBS"])[-1],
                        start="2010-05-01T00:00:00.00Z",
                        end="2015-12-31T23:59:59.00Z",
                        freq="6T",
                    )
                else:
                    selected_times = pd.date_range(
                        # start=sorted(chunked_list[0][0].attrs["T_OBS"])[0],
                        # end=sorted(chunked_list[-1][0].attrs["T_OBS"])[-1],
                        start="2010.05.01T00:00:00",
                        end="2015.12.31T23:59:59",
                        freq="6T",
                    )

                # logging.info(f'selected times {selected_times}')

                self.chunked_list = chunked_list
        else:
            raise NotImplementedError

        # Generate a ``pd.DataFrame`` of the indices for the
        # selected times and channels
        
        start = timeit.default_timer()
        df = self._get_cotemporal_data(selected_times)
        end = timeit.default_timer()
        logging.info(f"time to self._get_cotemporal_data: {end-start}")

        logging.info(f'The dataframe is of shape {df.shape}')
        print(f'df {df}')
        # -- Obtain the image data

        images = []
        # iterate through channel, i; year, j
        start = timeit.default_timer()
        for i in range(len(self.chunked_list[0])): # channels [94, 131]
            im_ = da.concatenate(
                [self.chunked_list[j][i] for j in range(len(self.chunked_list))], 
                axis=0
                )

            zarr_imgs = im_[
                list(df[get_aia_channel_name(self.chunked_list[0][i])].to_numpy())
            ]
            images.append(zarr_imgs)

        # all images are stored in ``self.all_images``
        self.all_images = da.stack(images, axis=1)
        end = timeit.default_timer()
        print('time taken to combine images: ', end-start)

        start = timeit.default_timer()
        # array to hold channels
        dictins = []
        for channel in range(len(self.chunked_list[0])): # looping through channels
            dd = defaultdict(list)
            dicts = [self.chunked_list[j][channel] 
                    for j in range(len(self.chunked_list))
                    ] # combines the dictionary for all years
            for d in dicts: # you can list as many input dicts as you want here
                for key, value in d.attrs.items():
                    dd[key].extend(value) # this should be for all years

            dictins.append(dd)

        end = timeit.default_timer()
        logging.info(f"combining metadata channels in year: {end-start}")
        # dictins[0] -- 094 for 2010, 2011
        # dictins[1] -- 131 for 2010, 2011

        start = timeit.default_timer()
        for i in range(len(dictins)):
            for key, value in dictins[i].items():
                dictins[i][key] = np.array(value, dtype='object')[
                    list(df[
                        get_aia_channel_name(self.chunked_list[0][i])
                        ])
                    ] # if dtype isn't defined, we get issues with creating a
                    # sunpy.map where the CTYPE1/CTYPE2 is numpy.str_
                    # https://stackoverflow.com/questions/61403530/conversion-of-numpy-str-to-byte-or-string-datatype-error
        end = timeit.default_timer()
        logging.info(f"downsampling dictionary based on cotemporal indices: {end-start}")

        if required_keys is None:
            required_keys = list(self.data[0].attrs.keys())

        logging.info(f'self.all_images.shape {self.all_images.shape}')

        start = timeit.default_timer()
        dnr = [{k: [] for k in required_keys} for _ in range(self.all_images.shape[0])]
        for i in range(self.all_images.shape[0]): # items in dataset
            for d_ in dictins: # channels
                for key, value in d_.items():
                    dnr[i][key].append(value[i])
        end = timeit.default_timer()
        logging.info(f"combining into a list of item-specific dictionaries {end-start}")
        logging.info(f'time taken to combine into a list of item-specific dictionaries {end-start}')
        # all metadata is stored in ``self.attrs``
        self.attrs = dnr
        self.data_len = len(self.all_images)

        unique_len = np.unique([x.shape[0] for x in self.all_images])

        if len(np.unique([x.shape[0] for x in self.all_images])) == 1:
            logging.info(
                f"There are {len(self.all_images)} observations, each with {unique_len[0]} channels"
            )
        else:
            msg = "Not cotemporal observations have the same shape"
            raise ValueError(msg)

    def _get_images(self):
        pass

    def _get_meta(self):
        pass

    def _get_cotemporal_data(
        self,
        selected_times: pd.date_range,
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
            selected_times,
            index=np.arange(np.shape(selected_times)[0]),
            columns=["selected_times"],
        )

        # iterate through channels, finding indices that match the times
        # selected. This is required as the SDOML v2.+ data doesn't necessarily
        # have every channel for each timestep.
        
        # for i, channel in enumerate(self.data[0]):
        #     # i is the channel index, of the first year (self.data[0])
        #     selected_index = []

        #     # extract the 'T_OBS' from the data that exists.

        #     arr = []
        #     for idx in range(len(self.data)):
        #         print(self.data[idx][i])
        #         arr.extend(self.data[idx][i].attrs["T_OBS"])

        #     pd_series = pd.to_datetime(arr)

        # print('self.chunked_list: ', self.chunked_list)

        for i, channel in enumerate(self.chunked_list[0]): # self.data):
            s = timeit.default_timer()
            selected_index = []

            # extract the 'T_OBS' from the data that exists.
            # pd_series = pd.to_datetime(self.data[i].attrs["T_OBS"])

            arr = [] 
            s = timeit.default_timer()
            for j in range(len(self.chunked_list)):
                arr.extend(self.chunked_list[j][i].attrs["T_OBS"])
            logging.info(f'time to extract T_TOBS; {timeit.default_timer() - s} seconds')

            s = timeit.default_timer()
            pd_series = pd.to_datetime([item.replace('TAI','Z').replace('_', 'T').replace(':60T',':59T') for item in arr])
            logging.info(f'time to fix T_TOBS + create pd.to-datetime; {timeit.default_timer() - s} seconds')

            # loop through ``selected_time`` finding the closest match.
            s = timeit.default_timer()
            # this is a very expensive operaton
            for time in selected_times:
                selected_index.append(np.argmin(abs(time - pd_series)))
            logging.info(f'time to find closest match; {timeit.default_timer() - s} seconds')

            # for all matches, flag missing where the offset is > ``timedelta``
            # and set to NaN
            s = timeit.default_timer()
            missing_index = np.where(
                np.abs(pd_series[selected_index] - selected_times)
                > pd.Timedelta(timedelta)
            )[0].tolist()
            for midx in missing_index:
                # if there is a missing_index, set to NaN
                selected_index[midx] = pd.NA
            logging.info(f'time to find missing matches; {timeit.default_timer() - s} seconds')

            # insert a new row into the main ``pd.DataFrame`` for the channel
            s = timeit.default_timer()
            df.insert(i + 1, get_aia_channel_name(channel), selected_index)
            logging.info(f'inserted {channel}; {timeit.default_timer() - s} seconds')

        # drop all rows with a NaN, and reset the index
        df.dropna(inplace=True)
        df.reset_index(inplace=True)

        # ``pd.DataFrame`` with cotemporal observations across all channels in
        # the range of selected_times.
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

    sdos = timeit.default_timer()

    zr = "fdl-sdoml-v2/sdomlv2_small.zarr/"
    sdomlds = SDOMLDataset(
        storage_location="gcs",
        zarr_root=zr,
        cache_max_size=1*512*512*4096,
        years=["2010", "2011", "2012"], # 2009 doesn't exist in this data
        channels=["94A", "193A"], # 312 doesn't exist as an SDO channel
    )

    sdoe = timeit.default_timer()
    logging.info(f"time taken to run {zr} TOTAL {sdoe-sdos}")

    # -- Logging
    logging.info(f"Dataset length, ``sdomlds.__len__()``: {sdomlds.__len__()}")
    logging.info(
        f"``Shape of a single item: sdomlds.__getitem__(0)[0].shape``: {sdomlds.__getitem__(0)[0].shape}"
    )
    logging.info(
        f"``Number of keys: len(sdomlds.__getitem__(0)[1])``: {len(sdomlds.__getitem__(0)[1])}"
    )
