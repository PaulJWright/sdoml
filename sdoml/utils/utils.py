"""
A set of utility functions
"""

import os

import gcsfs
import zarr

from typing import List, Optional, Union, Tuple

__all__ = ["gcs_conn", "load_single_gcs_zarr", "inspect_single_gcs_zarr"]


def gcs_conn(path_to_zarr: os.path) -> gcsfs.GCSMap:
    """
    Instantiate connection to gcs for a given path ``ptz``
    """
    return gcsfs.GCSMap(
        path_to_zarr,
        gcs=gcsfs.GCSFileSystem(access="read_only"),
        check=False,
    )


def load_single_gcs_zarr(
    path_to_zarr: os.path,
    cache_max_single_size: int = None,
) -> Union[zarr.core.Array, zarr.hierarchy.Group]:
    """load zarr from gcs using LRU cache"""
    return zarr.open(
        zarr.LRUStoreCache(
            gcs_conn(path_to_zarr),
            max_size=cache_max_single_size,
        ),
        mode="r",
    )


def inspect_single_gcs_zarr(
    path_to_zarr: os.path,
) -> Union[zarr.core.Array, zarr.hierarchy.Group]:
    """load zarr from gcs *without* using cache"""
    return zarr.open(gcs_conn(path_to_zarr), mode="r")


def is_str_list(val: List[object]) -> bool:
    """Determines whether all objects in the list are strings"""
    return all(isinstance(x, str) for x in val)


def get_minvalue(inputlist: List) -> Tuple[float, int]:
    """Function to return min. value and corresponding index"""
    # get the minimum value in the list
    min_value = min(inputlist)
    # return the index of minimum value
    min_index = inputlist.index(min_value)
    return (min_value, min_index)
