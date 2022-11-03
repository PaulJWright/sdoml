"""
A set of utility functions
"""

import os
from typing import List, Optional, Tuple, Union

import gcsfs
import numpy as np
import zarr

__all__ = [
    "gcs_conn",
    "load_single_gcs_zarr",
    "inspect_single_gcs_zarr",
    # "load_single_zarr",
    # "inspect_single_zarr",
]


def gcs_conn(path_to_zarr: os.path) -> gcsfs.GCSMap:
    """
    Instantiate connection to gcs for a given path ``path_to_zarr``
    """
    return gcsfs.GCSMap(
        path_to_zarr,
        gcs=gcsfs.GCSFileSystem(access="read_only"),
        check=False,
    )


def load_single_gcs_zarr(
    path_to_zarr: os.path,
    cache_max_single_size: int = None,
) -> Union[zarr.Array, zarr.Group]:
    """
    load zarr from gcs using LRU cache
    """
    return zarr.open(
        zarr.LRUStoreCache(
            store=gcs_conn(path_to_zarr),
            max_size=cache_max_single_size,
        ),
        mode="r",
    )


def inspect_single_gcs_zarr(
    path_to_zarr: os.path,
) -> Union[zarr.Array, zarr.Group]:
    """
    load zarr from gcs *without* using cache
    """
    return zarr.open(store=gcs_conn(path_to_zarr), mode="r")


# def load_single_zarr(
#     # Figure out why I can't use cache here...
#     path_to_zarr: os.path,
#     # cache_max_single_size: int = None,
# ) -> Union[zarr.Array, zarr.Group]:
#     """
#     load zarr from gcs using LRU cache
#     """
#     return zarr.open(
#         store=path_to_zarr,
#         mode="r",
#     )


# def inspect_single_zarr(
#     path_to_zarr: os.path,
# ) -> Union[zarr.Array, zarr.Group]:
#     """
#     load zarr from gcs *without* using cache
#     """
#     return zarr.open(store=path_to_zarr, mode="r")


def is_str_list(val: List[object]) -> bool:
    """
    Determines whether all objects in the list are strings
    """
    return all(isinstance(x, str) for x in val)


def get_minvalue(inputlist: List) -> Tuple[float, int]:
    """
    Function to return min. value and corresponding index
    """
    # get the minimum value in the list
    min_value = min(inputlist)
    # return the index of minimum value
    min_index = inputlist.index(min_value)
    return (min_value, min_index)


def solve_list(a: List, b: List) -> List:
    """
    sort list ``a`` based on the order of items in list ``b``
    https://stackoverflow.com/questions/30504317/sort-a-subset-of-a-python-list-to-have-the-same-relative-order-as-in-other-list
    """
    dct = {x: i for i, x in enumerate(b)}
    items_in_a = [x for x in a if x in dct]
    items_in_a.sort(key=dct.get)
    it = iter(items_in_a)
    return [next(it) if x in dct else x for x in a]
