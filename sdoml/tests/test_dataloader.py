import pytest
import torch

from sdoml import SDOMLDataset
from sdoml.sources import DataSource

data_to_load = {
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
}

datasource_arr = [
    DataSource(instrument=k, meta=v) for k, v in data_to_load.items()
]


def test_sdomldataset():
    sdomlds = SDOMLDataset(
        cache_max_size=1 * 512 * 512 * 4096,
        years=["2010"],
        data_to_load=datasource_arr,
    )

    sdomlds.__getitem__(0)


# def test_sdomldataset():
#     import pandas as pd

#     sdomlds = SDOMLDataset(
#         cache_max_size=1 * 512 * 512 * 4096,
#         years=["2010"],
#         data_to_load=datasource_arr,
#     )

#     assert type(sdomlds.dataframe()) is pd.DataFrame
#     sdomlds.all_data()
#     sdomlds.all_meta()
#     sdomlds.available_channels()


def test_sdomldataset_noyears():
    sdomlds = SDOMLDataset(
        cache_max_size=1 * 512 * 512 * 4096,
        # years=["2010"],
        data_to_load=datasource_arr,
    )
    assert sdomlds._years is not None


def test_sdomldataset_nodata():
    with pytest.raises(TypeError):
        _ = SDOMLDataset(
            cache_max_size=1 * 512 * 512 * 4096,
            years=["2010"],
            # data_to_load=datasource_arr,
        )


def test_sdomldataset_nolist():
    with pytest.raises(TypeError):
        _ = SDOMLDataset(
            cache_max_size=1 * 512 * 512 * 4096,
            years=["2010"],
            data_to_load=datasource_arr[0],
        )
