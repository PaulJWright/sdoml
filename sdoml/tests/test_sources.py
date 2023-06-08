import pytest
from sunpy.map.map_factory import MultipleMatchError, NoMatchError

from sdoml.sources import SDOML_AIA, SDOML_HMI, DataSource


def test_sdoaiagcs_compliant():
    _ = DataSource(
        instrument="AIA",
        meta={
            "storage_location": "aws",
            "root": "s3://gov-nasa-hdrl-data1/contrib/fdl-sdoml/fdl-sdoml-v2/sdomlv2_small.zarr/",
        },
    )
    assert type(_) == SDOML_AIA


def test_sdoaiagcs_wrong_location():
    with pytest.raises(NoMatchError):
        _ = DataSource(
            instrument="AIA",
            meta={
                "storage_location": "gcs",
                "root": "s3://gov-nasa-hdrl-data1/contrib/fdl-sdoml/fdl-sdoml-v2/sdomlv2_small.zarr/",
            },
        )


def test_sdoaiagcs_wrong_root():
    with pytest.raises(NoMatchError):
        _ = DataSource(
            instrument="AIA",
            meta={
                "storage_location": "aws",
                "root": "s3://gov-nasa-hdrl-data1/contrib/fdl-sdoml/fdl-sdoml-v2/sdomlv2_test.zarr/",
            },
        )


def test_sdohmigcs_compliant():
    _ = DataSource(
        instrument="HMI",
        meta={
            "storage_location": "aws",
            "root": "s3://gov-nasa-hdrl-data1/contrib/fdl-sdoml/fdl-sdoml-v2/sdomlv2_hmi_small.zarr/",
        },
    )
    assert type(_) == SDOML_HMI


def test_sdohmigcs_wrong_location():
    with pytest.raises(NoMatchError):
        _ = DataSource(
            instrument="HMI",
            meta={
                "storage_location": "gcs",
                "root": "s3://gov-nasa-hdrl-data1/contrib/fdl-sdoml/fdl-sdoml-v2/sdomlv2_small.zarr/",
            },
        )


def test_sdohmigcs_wrong_root():
    with pytest.raises(NoMatchError):
        _ = DataSource(
            instrument="HMI",
            meta={
                "storage_location": "aws",
                "root": "s3://gov-nasa-hdrl-data1/contrib/fdl-sdoml/fdl-sdoml-v2/sdomlv2_test.zarr/",
            },
        )
