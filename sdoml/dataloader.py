from torch.utils.data import DataLoader, Dataset, random_split
import zarr
import logging
import sys

# -- Set up loogging
logger = logging.getLogger(__name__)
logging.getLogger("sdoml").addHandler(logging.NullHandler())


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
        located at ``storage_location`` == ``gcs``.

    cache_max_size: int, Nonetype, optional
        The maximum size that the ``zarr`` cache may grow to, 
        in number of bytes. If ``None`` the cache will have unlimited size.
    """

    def __init__(
        self,
        storage_location: str = "gcs",
        zarr_root: str = "fdl-sdoml-v2/sdomlv2_small.zarr/",
        cache_max_size: int = 1 * 1024 * 1024 * 2048,
    ):

        print(cache_max_size)

        if storage_location == "gcs":
            import gcsfs

            store = gcsfs.GCSMap(
                zarr_root,
                gcs=gcsfs.GCSFileSystem(access="read_only"),
                check=False,
            )
        else:
            raise NotImplementedError

        cache = zarr.LRUStoreCache(store, max_size=cache_max_size)
        self.root = zarr.open(store=cache, mode="r")

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s  %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    sdomldl = SDOMLDataset(
        storage_location="gcs",
        zarr_root="fdl-sdoml-v2/sdomlv2_small.zarr/",
        cache_max_size=None,
    )

    logging.info(
        f"The `.zarr` directory structure is: \n {sdomldl.root.tree()}"
    )
