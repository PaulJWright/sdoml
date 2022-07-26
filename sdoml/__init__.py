# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .version import __version__
from .dataloader import SDOMLDataset

# Then you can be explicit to control what ends up in the namespace,
__all__ = ["SDOMLDataset"]
