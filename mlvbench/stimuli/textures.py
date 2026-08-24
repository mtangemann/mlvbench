"""Texturization utilities."""

import logging
import tarfile
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from torch.utils.data import Dataset, Subset, random_split

from mlvbench.cache import (
    create_cached_resource,
    finalize_cached_resource,
    get_cached_resource,
)
from mlvbench.download import download

LOGGER = logging.getLogger(__name__)


class TextureDataset(Dataset):
    """Dataset of texture images."""

    def __init__(self, path: Path):
        """Initialize the dataset from a directory of *.jpg images."""
        self.paths = sorted(path.rglob("*.jpg"))

    def __len__(self) -> int:
        """Return the number of textures."""
        return len(self.paths)

    def __getitem__(self, index: int) -> PIL.Image.Image:
        """Return the texture at the given index as a PIL Image."""
        return PIL.Image.open(self.paths[index])


def load_textures(name: str, seed: int) -> dict[str, Subset]:
    """Load a texture dataset and return train, val, and test subsets.

    If not done already, this will download and extract the dataset. The textures are
    split into train, val, and test subsets (80%/10%/10%).

    Args:
        name: Name of the textures dataset. Only "dtd" is supported.
        seed: Random seed for splitting the dataset into train, val, and test subsets.

    Returns:
        A dictionary of subsets with keys "train", "val", and "test".

    Raises:
        ValueError: If the dataset name is not supported.
    """
    match name:
        case "dtd":
            path = _prepare_dtd()
        case _:
            raise ValueError(f"Unsupported texture dataset: {name}")

    dataset = TextureDataset(path)
    generator = torch.Generator().manual_seed(seed)
    train, val, test = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)
    return {
        "train": train,
        "val": val,
        "test": test,
    }


def texturize(
    region_map: np.ndarray,
    texture_dataset: Dataset,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fill each region in a region map with a unique random texture.

    Args:
        region_map: Integer-labeled region map of shape `(H, W, 1)` and dtype `uint8`.
        texture_dataset: Dataset of PIL Images. Each item must be a PIL Image
            that will be resized to match the region map dimensions. Must
            contain at least as many items as there are unique regions.
        rng: NumPy random number generator for reproducibility. Two calls with
            the same RNG state but different numbers of regions N and M will
            assign the same texture to the first `min(N, M)` regions.

    Returns:
        RGB image as uint8 NumPy array of shape `(H, W, 3)`.

    Raises:
        ValueError: If the number of unique regions exceeds the dataset size.
    """
    H, W = region_map.shape[:2]
    labels = np.unique(region_map[..., 0])
    num_regions = len(labels)

    if num_regions > len(texture_dataset):
        raise ValueError(
            f"Number of regions ({num_regions}) exceeds dataset size " +
            f"({len(texture_dataset)})"
        )

    # rng.choice would be more efficient, but is not guaranteed to sample the same first
    # textures when sampling different numbers of regions with the same RNG state.
    all_indices = np.arange(len(texture_dataset))
    rng.shuffle(all_indices)
    indices = all_indices[:num_regions]
    image = np.zeros((H, W, 3), dtype=np.uint8)

    for label, texture_index in zip(labels, indices, strict=True):
        texture = texture_dataset[int(texture_index)]
        if texture.size != (W, H):
            texture = texture.resize((W, H))
        texture_array = np.array(texture)
        mask = region_map[..., 0] == label
        image[mask] = texture_array[mask]

    return image


_DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
_DTD_VERSION = "2.0.0"


def _prepare_dtd() -> Path:
    """Download and extract the DTD dataset."""
    path = get_cached_resource("dtd", _DTD_VERSION, {})
    if path is not None:
        return path

    LOGGER.info(f"Preparing DTD dataset (v{_DTD_VERSION}) ...")
    path = create_cached_resource("dtd", _DTD_VERSION, {})

    archive_path = download(_DTD_URL)

    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            # All files are contained in a "dtd/" subdirectory. We extract them directly
            # into the cache directory.
            if member.name == "dtd":
                continue
            if member.name.startswith("dtd/"):
                member.name = member.name[len("dtd/"):]
            else:
                raise ValueError(f"Unexpected file in DTD archive: {member.name}")

            # Update the file timestamp so that the files get not cleaned up immediately
            # on time limited scratch storage.
            member.mtime = time.time()

            archive.extract(member, path, set_attrs=True, filter="data")

    path = finalize_cached_resource(path)
    LOGGER.info(f"Preparing DTD dataset (v{_DTD_VERSION}) completed")
    return path
