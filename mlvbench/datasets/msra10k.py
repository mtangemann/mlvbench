"""MSRA-10K dataset."""

import logging
import zipfile
from pathlib import Path

import PIL.Image
import torch
from torch.utils.data import Dataset, random_split
from torchvision.transforms import Compose

from mlvbench.cache import (
    PrecomputedDataset,
    create_cached_resource,
    finalize_cached_resource,
    get_cached_resource,
    precompute_dataset,
)
from mlvbench.datasets import DataModule
from mlvbench.download import download
from mlvbench.models import Model
from mlvbench.stimuli.transforms import (
    BinarizeSegmentation,
    CenterCrop,
    Resize,
    ToNumpy,
    ToTensor,
)

LOGGER = logging.getLogger(__name__)


_MSRA10K_URL = "http://mftp.mmcheng.net/Data/MSRA10K_Imgs_GT.zip"


class MSRA10KDataModule(DataModule):
    """Data module for the MSRA-10K dataset."""

    name = "msra10k"
    version = "1.0.1"

    def __init__(self, seed: int = 0, to_numpy: bool = False) -> None:
        """Initialize the data module.

        Args:
            seed: Seed used for splitting the dataset into train, val, and test subsets.
            to_numpy: Whether to convert the tensors to numpy arrays.
        """
        super().__init__()
        self.seed = seed
        self.to_numpy = to_numpy

    def configure(self, model: Model) -> None:
        """Configure the data module for the given model."""
        self.image_size = model.input_size

    def prepare_data(self) -> None:
        """Download the dataset."""
        self.path = download(_MSRA10K_URL)

    def setup(self) -> None:
        """Create the train/val/test index splits."""
        transforms = [
            ToTensor(),
            BinarizeSegmentation(),
            CenterCrop(),
            Resize(self.image_size),
        ]
        if self.to_numpy:
            transforms.append(ToNumpy())
        dataset = MSRA10KDataset(self.path, transform= Compose(transforms))
        generator = torch.Generator().manual_seed(self.seed)
        self._train, self._val, self._test = random_split(
            dataset, [0.8, 0.1, 0.1], generator=generator
        )

    def train_dataset(self):
        """Return the training split."""
        return self._train

    def val_dataset(self):
        """Return the validation split."""
        return self._val

    def test_dataset(self, subset: str = "test"):
        """Return the test split."""
        if subset != "test":
            raise ValueError(f"Unknown test subset: {subset}")
        return self._test


class PrecomputedMSRA10KDataModule(DataModule):
    """Precomputed data module for the MSRA-10K dataset."""

    name = "msra10k"
    version = "2.0.1"

    def __init__(self, seed: int = 0) -> None:
        """Initialize the data module.

        Args:
            seed: Seed used for splitting the dataset into train, val, and test subsets.
        """
        self.seed = seed

    def configure(self, model: Model) -> None:
        """Configure the data module for the given model."""
        self.image_size = model.input_size

    def prepare_data(self) -> None:
        """Precompute and cache the dataset."""
        parameters = {
            "seed": self.seed,
            "image_size": self.image_size,
        }

        path = get_cached_resource(self.name, self.version, parameters)
        if path is not None:
            self._path = path
            return

        msra10k = MSRA10KDataModule(self.seed, to_numpy=True)
        msra10k.image_size = self.image_size
        msra10k.prepare_data()
        msra10k.setup()

        path = create_cached_resource(self.name, self.version, parameters)

        precompute_dataset(msra10k.train_dataset(), path / "train")
        precompute_dataset(msra10k.val_dataset(), path / "val")
        precompute_dataset(msra10k.test_dataset(), path / "test")

        self._path = finalize_cached_resource(path)

    def train_dataset(self):
        """Return the training split."""
        return PrecomputedDataset(self._path / "train", mmap=False)

    def val_dataset(self):
        """Return the validation split."""
        return PrecomputedDataset(self._path / "val", mmap=False)

    def test_dataset(self, subset: str = "test"):
        """Return the test split."""
        if subset != "test":
            raise ValueError(f"Unknown test subset: {subset}")
        return PrecomputedDataset(self._path / "test", mmap=False)


class MSRA10KDataset(Dataset):
    """Dataset for the MSRA-10K dataset."""

    def __init__(self, path: Path, transform: callable = None):
        """Initialize the dataset.

        Args:
            path: Path to the dataset zip archive.
            transform: Optional transform applied to each sample dict.
        """
        self.path = path
        self.transform = transform
        self.archive = zipfile.ZipFile(path)
        self.image_paths = sorted(
            entry for entry in self.archive.namelist() if entry.endswith(".jpg")
        )

    def __del__(self) -> None:
        """Close the archive."""
        self.archive.close()

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        """Return the image and mask at the given index.

        Returns:
            A dictionary with keys "__key__" (image file stem), "image" (RGB PIL Image)
                and "segmentation" (PIL Image). If a transform is set, it is applied to
                the sample.
        """
        image_path = self.image_paths[index]
        mask_path = image_path[:-4] + ".png"

        with self.archive.open(image_path) as f:
            image = PIL.Image.open(f).convert("RGB")
            image.load()
        with self.archive.open(mask_path) as f:
            segmentation = PIL.Image.open(f).convert("L")
            segmentation.load()

        sample = {
            "__key__": Path(image_path).stem,
            "image": image,
            "segmentation": segmentation,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
