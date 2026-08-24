"""Dataset base classes."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mlvbench.cache import (
    PrecomputedDataset,
    create_cached_resource,
    finalize_cached_resource,
    get_cached_resource,
    precompute_dataset,
)
from mlvbench.models import Model


class DataModule:
    """Base class for data modules.

    A data module encapsulates all data processing steps for a dataset. It is inspired
    by the [DataModule API in PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/data/datamodule.html).
    """

    def configure(self, model: Model) -> None:
        """Configure the data module for the given model.

        Called by the trainer before [`prepare_data`][]. Override to store
        model-specific attributes (e.g., input size) that are needed during data
        preparation or dataset construction.

        Args:
            model: The model to configure the data module for.
        """

    def prepare_data(self) -> None:
        """Download, preprocess or generate the stimuli."""

    def setup(self) -> None:
        """Perform additional setup after prepare_data (defaults to no-op)."""
        pass

    def subsets(self) -> list[str]:
        """Return the names of all available subsets.

        Returns:
            The names of all available subsets.
        """
        return ["train", "val", "test"]

    def train_dataset(self) -> Dataset:
        """Return the training dataset.

        This method is called by the default implementation of `train_dataloader()`.
        Override this method to provide the training dataset, or override
        `train_dataloader()` to build a custom data loader.
        """
        raise NotImplementedError(
            "Override train_dataset() to provide a training dataset, or override " +
            "train_dataloader() to build a custom data loader."
        )

    def val_dataset(self) -> Dataset:
        """Return the validation dataset.

        This method is called by the default implementation of `val_dataloader()`.
        Override this method to provide the validation dataset, or override
        `val_dataloader()` to build a custom data loader.
        """
        raise NotImplementedError(
            "Override val_dataset() to provide a validation dataset, or override " +
            "val_dataloader() to build a custom data loader."
        )

    def test_dataset(self, subset: str = "test") -> Dataset:
        """Return the requested test dataset.

        This method is called by the default implementation of `test_dataloader()`.
        Override this method to provide the test dataset, or override
        `test_dataloader()` to build a custom data loader.

        Args:
            subset: The name of the test subset to construct, as returned by
                `subsets()`.
        """
        raise NotImplementedError(
            "Override test_dataset() to provide a test dataset, or override " +
            "test_dataloader() to build a custom data loader."
        )

    def train_dataloader(
        self,
        batch_size: int,
        num_workers: int = 0,
        repeat: bool = True,
        device: str | torch.device = "cpu",
    ) -> Iterable[dict]:
        """Return the data loader for the training set."""
        return self._build_dataloader(
            self.train_dataset(),
            batch_size,
            shuffle=True,
            repeat=repeat,
            num_workers=num_workers,
            device=device,
        )

    def val_dataloader(
        self,
        batch_size: int,
        num_workers: int = 0,
        device: str | torch.device = "cpu",
    ) -> Iterable[dict]:
        """Return the data loader for the validation set."""
        return self._build_dataloader(
            self.val_dataset(),
            batch_size,
            shuffle=False,
            repeat=False,
            num_workers=num_workers,
            device=device,
        )

    def test_dataloader(
        self,
        batch_size: int,
        num_workers: int = 0,
        device: str | torch.device = "cpu",
        subset: str = "test",
    ) -> Iterable[dict]:
        """Return the data loader for the requested test subset.

        Args:
            batch_size: The batch size.
            num_workers: The number of worker processes.
            device: The device the batches are placed on.
            subset: The name of the test subset to construct, as returned by
                `subsets()`.
        """
        return self._build_dataloader(
            self.test_dataset(subset),
            batch_size,
            shuffle=False,
            repeat=False,
            num_workers=num_workers,
            device=device,
        )

    @staticmethod
    def _build_dataloader(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        repeat: bool = False,
        num_workers: int = 0,
        device: str | torch.device = "cpu",
    ) -> Iterable[dict]:
        """Build a data loader for the given dataset."""
        if isinstance(device, str):
            device = torch.device(device)

        if (
            isinstance(dataset, PrecomputedDataset)
            and not dataset.mmap
            and device.type == "cuda"
        ):
            dataloader = OnDeviceDataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, device=device
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
            )
            dataloader = _to_device(dataloader, device=device)

        if repeat:
            return _repeat(dataloader)
        else:
            return dataloader


def _to_device(dataloader: Iterable, device: torch.device) -> Iterable:
    """Move all tensors in the batches to the given device."""
    for batch in dataloader:
        batch_on_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_on_device[key] = value.to(device, non_blocking=True)
            else:
                batch_on_device[key] = value
        yield batch_on_device


def _repeat(dataloader: Iterable) -> Iterable:
    """Yield batches from `dataloader` indefinitely, reshuffling each epoch.

    `itertools.cycle` would cache the first epoch's batches and replay that exact
    sequence forever, which causes any "hard" batch to recur at fixed step
    intervals and shows up as periodic loss spikes. Re-iterating the DataLoader
    instead lets its `RandomSampler` reshuffle each epoch.
    """
    while True:
        yield from dataloader


class ProceduralDataModule(DataModule, ABC):
    """Base class for data modules that generate stimuli procedurally."""

    @abstractmethod
    def _parameters(self) -> dict[str, Any]:
        """Return the configuration parameters used for caching the dataset.

        Called by [`prepare_data`][] after [`configure`][] has stored any
        model-specific attributes. Previously generated data will be reused if the
        same parameters are returned.

        Returns:
            A dictionary of configuration parameters used for caching the dataset.
                Previously generated data will be reused if the same configuration is
                used. The following keys are required:

            - "name": The name of the data module.
            - "version": The version of the data module.
            - "seed": The seed used to generate the dataset.
            - "num_samples": A dictionary of the number of samples to generate for each
                subset. The keys must be "train", "val", and "test".
        """

    def prepare_data(self) -> None:
        """Generate and cache the dataset subsets."""
        parameters = self._parameters()
        name = parameters.pop("name")
        version = parameters.pop("version")
        seed = parameters["seed"]
        num_samples = parameters["num_samples"]

        if set(num_samples.keys()) != {"train", "val", "test"}:
            raise ValueError(
                "num_samples must contain the keys 'train', 'val', and 'test'."
            )

        path = get_cached_resource(name, version, parameters)

        if path is not None:
            self._path = path
            return

        path = create_cached_resource(name, version, parameters)

        # We generate the preparation seed before generating the samples, such that the
        # preperation seed doesn't change when the number of samples change.
        rng = np.random.default_rng(seed=seed)
        before_generate_seed = rng.integers(0, 2**32).item()
        sample_seeds = self.generate_sample_seeds(num_samples, rng)

        self.before_generate(before_generate_seed)

        for subset, subset_sample_seeds in sample_seeds.items():
            self.generate_subset(subset, subset_sample_seeds, path)

        self.after_generate()

        self._path = finalize_cached_resource(path)

    @staticmethod
    def generate_sample_seeds(
        num_samples: dict[str, int],
        rng: np.random.Generator,
    ) -> dict[str, list[int]]:
        """Generate the sample seeds."""
        subsets = num_samples.keys()
        subset_rngs = rng.spawn(len(subsets))
        return {
            subset: subset_rng.integers(0, 2**32, size=num_samples[subset])
            for subset, subset_rng in zip(subsets, subset_rngs, strict=True)
        }

    def before_generate(self, seed: int) -> None:
        """Set up resources before generating the samples."""
        pass

    def generate_subset(self, subset: str, sample_seeds: list[int], path: Path) -> None:
        """Generate the samples for the given subset."""
        generate_sample = self.generate_sample

        class _Subset(Dataset):
            def __init__(self, sample_seeds: list[int]):
                self.sample_seeds = sample_seeds

            def __len__(self):
                return len(self.sample_seeds)

            def __getitem__(self, index: int) -> dict[str, np.ndarray]:
                seed = self.sample_seeds[index]
                sample = generate_sample(subset, seed)
                sample["__key__"] = str(index)
                return sample

        dataset = _Subset(sample_seeds)
        precompute_dataset(dataset, path / subset)

    @abstractmethod
    def generate_sample(self, subset: str, seed: int) -> dict[str, np.ndarray]:
        """Generate a sample for the given subset."""
        pass

    def after_generate(self) -> None:
        """Clean up resources after generating the samples."""
        pass

    def train_dataset(self) -> PrecomputedDataset:
        """Return the training split."""
        if not hasattr(self, "_path"):
            raise ValueError("Data module not prepared. Call prepare_data() first.")
        return PrecomputedDataset(self._path / "train", mmap=False)

    def val_dataset(self) -> PrecomputedDataset:
        """Return the validation split."""
        if not hasattr(self, "_path"):
            raise ValueError("Data module not prepared. Call prepare_data() first.")
        return PrecomputedDataset(self._path / "val", mmap=False)

    def test_dataset(self, subset: str = "test") -> PrecomputedDataset:
        """Return the test split."""
        if subset != "test":
            raise ValueError(f"Unknown test subset: {subset}")
        if not hasattr(self, "_path"):
            raise ValueError("Data module not prepared. Call prepare_data() first.")
        return PrecomputedDataset(self._path / "test", mmap=False)


class OnDeviceDataLoader:
    """DataLoader for datasets on the GPU."""

    def __init__(
        self,
        dataset: PrecomputedDataset,
        device: torch.device,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        """Initialize the data loader."""
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

        for key in dataset.buffers:
            self.dataset.buffers[key] = self.dataset.buffers[key].to(device)

        if shuffle:
            self.sampler = torch.utils.data.RandomSampler(dataset)
        else:
            self.sampler = torch.utils.data.SequentialSampler(dataset)

        self.sampler = torch.utils.data.BatchSampler(
            self.sampler, batch_size, drop_last
        )

    def __len__(self) -> int:
        """Return the number of batches in this data loader."""
        return len(self.sampler)

    def __iter__(self) -> Generator[dict, None, None]:
        """Yield the batches in this data loader."""
        for indices in self.sampler:
            index_tensor = torch.tensor(indices, device=self.device)
            sample = {
                key: buffer[index_tensor]
                for key, buffer in self.dataset.buffers.items()
            }
            sample["__key__"] = [self.dataset.keys[i] for i in indices]
            yield sample
