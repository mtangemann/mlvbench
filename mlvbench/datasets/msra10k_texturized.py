"""Data modules for testing figure-ground shape cues."""

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from mlvbench.cache import PrecomputedDataset, precompute_dataset
from mlvbench.datasets import MSRA10KDataModule
from mlvbench.datasets._base import ProceduralDataModule
from mlvbench.models import Model
from mlvbench.stimuli.colors import colorize
from mlvbench.stimuli.shapes import split_figure
from mlvbench.stimuli.textures import load_textures, texturize


class TexturizedMSRA10KDataModule(ProceduralDataModule):
    """Data module filling MSRA-10K segmentation masks with DTD textures or colors."""

    name = "msra10k_texturized"
    version = "1.0.0-dev"

    # All test subsets are views onto the same generated test split and differ only in
    # the image variant they read. `texture_segmentation` is the region map that was
    # used for texturizing the image.
    _TEST_FEATURES = {
        "test": {"image": "image", "segmentation": "segmentation"},
        "test_reversed": {"image": "image_reversed", "segmentation": "segmentation"},
        "test_split": {
            "image": "image_split",
            "segmentation": "segmentation",
            "texture_segmentation": "segmentation_split",
        },
        "test_color": {"image": "image_color", "segmentation": "segmentation"},
        "test_color_reversed": {
            "image": "image_color_reversed",
            "segmentation": "segmentation",
        },
        "test_color_split": {
            "image": "image_color_split",
            "segmentation": "segmentation",
            "texture_segmentation": "segmentation_split",
        },
    }

    def __init__(self, seed: int = 0):
        """Initialize the data module.

        Args:
            seed: Seed used for reproducibility.
        """
        super().__init__()
        self.seed = seed

    def configure(self, model: Model) -> None:
        """Configure the data module for the given model."""
        self.image_size = model.input_size

    def _parameters(self) -> dict[str, Any]:
        """Return the cache parameters for this data module."""
        return {
            "name": self.name,
            "version": self.version,
            "seed": self.seed,
            "num_samples": self.num_samples,
            "image_size": self.image_size,
        }

    def prepare_data(self) -> None:
        """Set up the MSRA-10K source data and generate the texturized dataset."""
        # Reusing the natural data module verbatim is what guarantees that the masks,
        # the split assignment and the order within each split match the natural
        # condition exactly.
        source = MSRA10KDataModule(seed=self.seed, to_numpy=True)
        source.image_size = self.image_size
        source.prepare_data()
        source.setup()

        # The insertion order must match the order in which `generate_sample_seeds`
        # spawns the per-subset generators.
        self._splits = {
            "train": source.train_dataset(),
            "val": source.val_dataset(),
            "test": source.test_dataset(),
        }
        self.num_samples = {
            subset: len(split) for subset, split in self._splits.items()
        }

        super().prepare_data()

    def before_generate(self, seed: int) -> None:
        """Load the DTD textures."""
        self.textures = load_textures("dtd", seed)

    def generate_subset(self, subset: str, sample_seeds: list[int], path: Path) -> None:
        """Generate the samples for the given subset.

        This dataset is not purely procedural since but derived from the original
        MSRA10K dataset. So we need to customize this function.
        """
        split = self._splits[subset]
        generate_sample = self.generate_sample

        class _Subset(Dataset):
            def __init__(self, sample_seeds: list[int]):
                self.sample_seeds = sample_seeds

            def __len__(self):
                return len(self.sample_seeds)

            def __getitem__(self, index: int) -> dict[str, np.ndarray]:
                source = split[index]
                sample = generate_sample(subset, self.sample_seeds[index], source)
                sample["__key__"] = source["__key__"]
                return sample

        precompute_dataset(_Subset(sample_seeds), path / subset)

    def generate_sample(
        self,
        subset: str,
        seed: int,
        source: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Texturize a single MSRA-10K sample.

        The signature is widened by `source` compared to the base class.
        `generate_subset` above is the only caller.

        Args:
            subset: The name of the subset the sample belongs to.
            seed: The sample seed drawn by [`ProceduralDataModule.prepare_data`][].
            source: The MSRA-10K sample providing the segmentation mask.

        Returns:
            The texturized sample.
        """
        rng = np.random.default_rng(seed=seed)
        # The shape seed is unused here because the mask is given, but it must still be
        # drawn so that we use the same texture seed as the other procedural datasets.
        _shape_seed, texture_seed = rng.integers(0, 2**32, size=2)
        texture_rng = np.random.default_rng(seed=texture_seed)
        split_seed = rng.integers(0, 2**32, size=1)
        split_rng = np.random.default_rng(seed=split_seed)

        segmentation = source["segmentation"]
        textures = self.textures[subset]

        if subset == "test":
            # For the test set, we consider all colorization and texturization variants
            # for each example.
            image = texturize(segmentation, textures, rng=texture_rng)
            sample = {"image": image, "segmentation": segmentation}

            # Reinitialize the texture RNG, so that we sample the same textures for the
            # reversed image.
            texture_rng = np.random.default_rng(seed=texture_seed)
            image_reversed = texturize(1 - segmentation, textures, rng=texture_rng)
            sample["image_reversed"] = image_reversed

            # Split the foreground into two sub-regions, each with a different texture,
            # while keeping the segmentation label unchanged (single foreground region).
            texture_rng = np.random.default_rng(seed=texture_seed)
            split_region_map = split_figure(
                segmentation, split_rng, min_region_size=0.25
            )
            image_split = texturize(split_region_map, textures, rng=texture_rng)
            sample["image_split"] = image_split
            sample["segmentation_split"] = split_region_map

            # The color variants reuse the texture seed, so a region keeps its
            # color across the variants just like it keeps its texture.
            color_rng = np.random.default_rng(seed=texture_seed)
            sample["image_color"] = colorize(segmentation, rng=color_rng)

            color_rng = np.random.default_rng(seed=texture_seed)
            sample["image_color_reversed"] = colorize(1 - segmentation, rng=color_rng)

            color_rng = np.random.default_rng(seed=texture_seed)
            sample["image_color_split"] = colorize(split_region_map, rng=color_rng)

        else:
            # For the training and validation set, we randomly select a colorization or
            # texturization variant.
            texture_segmentation = segmentation
            if rng.random() < 1/3:
                texture_segmentation = split_figure(
                    segmentation, split_rng, min_region_size=0.25
                )
            if rng.random() < 0.5:
                image = texturize(texture_segmentation, textures, rng=texture_rng)
            else:
                image = colorize(texture_segmentation, rng=texture_rng)

            sample = {"image": image, "segmentation": segmentation}

        return sample

    def after_generate(self) -> None:
        """Clean up the textures after generating the samples."""
        del self.textures

    def subsets(self) -> list[str]:
        """Return the names of all available subsets."""
        return ["train", "val", *self._TEST_FEATURES]

    def test_dataset(self, subset: str = "test") -> PrecomputedDataset:
        """Return the requested test split (standard, reversed, or figure-split)."""
        if subset not in self._TEST_FEATURES:
            raise ValueError(f"Unknown test subset: {subset}")
        return PrecomputedDataset(
            self._path / "test",
            features=self._TEST_FEATURES[subset],
            mmap=False,
        )
