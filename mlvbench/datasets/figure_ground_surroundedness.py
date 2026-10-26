"""Figure-Ground Surroundedness Data Module."""

from typing import Any

import numpy as np

from mlvbench.cache import PrecomputedDataset
from mlvbench.datasets._base import ProceduralDataModule
from mlvbench.models import Model
from mlvbench.stimuli.shapes import random_idsprite, split_figure
from mlvbench.stimuli.textures import load_textures, texturize


class FigureGroundSurroundednessDataModule(ProceduralDataModule):
    """Data module for testing surroundedness as a figure-ground cue.

    Each image contains a random foreground shape from the idsprites dataset. The
    foreground and background regions are filled with different textures from the DTD
    dataset.
    """

    name = "figure_ground_surroundedness"
    version = "2.0.0-dev"

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
    }

    def __init__(self, seed: int = 0, margin: float = 0.01):
        """Initialize the data module.

        Args:
            seed: Random seed for generating the stimuli.
            margin: Minimum distance between the figure and the image border, as a
                fraction of the image size. Guarantees that the figure is surrounded.
        """
        super().__init__()
        self.seed = seed
        self.margin = margin
        self.num_samples = {
            "train": 10000,
            "val": 1000,
            "test": 1000,
        }

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
            "margin": self.margin,
        }

    def before_generate(self, seed: int) -> None:
        """Load the DTD textures."""
        self.textures = load_textures("dtd", seed)

    def generate_sample(self, subset: str, seed: int) -> dict[str, np.ndarray]:
        """Generate a single figure-ground surroundedness sample."""
        rng = np.random.default_rng(seed=seed)
        shape_seed, texture_seed = rng.integers(0, 2**32, size=2)
        shape_rng = np.random.default_rng(seed=shape_seed)
        texture_rng = np.random.default_rng(seed=texture_seed)

        # Sample shape area and center coverage such that both are uninformative about
        # which region is foreground: There's a 50% chance of covering the center. If
        # the center is covered, the shape is larger than the background are and vice
        # versa.
        covers_center = bool(shape_rng.random() < 0.5)
        low, high = (
            (0.50, 0.60) if covers_center else (0.08, 0.25)
        )
        area = shape_rng.uniform(low, high)

        segmentation = random_idsprite(
            self.image_size,
            rng=shape_rng,
            area=area,
            covers_center=covers_center,
            center_radius=0.02,
            margin=self.margin,
        )
        segmentation = (segmentation > 0).astype(np.uint8)
        textures = self.textures[subset]
        image = texturize(segmentation, textures, rng=texture_rng)
        sample = {"image": image, "segmentation": segmentation}

        if subset == "test":
            # Reinitialize the texture RNG, so that we sample the same textures for the
            # reversed image.
            texture_rng = np.random.default_rng(seed=texture_seed)
            image_reversed = texturize(1 - segmentation, textures, rng=texture_rng)
            sample["image_reversed"] = image_reversed

            # Split the foreground into two sub-regions, each with a different texture,
            # while keeping the segmentation label unchanged (single foreground region).
            texture_rng = np.random.default_rng(seed=texture_seed)
            split_seed = rng.integers(0, 2**32, size=1)
            split_rng = np.random.default_rng(seed=split_seed)
            split_region_map = split_figure(
                segmentation, split_rng, min_region_size=0.25
            )
            image_split = texturize(split_region_map, textures, rng=texture_rng)
            sample["image_split"] = image_split
            sample["segmentation_split"] = split_region_map

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
