"""Figure-Ground Convexity Data Module."""

from typing import Any

import numpy as np

from mlvbench.cache import PrecomputedDataset
from mlvbench.datasets._base import ProceduralDataModule
from mlvbench.models import Model
from mlvbench.stimuli.shapes import split_figure
from mlvbench.stimuli.textures import load_textures, texturize


class FigureGroundConvexityDataModule(ProceduralDataModule):
    """Data module for testing convexity as a figure-ground cue.

    Each image is split into two regions by a parabolic curve. The convex region is
    labelled as the foreground, and the concave region as the background. Both regions
    are filled with different textures from the DTD dataset.
    """

    name = "figure_ground_convexity"
    version = "1.0.0"

    # All test subsets are views onto the same generated test split and differ only in
    # the image variant they read.
    _TEST_FEATURE_MAPS = {
        "test": {"image": "image", "segmentation": "segmentation"},
        "test_reversed": {"image_reversed": "image", "segmentation": "segmentation"},
        "test_split": {"image_split": "image", "segmentation": "segmentation"},
    }

    def __init__(self, seed: int = 0):
        """Initialize the data module."""
        super().__init__()
        self.seed = seed
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
        }

    def before_generate(self, seed: int) -> None:
        """Load the DTD textures."""
        self.textures = load_textures("dtd", seed)

    def generate_sample(self, subset: str, seed: int) -> dict[str, np.ndarray]:
        """Generate a single figure-ground convexity sample."""
        rng = np.random.default_rng(seed=seed)
        shape_seed, texture_seed = rng.integers(0, 2**32, size=2)
        shape_rng = np.random.default_rng(seed=shape_seed)
        texture_rng = np.random.default_rng(seed=texture_seed)

        segmentation = _random_arc_segmentation(self.image_size, rng=shape_rng)
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

        return sample

    def after_generate(self) -> None:
        """Clean up the textures after generating the samples."""
        del self.textures

    def subsets(self) -> list[str]:
        """Return the names of all available subsets."""
        return ["train", "val", *self._TEST_FEATURE_MAPS]

    def test_dataset(self, subset: str = "test") -> PrecomputedDataset:
        """Return the requested test split (standard, reversed, or figure-split)."""
        if subset not in self._TEST_FEATURE_MAPS:
            raise ValueError(f"Unknown test subset: {subset}")
        return PrecomputedDataset(
            self._path / "test",
            feature_map=self._TEST_FEATURE_MAPS[subset],
            mmap=False,
        )


def _random_arc_segmentation(
    image_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a binary segmentation map by splitting a square image with a bent arc.

    The arc is a parabola in a randomly oriented local coordinate system. The convex
    region (label 1) is the set of pixels above the parabola; the concave region
    (label 0) is the set of pixels below it.

    The parabola is vertically shifted so that exactly 50% of pixels fall in each
    region.

    Args:
        image_size: Side length of the square image in pixels.
        rng: Random number generator used for reproducibility.

    Returns:
        Segmentation map as uint8 numpy array of shape `(image_size, image_size, 1)`
            with values 0 (concave/background) and 1 (convex/foreground). Each region
            covers approximately 50% of the image.
    """
    coordinates = np.arange(image_size)
    xs, ys = np.meshgrid(coordinates, coordinates)

    pivot_x = rng.uniform(0.25 * image_size, 0.75 * image_size)
    pivot_y = rng.uniform(0.25 * image_size, 0.75 * image_size)
    angle = rng.uniform(0, 2 * np.pi)
    bend = rng.uniform(1.0, 3.0)

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # Local coordinates normalised by image_size so that bend is scale-invariant.
    u = ((xs - pivot_x) * cos_a + (ys - pivot_y) * sin_a) / image_size
    v = (-(xs - pivot_x) * sin_a + (ys - pivot_y) * cos_a) / image_size

    # Shift the parabola so that the median pixel sits exactly on the arc, giving a
    # 50/50 split. The shifted parabola v = bend*u² + offset is still the boundary of
    # a convex region (shifting does not affect convexity).
    arc = bend * u**2
    offset = np.median(v - arc)
    result = (v - arc > offset).astype(np.uint8)

    return result[..., np.newaxis]
