"""Figure-Ground Convexity Data Module."""

from typing import Any

import numpy as np

from mlvbench.cache import PrecomputedDataset
from mlvbench.datasets._base import ProceduralDataModule
from mlvbench.models import Model
from mlvbench.stimuli.colors import colorize
from mlvbench.stimuli.shapes import split_figure
from mlvbench.stimuli.textures import load_textures, texturize


class FigureGroundConvexityDataModule(ProceduralDataModule):
    """Data module for testing convexity as a figure-ground cue.

    Each image is split into two regions by a parabolic curve. The convex region is
    labelled as the foreground, and the concave region as the background.
    """

    name = "figure_ground_convexity"
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
        split_seed = rng.integers(0, 2**32, size=1)
        split_rng = np.random.default_rng(seed=split_seed)

        # Sample area and center coverage such that both are uninformative about which
        # region is foreground: There's a 50% chance of the convex area covering the
        # center. If the center is covered, the convex area is larger than the
        # background are and vice versa.
        covers_center = bool(shape_rng.random() < 0.5)
        low, high = (
            (0.70, 0.80) if covers_center else (0.20, 0.30)
        )
        area = shape_rng.uniform(low, high)

        segmentation = _random_arc_segmentation(
            self.image_size, rng=shape_rng, area=area,
        )
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


def _random_arc_segmentation(
    image_size: int,
    rng: np.random.Generator,
    area: float = 0.5,
) -> np.ndarray:
    """Create a binary segmentation map by splitting a square image with a bent arc.

    The arc is a parabola in a randomly oriented local coordinate system. The convex
    region (label 1) is the set of pixels above the parabola; the concave region
    (label 0) is the set of pixels below it.

    The parabola is shifted perpendicular to its axis so that the convex region covers
    the requested fraction of the image.

    Args:
        image_size: Side length of the square image in pixels.
        rng: Random number generator used for reproducibility.
        area: Fraction of the image covered by the convex region.

    Returns:
        Segmentation map as uint8 numpy array of shape `(image_size, image_size, 1)`
            with values 0 (concave/background) and 1 (convex/foreground). The convex
            region covers approximately `area` of the image.
    """
    coordinates = np.arange(image_size)
    xs, ys = np.meshgrid(coordinates, coordinates)

    pivot_x = rng.uniform(0.25 * image_size, 0.75 * image_size)
    pivot_y = rng.uniform(0.25 * image_size, 0.75 * image_size)
    angle = rng.uniform(0, 2 * np.pi)
    curvature = rng.uniform(1.0, 2.0)

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # Local coordinates normalised by image_size so that bend is scale-invariant.
    u = ((xs - pivot_x) * cos_a + (ys - pivot_y) * sin_a) / image_size
    v = (-(xs - pivot_x) * sin_a + (ys - pivot_y) * cos_a) / image_size

    # Shift the parabola to the quantile that leaves the requested fraction of pixels
    # above the arc.
    arc = curvature * u**2
    offset = np.quantile(v - arc, 1.0 - area)
    result = (v - arc > offset).astype(np.uint8)

    return result[..., np.newaxis]
