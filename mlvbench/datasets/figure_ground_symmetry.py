"""Figure-Ground Symmetry Data Module."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from matplotlib.path import Path as MPath
from skimage.draw import polygon as sk_polygon

from mlvbench.cache import PrecomputedDataset
from mlvbench.datasets._base import ProceduralDataModule
from mlvbench.models import Model
from mlvbench.stimuli.textures import load_textures, texturize


class FigureGroundSymmetryDataModule(ProceduralDataModule):
    """Data module for testing symmetry as a figure-ground cue."""

    name = "figure_ground_symmetry"
    version = "1.0.0"

    # All test subsets are views onto the same generated test split and differ only in
    # the image variant they read.
    _TEST_FEATURE_MAPS = {
        "test": { "image": "image", "segmentation": "segmentation" },
        "test_reversed": { "image_reversed": "image", "segmentation": "segmentation" },
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
        """Generate a single figure-ground symmetry sample."""
        rng = np.random.default_rng(seed=seed)
        shape_seed, texture_seed = rng.integers(0, 2**32, size=2)
        shape_rng = np.random.default_rng(seed=shape_seed)
        texture_rng = np.random.default_rng(seed=texture_seed)

        segmentation = _random_symmetry_columns(
            self.image_size,
            rng=shape_rng,
            symmetry_val=rng.choice([-1.0, 1.0]),
            num_control_points=10,
            offset_scale=25.0,
        )

        textures = self.textures[subset]
        image = texturize(segmentation, textures, rng=texture_rng)
        sample = {
            "image": image,
            "segmentation": segmentation,
        }

        if subset == "test":
            # Reinitialize the texture RNG, so that we sample the same textures for the
            # reversed image.
            texture_rng = np.random.default_rng(seed=texture_seed)
            image_reversed = texturize(1 - segmentation, textures, rng=texture_rng)
            sample["image_reversed"] = image_reversed

        return sample

    def after_generate(self) -> None:
        """Clean up the textures after generating the samples."""
        del self.textures

    def subsets(self) -> list[str]:
        """Return the names of all available subsets."""
        return ["train", "val", *self._TEST_FEATURE_MAPS]

    def test_dataset(self, subset: str = "test") -> PrecomputedDataset:
        """Return the requested test split (standard or reversed variant)."""
        if subset not in self._TEST_FEATURE_MAPS:
            raise ValueError(f"Unknown test subset: {subset}")
        return PrecomputedDataset(
            self._path / "test",
            feature_map=self._TEST_FEATURE_MAPS[subset],
            mmap=False,
        )


def _random_symmetry_columns(
    image_size: int,
    rng: np.random.Generator,
    num_control_points: int = 25,
    symmetry_val: float = 1.0,
    offset_scale: float = 10.0,
) -> np.ndarray:
    """Create a segmentation map by splitting an image into two regions using columns.

    The columns are defined as having width of 1/4 of the image and are parameterized
    by three vertical edges placed at 2/8 , 4/8  and 6/8 nominal positions horizontally
    across the image.

    The symmetric foreground region (label 1) is the set of pixels enclosed
    on both sides by two of the three vertical edges (which two depends on the value
    of symmetry_val), as well as the region enclosed by the third edge and the nearest
    vertical image boundary. The asymmetric background region (label 0) is the set of
    pixels in between the foreground columns.

    The columns are designed to have equal areas between them, such that foreground
    and background pixels are roughly equal for each image. Column areas are
    approximately equal between seeds.

    Args:
        image_size: Side length of the square image in pixels.
        rng: NumPy random number generator used for reproducibility.
        num_control_points: Number of Bezier curve control points along the columns.
        symmetry_val: How symmetric each edge should be. 1.0 -> symmetric,
                0 -> asymmetric, -1 -> oppositely symmetric
        offset_scale: Scale of perturbations on each side of the column.
                0.0 gives perfect rectangles.

    Returns:
        Segmentation map as uint8 numpy array of shape
            `(image_size, image_size, 1)` with values 0 (background) and 1 (foreground).
                Each region covers approximately 50% of the image.
    """
    nominal_xs = (
        int(1 / 6 * image_size),  # E1
        int(3 / 6 * image_size),  # E2
        int(5 / 6 * image_size),  # E3
    )

    # Pad beyond the image by 50 pixels on each side to prevent rendering artefacts
    y_coords = np.linspace(-50, image_size + 50, num_control_points)

    # Sample three edges according to seed and symmetry value
    E1, E2, E3 = _sample_three_edges(
        rng, nominal_xs, y_coords, offset_scale, symmetry_val
    )

    # Right-side extends off screen; Represent as straight synthetic edge for renderer
    off_right = _Edge(float(image_size + 500), y_coords, np.zeros_like(y_coords))
    black_regions = [(E1, E2), (E3, off_right)]

    # Assign figure/ground based on symmetry value
    black_foreground_mask = _render_mask(image_size, black_regions)
    white_foreground_mask = (1 - black_foreground_mask).astype(np.uint8)

    if symmetry_val > 0:
        label = black_foreground_mask
    elif symmetry_val < 0:
        label = white_foreground_mask
    else:  # No dominant foreground, we randomly pick a labelling
        choices = [white_foreground_mask, black_foreground_mask]
        label = choices[rng.choice([0, 1])]

    return label[..., np.newaxis]


@dataclass(frozen=True)
class _Edge:
    """A wavy vertical edge at a position x, defined as control points with offsets."""

    x_position: float
    y_coords: np.ndarray  # shape (N,)
    x_offsets: np.ndarray  # shape (N,)

    @property
    def xs(self) -> np.ndarray:
        """Offset x-coordinates for edge control points."""
        return self.x_position + self.x_offsets


def _sample_three_edges(
    rng: np.random.Generator,
    nominal_xs: tuple[float, float, float],
    y_coords: np.ndarray,
    offset_scale: float,
    symmetry_val: float,
) -> tuple[_Edge, _Edge, _Edge]:
    """Sample the three stimulus edges parameterized by a symmetry value.

    symmetry_val = +1 -> Edge 2 Mirrors Edge 1. Edge 3 is random
    symmetry_val = 0 -> Edges 1, 2, and 3 are all independent
    symmetry_val = -1 -> Edge 2 Mirrors Edge 3. Edge 1 is random
    """
    n = len(y_coords)
    shape_pos = rng.uniform(-offset_scale, offset_scale, n)
    shape_neg = rng.uniform(-offset_scale, offset_scale, n)

    a = (1 + symmetry_val) / 2  # Weight on the black-figure regime
    b = 1 - a  # Weight on the white-figure regime

    offset_1 = -shape_pos
    offset_2 = a * shape_pos - b * shape_neg
    offset_3 = shape_neg

    offsets = [o - o.mean() for o in (offset_1, offset_2, offset_3)]
    edges = tuple(
        _Edge(x, y_coords, o) for x, o in zip(nominal_xs, offsets, strict=False)
    )
    return cast(tuple[_Edge, _Edge, _Edge], tuple(edges))


def _column_path(left: _Edge, right: _Edge) -> MPath:
    """Return a closed matplotlib ath for the region between two edges."""
    # Initialize start point
    verts = [(left.xs[0], left.y_coords[0])]
    codes = [MPath.MOVETO]

    for control, endpoint in _get_bezier_segments(left, reverse=False):
        verts.extend([control, endpoint])
        codes.extend([MPath.CURVE3, MPath.CURVE3])

    # Draw line going from left edge to right edge outside of image boundary
    verts.append((right.xs[-1], right.y_coords[-1]))
    codes.append(MPath.LINETO)

    for control, endpoint in _get_bezier_segments(right, reverse=True):
        verts.extend([control, endpoint])
        codes.extend([MPath.CURVE3, MPath.CURVE3])

    # Close the curve
    verts.append((0.0, 0.0))
    codes.append(MPath.CLOSEPOLY)

    return MPath(verts, codes)


def _get_bezier_segments(edge: _Edge, reverse: bool = False):
    """Yield (control, endpoint) vertex pairs for a CURVE3 traversal of an edge.

    Each segment's control point sits on the x line at the endpoint's y.
    This has the effect of making each curve pass through the invisible rectangular
    column edges.
    """
    xs, ys = edge.xs, edge.y_coords
    idx = range(len(xs) - 1, -1, -1) if reverse else range(len(xs))

    for k in range(1, len(idx)):
        i_prev, i_curr = idx[k - 1], idx[k]
        y_mid = 0.5 * (ys[i_curr] + ys[i_prev])
        control = (edge.x_position, y_mid)
        endpoint = (xs[i_curr], ys[i_curr])

        yield control, endpoint


def _render_mask(image_size, edge_pairs) -> np.ndarray:
    """Render the segmentation foreground/background mask."""
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    for left, right in edge_pairs:
        path = _column_path(left, right)
        verts = path.to_polygons(closed_only=False)[0]
        rr, cc = sk_polygon(verts[:, 1], verts[:, 0], shape=mask.shape)  # type: ignore
        mask[rr, cc] = True
    return mask
