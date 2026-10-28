"""Data modules for testing figure-ground shape cues."""

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.ndimage import distance_transform_edt

from mlvbench.cache import PrecomputedDataset
from mlvbench.datasets._base import ProceduralDataModule
from mlvbench.models import Model
from mlvbench.stimuli.colors import colorize
from mlvbench.stimuli.textures import load_textures, texturize


class FigureGroundParallelismDataModule(ProceduralDataModule):
    """Data module for testing parallelism as a figure-ground cue."""

    name = "figure_ground_parallelism"
    version = "1.0.0-dev"

    # All test subsets are views onto the same generated test split and differ only in
    # the image variant they read.
    _TEST_FEATURES = {
        "test": { "image": "image", "segmentation": "segmentation" },
        "test_reversed": { "image": "image_reversed", "segmentation": "segmentation" },
        "test_color": {"image": "image_color", "segmentation": "segmentation"},
        "test_color_reversed": {
            "image": "image_color_reversed",
            "segmentation": "segmentation",
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
        """Generate a single figure-ground parallelism sample."""
        rng = np.random.default_rng(seed=seed)
        shape_seed, texture_seed = rng.integers(0, 2**32, size=2)
        shape_rng = np.random.default_rng(seed=shape_seed)
        texture_rng = np.random.default_rng(seed=texture_seed)

        segmentation = random_segmentation(shape_rng, self.image_size)
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

            # The color variants reuse the texture seed, so a region keeps its
            # color across the variants just like it keeps its texture.
            color_rng = np.random.default_rng(seed=texture_seed)
            sample["image_color"] = colorize(segmentation, rng=color_rng)

            color_rng = np.random.default_rng(seed=texture_seed)
            sample["image_color_reversed"] = colorize(1 - segmentation, rng=color_rng)

        else:
            # For the training and validation set, we randomly select a colorization or
            # texturization variant.
            if rng.random() < 0.5:
                image = texturize(segmentation, textures, rng=texture_rng)
            else:
                image = colorize(segmentation, rng=texture_rng)

            sample = {"image": image, "segmentation": segmentation}

        return sample

    def after_generate(self) -> None:
        """Clean up the textures after generating the samples."""
        del self.textures

    def subsets(self) -> list[str]:
        """Return the names of all available subsets."""
        return ["train", "val", *self._TEST_FEATURES]

    def test_dataset(self, subset: str = "test") -> PrecomputedDataset:
        """Return the requested test split (standard or reversed variant)."""
        if subset not in self._TEST_FEATURES:
            raise ValueError(f"Unknown test subset: {subset}")
        return PrecomputedDataset(
            self._path / "test",
            features=self._TEST_FEATURES[subset],
            mmap=False,
        )



def random_segmentation(
    rng, size: int = 256, thickness_frac: float = 0.33, min_gap: int = 6,
    hero_min_length_frac: float = 1.0, fill_min_length_frac: float = 0.4,
    max_failures: int = 50,
):
    """Segmentation of parallel-contoured strips: a hero strip plus space-filling ones.

    The hero strip (label 1) must reach `hero_min_length_frac * size` and not
    self-intersect. Further strips are then packed into the leftover space against a
    lower length threshold (`fill_min_length_frac * size`); a candidate is rejected if
    it is too short, self-intersects, or lies within `min_gap` pixels of an
    already-placed strip, and sampling stops after `max_failures` consecutive
    rejections. Each placed strip gets its own integer label (1, 2, 3, ...). Returns a
    `(size, size)` uint8 array.
    """
    radius = 0.5 * thickness_frac * size
    pad = radius + 2.0  # extend axis ends past the frame so round caps are clipped
    xs = np.arange(0.0, 2.0 * size, 1.0)
    length_scale = 0.33 * size
    gpp = SpectralGaussianProcessPrior(
        xs, length_scale=length_scale, sigma=0.5 / length_scale
    )

    axis_x, axis_y = _hero_axis(
        rng, size, gpp, min_length_frac=hero_min_length_frac
    )
    segmentation = np.zeros((size, size), dtype=np.uint8)
    segmentation[_strip_mask(*_pad_axis_ends(axis_x, axis_y, pad), radius, size)] = 1

    # Fill the leftover space. `dist_free` is the distance from every pixel to the
    # nearest placed-strip pixel; it only changes when a strip is placed. A
    # candidate's dilated strip would overlap or come within `min_gap` of a placed
    # strip exactly when its medial axis passes within `radius + min_gap` of one, so
    # the reject test just samples `dist_free` along the axis -- no need to rasterise
    # a strip that is not going to be kept.
    fill_min_length = fill_min_length_frac * size
    clearance = radius + min_gap
    dist_free = distance_transform_edt(segmentation == 0)
    label, failures = 1, 0
    while failures < max_failures:
        axis_x, axis_y = _sample_axis(rng, size, gpp)
        length = float(np.hypot(np.diff(axis_x), np.diff(axis_y)).sum())
        if length < fill_min_length:
            failures += 1
            continue
        padded_x, padded_y = _pad_axis_ends(axis_x, axis_y, pad)
        px, py = _densify(padded_x, padded_y, spacing=1.0)
        ix = np.clip(np.round(px).astype(int), 0, size - 1)
        iy = np.clip(np.round(py).astype(int), 0, size - 1)
        if dist_free[iy, ix].min() <= clearance:
            failures += 1
            continue
        label += 1
        segmentation[_strip_mask(padded_x, padded_y, radius, size)] = label
        dist_free = distance_transform_edt(segmentation == 0)
        failures = 0

    return (segmentation > 0).astype(np.uint8)[..., np.newaxis]


class SpectralGaussianProcessPrior:
    """Finite spectral approximation to a GP prior with an RBF kernel.

    Approximates the stationary covariance

        k(x, x') = sigma**2
                    * exp(-(x - x')**2 / (2 * length_scale**2))

    using a finite random Fourier expansion.

    The prior itself is independent of the points at which it is evaluated.
    Given RNGs in identical states, calls at different evaluation grids sample
    the same latent Fourier coefficients and therefore evaluate the same
    random function.
    """

    def __init__(
        self,
        xs: npt.ArrayLike,
        sigma: float = 1.0,
        length_scale: float = 0.1,
        num_frequencies: int = 256,
        spectral_cutoff: float = 6.0,
    ):
        """Initialize the Gaussian Process prior.

        Args:
            xs:
                Points at which to evaluate the sampled function. The points may be
                arbitrary and do not need to be regularly spaced.
            sigma:
                Pointwise standard deviation of the GP. The kernel variance is therefore
                sigma**2.
            length_scale:
                RBF length scale. Smaller values produce functions that vary more
                rapidly with x.
            num_frequencies:
                Number of positive Fourier frequencies used in the approximation.
            spectral_cutoff:
                Maximum dimensionless frequency `length_scale * omega`.
                The RBF spectral density decays as

                    exp(-0.5 * (length_scale * omega)**2),

                so the default of 6 discards only extremely small
                high-frequency contributions.
        """
        if sigma < 0:
            raise ValueError("sigma must be non-negative.")
        if length_scale <= 0:
            raise ValueError("length_scale must be positive.")
        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive.")
        if spectral_cutoff <= 0:
            raise ValueError("spectral_cutoff must be positive.")

        self.sigma = sigma
        self.length_scale = length_scale
        self.num_frequencies = num_frequencies
        self.spectral_cutoff = spectral_cutoff

        # Midpoint quadrature grid in the dimensionless variable
        #
        #     u = length_scale * omega.
        #
        # This makes the frequency representation depend only on the kernel,
        # not on the eventual evaluation grid.
        du = spectral_cutoff / num_frequencies
        u = (np.arange(num_frequencies) + 0.5) * du

        self.omega = u / length_scale

        # RBF spectral density:
        #
        #     S(omega) ∝ exp(-0.5 * (ell * omega)**2)
        #
        # Fourier coefficient std is proportional to sqrt(S), hence -0.25.
        weights = np.exp(-0.25 * u**2)

        # Normalize the finite approximation to have exactly
        #
        #     Var[f(x)] = sigma**2
        #
        # at every x.
        self.weights = sigma * weights / np.linalg.norm(weights)

        # Precompute the sin and cos basis functions for the Fourier approximation
        xs = np.asarray(xs, dtype=np.float64)
        phase = np.multiply.outer(xs, self.omega)
        self.basis_cos = np.cos(phase)
        self.basis_sin = np.sin(phase)

    def __call__(
        self,
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        """Sample and evaluate a function from the GP prior.

        Args:
            rng:
                Random number generator used to sample the latent Fourier coefficients.
                RNGs in identical states produce the same underlying random function,
                regardless of `xs`.

        Returns:
            Sampled function evaluated at `xs`.
        """
        # These random coefficients completely specify the sampled function.
        z_cos = rng.standard_normal(self.num_frequencies)
        z_sin = rng.standard_normal(self.num_frequencies)

        coefficients_cos = self.weights * z_cos
        coefficients_sin = self.weights * z_sin

        return (
            self.basis_cos @ coefficients_cos
            + self.basis_sin @ coefficients_sin
        )


def _densify(axis_x, axis_y, spacing):
    """Resample the polyline (axis_x, axis_y) to points at most `spacing` pixels apart.

    Cumulative arc length plus two `np.interp` calls, so the whole axis is resampled
    in one vectorised shot regardless of how the original vertices are spaced
    (including the long final segments `_pad_axis_ends` tacks on). Returns `(x, y)`
    arrays.
    """
    ax = np.asarray(axis_x, dtype=float)
    ay = np.asarray(axis_y, dtype=float)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(ax), np.diff(ay)))])
    if s[-1] == 0.0:
        return ax.copy(), ay.copy()
    u = np.arange(0.0, s[-1], spacing)
    return np.interp(u, s, ax), np.interp(u, s, ay)


def _strip_mask(axis_x, axis_y, radius, size, spacing=0.4):
    """Binary mask of pixels within `radius` of polyline (axis_x, axis_y)."""
    px, py = _densify(axis_x, axis_y, spacing)
    ix = np.round(px).astype(int)
    iy = np.round(py).astype(int)
    inside = (ix >= 0) & (ix < size) & (iy >= 0) & (iy < size)
    seed = np.ones((size, size), dtype=bool)
    seed[iy[inside], ix[inside]] = False
    return distance_transform_edt(seed) <= radius


def _pad_axis_ends(axis_x, axis_y, pad):
    """Linearly extend the medial axis by `pad` pixels at each end."""
    if len(axis_x) < 2:
        return axis_x, axis_y

    def _tip(tip_x, tip_y, prev_x, prev_y):
        dx, dy = tip_x - prev_x, tip_y - prev_y
        n = math.hypot(dx, dy) or 1.0
        return tip_x + pad * dx / n, tip_y + pad * dy / n

    sx, sy = _tip(axis_x[0], axis_y[0], axis_x[1], axis_y[1])
    ex, ey = _tip(axis_x[-1], axis_y[-1], axis_x[-2], axis_y[-2])
    return (
        np.concatenate([[sx], axis_x, [ex]]),
        np.concatenate([[sy], axis_y, [ey]]),
    )


def _border_point(rng, size, side, margin=0.10):
    """Random point on an image border (0=top, 1=right, 2=bottom, 3=left)."""
    u = float(rng.uniform(margin * size, (1.0 - margin) * size))
    return np.array([(u, 0.0), (size, u), (u, size), (0.0, u)][side], dtype=float)


def _sample_axis(rng, size, gpp, step=1.0):
    """Draw a single random-walk medial axis, clipped where it first leaves the image.

    Enters at a random border point heading roughly inward, then marches with a
    curvature `kappa(s)` drawn from the spectral GP prior via `_sample_gp`, integrated
    once to a heading and again to the polyline. No acceptance test is applied here.
    Returns `(axis_x, axis_y)`.
    """
    side = int(rng.integers(4))
    entry = _border_point(rng, size, side)
    inward = [(0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0)][side]
    spread = rng.uniform(-math.pi / 2, math.pi / 2)
    theta0 = math.atan2(inward[1], inward[0]) + float(spread)

    kappa = gpp(rng)

    theta = theta0 + np.cumsum(kappa * step)
    x = np.concatenate([[entry[0]], entry[0] + np.cumsum(step * np.cos(theta))])
    y = np.concatenate([[entry[1]], entry[1] + np.cumsum(step * np.sin(theta))])

    inside = (x >= 0.0) & (x <= size) & (y >= 0.0) & (y <= size)
    stop = len(x) - 1 if inside.all() else max(int(np.argmin(inside)), 1)
    return x[: stop + 1], y[: stop + 1]


def _hero_axis(rng, size, gpp, step=1.0, min_length_frac=1.0, max_tries=200):
    """Medial axis for the hero strip: `_sample_axis` draws that pass acceptance test.

    Axes that self-intersect or are shorter than `min_length_frac * size` are rejected
    and resampled, up to `max_tries` times (the last attempt is returned if all fail).
    Returns `(axis_x, axis_y)`.
    """
    min_length = min_length_frac * size
    for _ in range(max_tries):
        axis_x, axis_y = _sample_axis(rng, size, gpp, step)
        length = float(np.hypot(np.diff(axis_x), np.diff(axis_y)).sum())
        if length >= min_length:
            return axis_x, axis_y
    return axis_x, axis_y
