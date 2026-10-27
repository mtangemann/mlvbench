"""Tests for mlvbench.stimuli.colors."""

import numpy as np
import pytest
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab

from mlvbench.stimuli.colors import colorize, sample_colors


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _region_map(num_regions: int = 2, size: int = 32) -> np.ndarray:
    """A `size x size` region map split into `num_regions` horizontal bands."""
    region_map = np.zeros((size, size, 1), dtype=np.uint8)
    for label in range(num_regions):
        region_map[label * size // num_regions:, :, 0] = label
    return region_map


class TestSampleColors:
    def test_output_shape_and_dtype(self):
        colors = sample_colors(4, _rng())
        assert colors.shape == (4, 3)
        assert colors.dtype == np.uint8

    def test_zero_colors(self):
        colors = sample_colors(0, _rng())
        assert colors.shape == (0, 3)
        assert colors.dtype == np.uint8

    def test_negative_number_of_colors(self):
        with pytest.raises(ValueError, match="non-negative"):
            sample_colors(-1, _rng())

    def test_deterministic(self):
        assert np.array_equal(sample_colors(4, _rng(0)), sample_colors(4, _rng(0)))

    def test_seed_sensitive(self):
        assert not np.array_equal(sample_colors(4, _rng(0)), sample_colors(4, _rng(1)))

    def test_colors_are_in_gamut(self):
        """Sampled colors round-trip through CIELAB, i.e. they were not clipped."""
        colors = sample_colors(8, _rng()) / 255.0
        lab = rgb2lab(colors)
        # uint8 quantization is the only expected round-trip error.
        assert np.abs(colors - lab2rgb(lab)).max() < 1e-6

    def test_minimum_distance(self):
        lab = rgb2lab(sample_colors(8, _rng()) / 255.0)
        for i in range(len(lab)):
            for j in range(i + 1, len(lab)):
                distance = deltaE_ciede2000(lab[i:i + 1], lab[j:j + 1])[0]
                # Add some tolerance to ccount for uint8 quantization.
                assert distance >= 25.0 - 0.1

    def test_lightness_range_is_respected(self):
        colors = sample_colors(4, _rng(), lightness_range=(60.0, 70.0))
        lab = rgb2lab(colors / 255.0)
        assert lab[:, 0].min() >= 60.0 - 0.5
        assert lab[:, 0].max() <= 70.0 + 0.5

    def test_prefix_stability(self):
        """The first `min(N, M)` colors do not depend on how many are requested.

        `test_color_split` relies on this: the three-region split image must share its
        first two colors with the two-region standard image.
        """
        many = sample_colors(5, _rng(3))
        for num_colors in range(1, 5):
            few = sample_colors(num_colors, _rng(3))
            assert np.array_equal(few, many[:num_colors])

    def test_raises_when_attempts_are_exhausted(self):
        with pytest.raises(ValueError, match="within 128 attempts"):
            sample_colors(
                4,
                _rng(),
                lightness_range=(50.0, 51.0),
                chroma_range=(0.0, 1.0),
                min_delta_e=50.0,
                max_attempts=128,
            )


class TestColorize:
    def test_output_shape_and_dtype(self):
        result = colorize(_region_map(), _rng())
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8

    def test_regions_are_uniformly_filled(self):
        region_map = _region_map(num_regions=3)
        result = colorize(region_map, _rng())
        for label in np.unique(region_map[..., 0]):
            region = result[region_map[..., 0] == label]
            assert len(np.unique(region, axis=0)) == 1

    def test_regions_get_distinct_colors(self):
        region_map = _region_map(num_regions=3)
        result = colorize(region_map, _rng())
        colors = {tuple(result[region_map[..., 0] == label][0]) for label in range(3)}
        assert len(colors) == 3

    def test_deterministic(self):
        region_map = _region_map()
        assert np.array_equal(
            colorize(region_map, _rng(0)), colorize(region_map, _rng(0))
        )

    def test_seed_sensitive(self):
        region_map = _region_map()
        assert not np.array_equal(
            colorize(region_map, _rng(0)), colorize(region_map, _rng(1))
        )

    def test_uses_the_same_colors_for_a_reversed_region_map(self):
        region_map = _region_map()
        image = colorize(region_map, _rng(0))
        reversed_image = colorize(1 - region_map, _rng(0))
        assert np.array_equal(image[0, 0], reversed_image[-1, 0])
        assert np.array_equal(image[-1, 0], reversed_image[0, 0])

    def test_shares_colors_with_a_split_region_map(self):
        """Splitting a region keeps the colors of the regions that come before it."""
        region_map = _region_map(num_regions=2)
        split_region_map = region_map.copy()
        split_region_map[24:, :, 0] = 2

        image = colorize(region_map, _rng(0))
        split_image = colorize(split_region_map, _rng(0))

        assert np.array_equal(image[0, 0], split_image[0, 0])
        assert np.array_equal(image[16, 0], split_image[16, 0])
        assert not np.array_equal(split_image[16, 0], split_image[24, 0])

    def test_non_contiguous_labels(self):
        region_map = np.zeros((16, 16, 1), dtype=np.uint8)
        region_map[8:, :, 0] = 7
        result = colorize(region_map, _rng())
        assert len(np.unique(result.reshape(-1, 3), axis=0)) == 2
