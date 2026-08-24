"""Tests for mlvbench.stimuli."""

import numpy as np
import pytest

from mlvbench.stimuli.shapes import (
    random_idsprite,
    split_figure,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


class TestSplitFigure:
    @pytest.fixture
    def binary_segmentation(self):
        """A 32x32 binary segmentation with a circular foreground region."""
        seg = np.zeros((32, 32, 1), dtype=np.uint8)
        cy, cx = 16, 16
        for y in range(32):
            for x in range(32):
                if (y - cy) ** 2 + (x - cx) ** 2 < 10**2:
                    seg[y, x, 0] = 1
        return seg

    def test_output_shape(self, binary_segmentation):
        """Output shape is (H, W, 1)."""
        result = split_figure(binary_segmentation, _rng())
        assert result.shape == binary_segmentation.shape

    def test_output_dtype(self, binary_segmentation):
        """Output dtype is uint8."""
        result = split_figure(binary_segmentation, _rng())
        assert result.dtype == np.uint8

    def test_output_values(self, binary_segmentation):
        """All values are in {0, 1, 2}."""
        result = split_figure(binary_segmentation, _rng())
        assert set(np.unique(result)).issubset({0, 1, 2})

    def test_background_unchanged(self, binary_segmentation):
        """Pixels that are background in the input remain 0 in the output."""
        result = split_figure(binary_segmentation, _rng())
        background = binary_segmentation[..., 0] == 0
        assert np.all(result[background] == 0)

    def test_foreground_fully_covered(self, binary_segmentation):
        """Every foreground pixel is assigned to sub-region 1 or 2."""
        result = split_figure(binary_segmentation, _rng())
        foreground = binary_segmentation[..., 0] > 0
        assert np.all(result[foreground] > 0)

    def test_both_subregions_nonempty(self, binary_segmentation):
        """Both sub-region labels (1 and 2) appear in the output."""
        result = split_figure(binary_segmentation, _rng())
        assert 1 in result
        assert 2 in result

    @pytest.mark.parametrize("min_region_size", [0.1, 0.2, 0.4])
    def test_min_region_size_respected(self, binary_segmentation, min_region_size):
        """Each sub-region is at least min_region_size of the total foreground."""
        result = split_figure(
            binary_segmentation, _rng(), min_region_size=min_region_size
        )
        foreground_size = int((binary_segmentation[..., 0] > 0).sum())
        a_size = int((result[..., 0] == 1).sum())
        b_size = int((result[..., 0] == 2).sum())
        assert min(a_size, b_size) / foreground_size >= min_region_size

    def test_same_seed_produces_same_result(self, binary_segmentation):
        """Two calls with the same seed return identical arrays."""
        result_a = split_figure(binary_segmentation, _rng(7))
        result_b = split_figure(binary_segmentation, _rng(7))
        np.testing.assert_array_equal(result_a, result_b)

    def test_different_seeds_produce_different_results(self, binary_segmentation):
        """Different seeds yield different splits."""
        result_a = split_figure(binary_segmentation, _rng(0))
        result_b = split_figure(binary_segmentation, _rng(1))
        assert not np.array_equal(result_a, result_b)

    @pytest.mark.parametrize("bad_value", [-0.1, 0.51, 1.0])
    def test_invalid_min_region_size_raises(self, binary_segmentation, bad_value):
        """Values outside [0.0, 0.5] raise ValueError."""
        with pytest.raises(ValueError):
            split_figure(binary_segmentation, _rng(), min_region_size=bad_value)


class TestRandomIdsprite:
    @pytest.mark.parametrize("image_size", [32, 64, 224])
    def test_output_shape(self, image_size):
        """Output shape is (image_size, image_size, 1)."""
        result = random_idsprite(image_size, _rng())
        assert result.shape == (image_size, image_size, 1)

    def test_output_dtype(self):
        """Output dtype is uint8."""
        result = random_idsprite(64, _rng())
        assert result.dtype == np.uint8

    def test_output_values(self):
        """All values are in {0, 1}."""
        result = random_idsprite(64, _rng())
        assert set(np.unique(result)) == {0, 1}
        assert set(np.unique(result)) == {0, 1}

    def test_same_seed_produces_same_result(self):
        """Same seed yields identical output."""
        result_a = random_idsprite(64, _rng(0))
        result_b = random_idsprite(64, _rng(0))
        np.testing.assert_array_equal(result_a, result_b)

    def test_different_seeds_produce_different_results(self):
        """Different seeds yield different outputs."""
        result_a = random_idsprite(64, _rng(0))
        result_b = random_idsprite(64, _rng(1))
        assert not np.array_equal(result_a, result_b)
