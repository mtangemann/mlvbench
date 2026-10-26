"""Tests for mlvbench.stimuli."""

import numpy as np
import pytest

from mlvbench.stimuli.shapes import (
    _central_disc,
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

    @pytest.mark.parametrize("area", [0.1, 0.3, 0.55])
    def test_rendered_area_matches_target(self, area):
        """The rendered mask covers the requested fraction of the image."""
        result = random_idsprite(224, _rng(0), area=area, margin=0.01)
        assert result.mean() == pytest.approx(area, abs=0.005)

    @pytest.mark.parametrize(("covers_center", "area"), [(True, 0.55), (False, 0.15)])
    def test_center_constraint_respected(self, covers_center, area):
        """The shape covers the central disc completely, or misses it completely."""
        center = _central_disc(224, 0.02)
        rng = _rng(0)

        for _ in range(10):
            result = random_idsprite(
                224,
                rng,
                area=area,
                covers_center=covers_center,
                margin=0.01,
            )
            covered = result[..., 0][center] > 0
            assert covered.all() if covers_center else not covered.any()

    @pytest.mark.parametrize("area", [0.1, 0.55])
    def test_margin_keeps_the_shape_surrounded(self, area):
        """No foreground pixel touches the image border."""
        rng = _rng(0)

        for _ in range(10):
            mask = random_idsprite(224, rng, area=area, margin=0.01)[..., 0]
            assert not mask[0].any()
            assert not mask[-1].any()
            assert not mask[:, 0].any()
            assert not mask[:, -1].any()

    @pytest.mark.parametrize("bad_area", [0.0, 1.0, -0.5])
    def test_invalid_area_raises(self, bad_area):
        """Areas outside (0.0, 1.0) raise ValueError."""
        with pytest.raises(ValueError):
            random_idsprite(64, _rng(), area=bad_area)

    def test_unreachable_area_raises(self):
        """An area no shape can reach raises instead of looping forever."""
        with pytest.raises(ValueError):
            random_idsprite(64, _rng(), area=0.95, max_attempts=20)


class TestCentralDisc:
    def test_shape_and_dtype(self):
        """Output is a boolean mask of shape (image_size, image_size)."""
        disc = _central_disc(64, 0.05)
        assert disc.shape == (64, 64)
        assert disc.dtype == np.bool_

    def test_contains_the_image_center(self):
        """The disc covers the pixel closest to the image center."""
        disc = _central_disc(64, 0.05)
        assert disc[32, 32]

    def test_is_never_empty(self):
        """A radius rounding down to less than a pixel still yields one pixel."""
        assert _central_disc(64, 0.0).sum() == 1
