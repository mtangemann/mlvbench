"""Tests for figure-ground symmetry stimulus generation."""

import numpy as np
import pytest

from mlvbench.datasets.figure_ground_symmetry import (
    _random_symmetry_columns,
    _sample_three_edges,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _symmetry_nominal_xs() -> tuple[float, float, float]:
    return (1 / 6, 3 / 6, 5 / 6)


class TestRandomSymmetryColumns:
    @pytest.mark.parametrize("image_size", [64, 112, 224])
    def test_output_shape(self, image_size):
        """Both outputs have shape (image_size, image_size, 1)."""
        seg = _random_symmetry_columns(image_size, _rng())
        assert seg.shape == (image_size, image_size, 1)

    def test_output_dtype(self):
        """Both outputs are uint8."""
        seg = _random_symmetry_columns(224, _rng())
        assert seg.dtype == np.uint8

    def test_segmentation_values_are_zero_and_one(self):
        """Segmentation contains only 0 and 1."""
        seg = _random_symmetry_columns(224, _rng())
        assert set(np.unique(seg)).issubset({0, 1})

    def test_same_seed_produces_same_result(self):
        """Same RNG seed yields identical segmentation and ignore mask."""
        seg_a = _random_symmetry_columns(224, _rng(0))
        seg_b = _random_symmetry_columns(224, _rng(0))
        np.testing.assert_array_equal(seg_a, seg_b)

    def test_different_seeds_produce_different_results(self):
        """Different seeds yield different segmentation maps."""
        seg_a = _random_symmetry_columns(224, _rng(0))
        seg_b = _random_symmetry_columns(224, _rng(1))
        assert not np.array_equal(seg_a, seg_b)

    def test_class_balance(self):
        """Mean foreground fraction is approximately 0.5 across seeds."""
        fractions = [_random_symmetry_columns(224, _rng(s)).mean() for s in range(50)]
        assert abs(np.mean(fractions) - 0.5) < 0.05

    def test_area_consistency_across_seeds(self):
        """With no edge perturbation, foreground area is constant across seeds.

        Because edge offsets are forced to be mean-zero, the foreground area is
        theoretically constant per seed; any remaining variance is rasterisation noise.
        """
        fractions = [
            _random_symmetry_columns(224, _rng(s), offset_scale=0.0).mean()
            for s in range(100)
        ]
        assert np.std(fractions) < 0.001

    @pytest.mark.parametrize("seed", range(5))
    def test_resolution_independence(self, seed):
        """A stimulus at 448px is an upscaled version of the stimulus at 224px.

        Up to rasterisation, downsampling the 448px stimulus by averaging 2x2 blocks
        recovers the 224px stimulus. Only pixels along the edges may disagree.
        """
        low = _random_symmetry_columns(224, _rng(seed))[..., 0]
        high = _random_symmetry_columns(448, _rng(seed))[..., 0]
        downsampled = high.reshape(224, 2, 224, 2).mean(axis=(1, 3))
        disagreement = np.abs(downsampled - low) > 0.5
        assert disagreement.mean() < 0.005

    def test_reversed_segmentation_is_complement(self):
        """1 - segmentation is the logical complement with uint8 dtype."""
        seg = _random_symmetry_columns(64, _rng(0))
        rev = (1 - seg).astype(np.uint8)
        assert rev.dtype == np.uint8
        np.testing.assert_array_equal(seg + rev, np.ones_like(seg))

    def test_symmetry_val_negative_produces_valid_output(self):
        """symmetry_val=-1 (E2-E3 symmetric foreground) returns a valid segmentation."""
        seg = _random_symmetry_columns(224, _rng(), symmetry_val=-1.0)
        assert seg.dtype == np.uint8
        assert set(np.unique(seg)).issubset({0, 1})

    def test_edges_are_mirror_symmetric(self):
        """When symmetry_val=1, E1 and E2 x-offsets are exact negatives of each other.

        With symmetry_val=1: a=1, b=0, so offset_2 = shape_pos = -offset_1.
        Mean-subtraction preserves the negation, so e1.x_offsets == -e2.x_offsets.
        """
        y_coords = np.linspace(-0.2, 1.2, 25)
        e1, e2, _ = _sample_three_edges(
            _rng(),
            _symmetry_nominal_xs(),
            y_coords,
            offset_scale=0.05,
            symmetry_val=1.0,
        )
        np.testing.assert_array_equal(e1.x_offsets, -e2.x_offsets)
