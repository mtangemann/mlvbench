"""Tests for the metrics module."""

import pytest
import torch

from mlvbench.metrics import intersection_over_union


class TestIntersectionOverUnion:
    """Tests for Intersection over Union (IoU) for binary masks."""

    def test_identical_masks(self):
        mask = torch.tensor([True, False, True, False])
        assert intersection_over_union(mask, mask) == 1.0

    def test_disjoint_masks(self):
        prediction = torch.tensor([True, True, False, False])
        target = torch.tensor([False, False, True, True])
        assert intersection_over_union(prediction, target) == 0.0

    def test_partial_overlap(self):
        # Intersection: 1 element, union: 3 elements.
        prediction = torch.tensor([True, True, False, False])
        target = torch.tensor([False, True, True, False])
        assert intersection_over_union(prediction, target) == pytest.approx(1 / 3)

    def test_multidimensional(self):
        # Intersection: 2 elements, union: 4 elements.
        prediction = torch.tensor([[True, True], [True, False]])
        target = torch.tensor([[True, False], [True, True]])
        assert intersection_over_union(prediction, target) == pytest.approx(0.5)

    def test_symmetric(self):
        prediction = torch.tensor([True, True, True, False])
        target = torch.tensor([False, True, False, False])
        assert intersection_over_union(prediction, target) == pytest.approx(
            intersection_over_union(target, prediction)
        )

    def test_empty_masks(self):
        """IoU should be 1.0 if both masks are empty."""
        mask = torch.zeros(4, dtype=torch.bool)
        assert intersection_over_union(mask, mask) == 1.0

    def test_empty_prediction(self):
        prediction = torch.zeros(4, dtype=torch.bool)
        target = torch.tensor([True, False, False, False])
        assert intersection_over_union(prediction, target) == 0.0

    def test_requires_boolean_masks(self):
        mask = torch.tensor([1.0, 0.0])
        with pytest.raises(TypeError, match="boolean"):
            intersection_over_union(mask, mask > 0)

    def test_requires_same_shape(self):
        with pytest.raises(ValueError, match="same shape"):
            intersection_over_union(
                torch.ones(2, dtype=torch.bool),
                torch.ones(3, dtype=torch.bool),
            )
