"""Test for the binary segmentation task."""

import pytest
import torch
import torch.nn.functional as F

from mlvbench.tasks.binary_segmentation import (
    BinarySegmentationEvaluator,
    BinarySegmentationTask,
)


def _make_batch(
    batch_size: int,
    grid_h: int,
    grid_w: int,
    patch_size: int,
    label_fn,
):
    """Build a batch for testing."""
    segmentation = torch.zeros(batch_size, 1, grid_h, grid_w, dtype=torch.uint8)
    for b in range(batch_size):
        for y in range(grid_h):
            for x in range(grid_w):
                segmentation[b, 0, y, x] = label_fn(b, y, x)

    segmentation = segmentation.repeat_interleave(patch_size, dim=2)
    segmentation = segmentation.repeat_interleave(patch_size, dim=3)

    return {"segmentation": segmentation}


class TestBinarySegmentationTask:
    def test_prepare_target_shapes(self):
        """Targets are (B, N, 1) and weights are (B, N, 1)."""
        B, H, W, P = 2, 4, 4, 14
        task = BinarySegmentationTask(patch_size=P)
        batch = _make_batch(B, H, W, P, lambda b, y, x: 0)

        targets, weights = task.prepare_target(batch)

        assert targets.shape == (B, H * W, 1)
        assert weights.shape == (B, H * W, 1)

    def test_prepare_target_all_background(self):
        """Segmentation all zeros → targets are all 0."""
        B, H, W, P = 1, 2, 2, 14
        task = BinarySegmentationTask(patch_size=P)
        batch = _make_batch(B, H, W, P, lambda b, y, x: 0)

        targets, _ = task.prepare_target(batch)
        assert torch.all(targets == 0.0)

    def test_prepare_target_all_foreground(self):
        """Segmentation all nonzero → targets are all 1."""
        B, H, W, P = 1, 2, 2, 14
        task = BinarySegmentationTask(patch_size=P)
        batch = _make_batch(B, H, W, P, lambda b, y, x: 1)

        targets, _ = task.prepare_target(batch)
        assert torch.all(targets == 1.0)

    def test_prepare_target_mixed(self):
        """Patches with label > 0 get target 1, patches with label 0 get target 0."""
        B, H, W, P = 1, 2, 2, 14
        task = BinarySegmentationTask(patch_size=P)
        # Row 0 (y=0): label 0 → background. Row 1 (y=1): label 1 → foreground.
        batch = _make_batch(B, H, W, P, lambda b, y, x: y)

        targets, _ = task.prepare_target(batch)
        expected = torch.tensor([[0.0, 0.0, 1.0, 1.0]]).unsqueeze(-1)
        assert torch.all(targets == expected)

    def test_pure_patches_have_weight_one(self):
        """Patches with a single segment label get weight 1."""
        B, H, W, P = 1, 2, 2, 14
        task = BinarySegmentationTask(patch_size=P)
        batch = _make_batch(B, H, W, P, lambda b, y, x: y)

        _, weights = task.prepare_target(batch)
        assert torch.all(weights == 1.0)

    def test_mixed_patches_have_weight_zero(self):
        """Patches containing more than one segment label get weight 0."""
        P = 4
        # Single patch (1×1 grid, P×P pixels): top half label 0, bottom half label 1.
        segmentation = torch.zeros(1, 1, P, P, dtype=torch.uint8)
        segmentation[0, 0, P // 2 :, :] = 1
        batch = {"segmentation": segmentation}

        task = BinarySegmentationTask(patch_size=P)
        _, weights = task.prepare_target(batch)
        assert weights[0, 0, 0] == 0.0


    def test_prepare_target_requires_patch_size(self):
        """Preparing targets without a patch size raises an error."""
        task = BinarySegmentationTask()
        batch = _make_batch(1, 2, 2, 14, lambda b, y, x: 0)

        with pytest.raises(RuntimeError, match="patch size"):
            task.prepare_target(batch)

    def test_get_evaluator_requires_patch_size(self):
        """Getting an evaluator without a patch size raises an error."""
        task = BinarySegmentationTask()

        with pytest.raises(RuntimeError, match="patch size"):
            task.get_evaluator()


class TestBinarySegmentationEvaluator:
    def test_bce(self):
        """BCE is computed from logits."""
        predictions = torch.tensor([[2.0, -2.0]])
        targets = torch.tensor([[1.0, 0.0]])

        evaluator = BinarySegmentationEvaluator(patch_size=1, save_predictions=0)
        evaluator.update({"__key__": ["sample0"]}, predictions, targets)
        summary, _, _ = evaluator.finalize()

        expected = F.binary_cross_entropy_with_logits(
            predictions[0], targets[0]
        ).item()
        assert summary["bce"] == pytest.approx(expected)

    def test_accuracy(self):
        """Test the accuracy calculation."""
        predictions_perfect = torch.tensor([[1.0, -1.0]])
        predictions_wrong = torch.tensor([[-1.0, 1.0]])
        targets = torch.tensor([[1.0, 0.0]])

        evaluator = BinarySegmentationEvaluator(patch_size=1, save_predictions=0)
        evaluator.update({"__key__": ["sample0"]}, predictions_perfect, targets)
        summary_perfect, _, _ = evaluator.finalize()
        assert summary_perfect["accuracy"] == 1.0

        evaluator = BinarySegmentationEvaluator(patch_size=1, save_predictions=0)
        evaluator.update({"__key__": ["sample0"]}, predictions_wrong, targets)
        summary_wrong, _, _ = evaluator.finalize()
        assert summary_wrong["accuracy"] == 0.0

        evaluator = BinarySegmentationEvaluator(patch_size=1, save_predictions=0)
        evaluator.update({"__key__": ["sample0"]}, predictions_perfect, targets)
        evaluator.update({"__key__": ["sample1"]}, predictions_wrong, targets)
        summary, _, _ = evaluator.finalize()
        assert summary["accuracy"] == 0.5

    def test_weight_zero_excludes_patches_from_metrics(self):
        """Patches with weight 0 are excluded from BCE and accuracy."""
        # Two patches: patch 0 is correct (weight 1), patch 1 is wrong (weight 0).
        # With weight 0 masking patch 1, the result should equal a single-patch
        # evaluation on patch 0 alone.
        predictions = torch.tensor([[1.0, -1.0]])   # patch 0 correct, patch 1 wrong
        targets = torch.tensor([[1.0, 1.0]])         # both foreground
        weights = torch.tensor([[1.0, 0.0]])         # ignore patch 1

        evaluator = BinarySegmentationEvaluator(patch_size=1, save_predictions=0)
        evaluator.update({"__key__": ["sample0"]}, predictions, targets, weights)
        summary, _, _ = evaluator.finalize()

        assert summary["accuracy"] == 1.0
        expected_bce = F.binary_cross_entropy_with_logits(
            torch.tensor([1.0]), torch.tensor([1.0])
        ).item()
        assert summary["bce"] == pytest.approx(expected_bce)


def _make_prediction_batch(batch_size: int, num_patches: int, offset: int = 0):
    """Build a batch together with matching predictions, targets and weights."""
    batch = {
        "__key__": [f"sample{offset + i}" for i in range(batch_size)],
        "image": torch.zeros(batch_size, 3, 8, 8, dtype=torch.uint8),
        "segmentation": torch.zeros(batch_size, 1, 8, 8, dtype=torch.uint8),
    }
    prediction = torch.zeros(batch_size, num_patches, 1)
    target = torch.ones(batch_size, num_patches, 1)
    weight = torch.ones(batch_size, num_patches, 1)
    return batch, prediction, target, weight


class TestBinarySegmentationEvaluatorPredictions:
    def test_predictions_are_limited(self):
        """Only the first `save_predictions` samples are kept."""
        evaluator = BinarySegmentationEvaluator(4, save_predictions=3)
        evaluator.update(*_make_prediction_batch(2, 4))
        evaluator.update(*_make_prediction_batch(2, 4, offset=2))

        _, _, predictions = evaluator.finalize()

        assert [record["__key__"] for record in predictions] == [
            "sample0",
            "sample1",
            "sample2",
        ]

    def test_predictions_disabled(self):
        """No predictions are kept when `save_predictions` is 0."""
        evaluator = BinarySegmentationEvaluator(4, save_predictions=0)
        evaluator.update(*_make_prediction_batch(2, 4))

        _, _, predictions = evaluator.finalize()

        assert predictions == []

    def test_prediction_record_contents(self):
        """Each record holds the inputs, the target and the prediction on the CPU."""
        evaluator = BinarySegmentationEvaluator(4, save_predictions=1)
        evaluator.update(*_make_prediction_batch(2, 4))

        _, _, predictions = evaluator.finalize()

        record = predictions[0]
        assert set(record) == {
            "__key__",
            "image",
            "segmentation",
            "target",
            "weight",
            "prediction",
        }
        assert record["image"].shape == (3, 8, 8)
        assert record["prediction"].shape == (2, 2)
        assert all(
            value.device.type == "cpu"
            for value in record.values()
            if isinstance(value, torch.Tensor)
        )
