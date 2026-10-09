"""Tests for image transforms."""

import torch

from mlvbench.stimuli.transforms import BinarizeSegmentation


class TestBinarizeSegmentation:
    def test_binary_mask_maps_to_zero_and_one(self):
        """A 0/255 mask is mapped to 0/1."""
        segmentation = torch.tensor([[[0, 255], [255, 0]]], dtype=torch.uint8)
        result = BinarizeSegmentation()({"segmentation": segmentation})
        expected = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.uint8)
        torch.testing.assert_close(result["segmentation"], expected)

    def test_anti_aliased_mask_is_thresholded(self):
        """Intermediate values are thresholded at 127."""
        segmentation = torch.tensor(
            [[[0, 1, 64, 127, 128, 200, 254, 255]]], dtype=torch.uint8
        )
        result = BinarizeSegmentation()({"segmentation": segmentation})
        expected = torch.tensor([[[0, 0, 0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)
        torch.testing.assert_close(result["segmentation"], expected)

    def test_threshold_is_configurable(self):
        """Values are compared against the given threshold."""
        segmentation = torch.tensor([[[0, 10, 50, 100]]], dtype=torch.uint8)
        result = BinarizeSegmentation(threshold=10)({"segmentation": segmentation})
        expected = torch.tensor([[[0, 0, 1, 1]]], dtype=torch.uint8)
        torch.testing.assert_close(result["segmentation"], expected)

    def test_shape_and_dtype_are_preserved(self):
        """The output has the same shape as the input and dtype uint8."""
        segmentation = torch.randint(0, 256, (1, 17, 23), dtype=torch.uint8)
        result = BinarizeSegmentation()({"segmentation": segmentation})
        assert result["segmentation"].shape == segmentation.shape
        assert result["segmentation"].dtype == torch.uint8

    def test_other_keys_pass_through(self):
        """Values other than the segmentation are passed through unchanged."""
        image = torch.randint(0, 256, (3, 4, 4), dtype=torch.uint8)
        sample = {
            "__key__": "sample",
            "image": image,
            "segmentation": torch.zeros((1, 4, 4), dtype=torch.uint8),
        }
        result = BinarizeSegmentation()(sample)
        assert result["__key__"] == "sample"
        assert result["image"] is image
