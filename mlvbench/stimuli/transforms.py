"""Image transforms."""

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class ToTensor:
    """Convert PIL Images in a sample dict to tensors.

    Images with shape `(H, W)` are converted to `(1, H, W)`. Images with shape
    `(H, W, C)` are converted to `(C, H, W)`. The "image" key is additionally
    converted from grayscale to RGB if it has a single channel.

    Non-image values are passed through unchanged.
    """

    def __call__(
        self,
        sample: dict[str, str | Image.Image],
    ) -> dict[str, str | torch.Tensor]:
        """Convert PIL Images in the sample to tensors.

        Args:
            sample: A dictionary whose values may be PIL Images or other types.

        Returns:
            A new dictionary with PIL Images replaced by `uint8` tensors of shape
                `(1, H, W)` or `(C, H, W)` (the "image" key is expanded to `(3, H, W)`).
        """
        new_sample = dict()
        for key, value in sample.items():
            if isinstance(value, Image.Image):
                arr = np.array(value)
                if arr.ndim == 2:
                    new_sample[key] = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
                else:
                    # (C, H, W)
                    new_sample[key] = torch.from_numpy(arr).permute(2, 0, 1)

                if key == "image" and new_sample[key].shape[0] == 1:
                    new_sample[key] = new_sample[key].repeat(3, 1, 1)
            else:
                new_sample[key] = value
        return new_sample


class ToNumpy:
    """Convert tensors in a sample dict to numpy arrays."""

    def __call__(
        self,
        sample: dict[str, str | torch.Tensor],
    ) -> dict[str, str | np.ndarray]:
        """Convert tensors in the sample to numpy arrays.

        Args:
            sample: A dictionary whose values may be tensors.

        Returns:
            A new dictionary with tensors replaced by numpy arrays.
        """
        new_sample = dict()
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                new_sample[key] = value.movedim(0, -1).numpy()
            else:
                new_sample[key] = value
        return new_sample


class Resize:
    """Resize images and segmentation maps so the shorter side equals image_size."""

    def __init__(self, image_size: int):
        """Initialize the transform.

        Args:
            image_size: The target size for the shorter side.
        """
        self.image_size = image_size

    def __call__(
        self,
        sample: dict[str, str | torch.Tensor],
    ) -> dict[str, str | torch.Tensor]:
        """Transform the given sample."""
        key = sample["__key__"]
        image = sample["image"]
        segmentation = sample["segmentation"]

        H, W = image.shape[-2:]

        if H > W:
            new_H = round(self.image_size * H / W)
            new_W = self.image_size
        elif W > H:
            new_H = self.image_size
            new_W = round(self.image_size * W / H)
        else:
            new_H = self.image_size
            new_W = self.image_size

        image = TF.resize(
            image, (new_H, new_W), interpolation=TF.InterpolationMode.BICUBIC
        )
        segmentation = TF.resize(
            segmentation,
            (new_H, new_W),
            interpolation=TF.InterpolationMode.NEAREST,
        )

        return {
            "__key__": key,
            "image": image,
            "segmentation": segmentation,
        }


class CenterCrop:
    """Crop a single square at the center.

    The crop size is the shorter dimension of the input image.
    """

    def __call__(
        self,
        sample: dict[str, str | torch.Tensor],
    ) -> dict[str, str | torch.Tensor]:
        """Transform the given sample."""
        key = sample["__key__"]
        image = sample["image"]
        segmentation = sample["segmentation"]

        H, W = image.shape[-2:]
        size = min(H, W)
        top = (H - size) // 2
        left = (W - size) // 2
        bounding_box = (top, left, size, size)

        return {
            "__key__": key,
            "image": TF.crop(image, *bounding_box),
            "segmentation": TF.crop(segmentation, *bounding_box),
        }
