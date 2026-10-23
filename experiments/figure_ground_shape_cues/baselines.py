"""Baselines for the figure-ground shape-cues experiment."""

import logging
from abc import ABC, abstractmethod
from typing import Generator

import torch

from mlvbench.datasets import DataModule
from mlvbench.tasks.binary_segmentation import BinarySegmentationTask

LOGGER = logging.getLogger(__name__)


def list_baselines() -> list[str]:
    """Return the names of all available baselines."""
    return list(_BASELINES.keys())


def build_baseline(name: str, input_size: int, patch_size: int) -> Baseline:
    """Build the baseline with the given name."""
    return _BASELINES[name](input_size, patch_size)


def iter_baselines(input_size: int, patch_size: int) -> Generator[Baseline, None, None]:
    """Return a generator over all baselines with the given config."""
    for baseline_cls in _BASELINES.values():
        yield baseline_cls(input_size=input_size, patch_size=patch_size)


class Baseline(ABC):
    """Base class for figure-ground baselines."""

    name: str

    def __init__(self, input_size: int, patch_size: int) -> None:
        """Initialize the baseline.

        The input size and patch size should be chosen to match the model to compare to.

        Args:
            input_size: The input image size.
            patch_size: The patch size to predict at.
        """
        self.input_size = input_size
        self.patch_size = patch_size

    def fit(self, data_module: DataModule) -> None:  # noqa: B027
        """Fit the baseline to the training data (default: no-op).

        Args:
            data_module: A configured and prepared data module. The training split may
                be used for fitting the baseline.
        """

    @abstractmethod
    def predict(
        self, batch: dict[str, torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        """Predict per-patch foreground logits for a batch.

        Args:
            batch: A batch providing `__key__`, `image`, and `segmentation`.
            target: Target binary segmentation mask with shape `(B, N, 1)` and
                dtype `uint8`, values in `[0, 1]`. Baselines may use the target to
                simulate prediction strategies.

        Returns:
            Per-patch logits with shape `(B, N, 1)`, on `target.device`.
        """
        raise NotImplementedError


class SpatialPriorBaseline(Baseline):
    """Predicts the per-patch foreground frequency of the training set.

    For each patch location, we count how often the patch at this location is
    foreground and predict the resulting probabilities for each test image.
    """

    name = "spatial_prior"

    def __init__(self, input_size: int, patch_size: int, batch_size: int = 256) -> None:
        """Initialize the baseline."""
        super().__init__(input_size, patch_size)
        self.prior_logits: torch.Tensor | None = None
        self.batch_size: int = batch_size

    def fit(self, data_module: DataModule) -> None:
        """Fit the per-location foreground prior logits on the training set.

        Args:
            data_module: A configured and prepared data module.
        """
        task = BinarySegmentationTask(patch_size=self.patch_size)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader = data_module.train_dataloader(
            batch_size=self.batch_size, device=device, repeat=False
        )

        target_sum = None
        weight_sum = None

        for batch in train_loader:
            target, weight = task.prepare_target(batch)  # (B, N, 1), (B, N, 1)

            if target_sum is None or weight_sum is None:
                num_patches = target.shape[1]
                target_sum = torch.zeros(num_patches, 1, device=device)
                weight_sum = torch.zeros(num_patches, 1, device=device)

            target_sum += (target * weight).sum(dim=0)
            weight_sum += weight.sum(dim=0)

        if target_sum is None or weight_sum is None:
            raise RuntimeError("Empty training set; cannot fit the prior.")

        rate = target_sum / weight_sum.clamp(min=1.0)
        self.prior_logits = torch.logit(rate, eps=1e-6)  # (N, 1)

    def predict(
        self, batch: dict[str, torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        """Broadcast the fitted per-location logits over the batch."""
        if self.prior_logits is None:
            raise RuntimeError("SpatialPriorBaseline.fit must run before predict.")
        prior_logits = self.prior_logits.to(target.device)
        return prior_logits.expand(target.shape).contiguous()


_BASELINES: dict[str, type[Baseline]] = {
    baseline_cls.name: baseline_cls
    for baseline_cls in (
        SpatialPriorBaseline,
    )
}
