"""Baselines for the figure-ground shape-cues experiment."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator

import numpy as np
import torch
import torch.nn.functional as F

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


# -------------------------------------------------------------------------------------
# Spatial prior
# -------------------------------------------------------------------------------------

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


# -------------------------------------------------------------------------------------
# Region properties
# -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Regions:
    """The textured regions of a batch of stimuli."""

    masks: torch.Tensor
    """Pixel masks of the regions, shape `(B, K, H, W)`, dtype `bool`."""

    patches: torch.Tensor
    """Region label of each patch, shape `(B, N)`, dtype `int64`."""

    valid: torch.Tensor
    """Whether each region occurs in the sample, shape `(B, K)`, dtype `bool`."""

    @property
    def num_regions(self) -> int:
        """Return the number of region labels `K`."""
        return self.masks.shape[1]

    @property
    def labels(self) -> torch.Tensor:
        """Return the region labels `0, ..., K - 1` with shape `(K,)`."""
        return torch.arange(self.num_regions, device=self.masks.device)


def _extract_regions(batch: dict[str, torch.Tensor], patch_size: int) -> Regions:
    """Extract the textured regions of a batch.

    Args:
        batch: A batch from the data module providing `segmentation` and optionally
            `texture_segmentation` with shape `(B, 1, H, W)`.
        patch_size: The patch size at which region labels are assigned to patches.

    Returns:
        The regions of the batch. `K` is the number of labels occurring anywhere in the
            batch, so regions missing from an individual sample are marked as invalid.

    Raises:
        ValueError: If the region map holds more than 64 distinct labels (likely an
            interpolated mask).
    """
    segmentation = batch.get("texture_segmentation", batch["segmentation"])
    values, region = torch.unique(segmentation, return_inverse=True)
    num_regions = values.numel()

    max_regions = 64
    if num_regions > max_regions:
        raise ValueError(
            f"Found {num_regions} distinct region labels, which exceeds the " +
            f"limit of {max_regions}. Expected a segmentation into a small number of " +
            "textured regions."
        )

    region = region.type(torch.uint8)  # (B, 1, H, W)
    labels = torch.arange(num_regions, device=region.device, dtype=region.dtype)

    masks = region == labels.view(1, -1, 1, 1)  # (B, K, H, W)
    valid = masks.flatten(2).any(dim=-1)  # (B, K)

    # Assign each patch the label of the majority of its pixels, mirroring how
    # `BinarySegmentationTask.prepare_target` derives the per-patch target. Patches
    # covering more than one region are excluded from the metrics by the task weights.
    patches = F.unfold(region.float(), kernel_size=patch_size, stride=patch_size)
    patches = patches.type(torch.uint8).mode(dim=1)[0].long()  # (B, N)

    return Regions(masks=masks, patches=patches, valid=valid)


class RegionPropertyBaseline(Baseline, ABC):
    """Predict one of the textured regions based on a scalar property.

    This is the basis for several baselines that simulate segmenting the input image by
    texture first and then selecting one of the resulting regions as the figure, based
    on a property of that region (e.g., its area or overlap with the center).
    """

    # Magnitude of a confident 0/1 baseline prediction in logit space. Large enough that
    # `accuracy` / `iou` / `miou` behave like a hard label; small enough that `bce` /
    # `bce_inverse` on wrong patches stay in a readable range.
    _hard_logit: float = 10.0

    def __init__(
        self, input_size: int, patch_size: int, tolerance: float = 1e-3, seed: int = 0,
    ) -> None:
        """Initialize the baseline.

        Args:
            input_size: The input image size.
            patch_size: The patch size to predict at.
            tolerance: Relative tolerance below which region scores count as tied.
            seed: Seed for the random selection among tied regions.
        """
        super().__init__(input_size, patch_size)
        self.tolerance = tolerance
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def score(
        self, batch: dict[str, torch.Tensor], regions: Regions
    ) -> torch.Tensor:
        """Return the property value of each region.

        Args:
            batch: A batch from the data module providing `__key__`, `image`,
                `segmentation`, and `texture_segmentation`.
            regions: The regions of the batch, as extracted from the texture
                segmentation.

        Returns:
            Property values with shape `(B, K)`, where index `k` holds the value of the
                region labelled `k`. Values of regions that are not valid for a sample
                are ignored.
        """

    def predict(
        self, batch: dict[str, torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        """Predict a constant map naming the highest-scoring region as figure."""
        regions = _extract_regions(batch, self.patch_size)
        scores = self.score(batch, regions)
        scores = scores.to(device=target.device, dtype=torch.float32)

        figure_label = self._select_figure(scores, regions.valid.to(target.device))
        is_figure = regions.patches.to(target.device) == figure_label.view(-1, 1)

        return torch.where(
            is_figure.unsqueeze(-1),
            target.new_full((), self._hard_logit),
            target.new_full((), -self._hard_logit),
        )

    def _select_figure(
        self, scores: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Return the label of the figure region for each sample.

        The highest-scoring valid region is selected. Regions scoring within `tolerance`
        of the best region, relative to the largest score magnitude, count as tied and
        one of them is selected uniformly at random.

        Args:
            scores: Region scores with shape `(B, K)`.
            valid: Mask of the regions occurring in each sample with shape `(B, K)`.

        Returns:
            The selected region labels with shape `(B,)`.
        """
        scores = scores.masked_fill(~valid, -torch.inf)
        best = scores.amax(dim=1, keepdim=True)
        scale = scores.masked_fill(~valid, 0.0).abs().amax(dim=1, keepdim=True)
        tied = valid & (best - scores <= self.tolerance * scale)

        # Taking the argmax of uniform noise over the tied regions selects one of them
        # uniformly at random.
        noise = torch.from_numpy(self.rng.random(tuple(scores.shape)))
        noise = noise.to(device=scores.device, dtype=scores.dtype)
        return noise.masked_fill(~tied, -torch.inf).argmax(dim=1)


class RegionAreaBaseline(RegionPropertyBaseline):
    """Area of the region in pixels."""

    name = "region_area"

    def score(
        self, batch: dict[str, torch.Tensor], regions: Regions
    ) -> torch.Tensor:
        """Count the pixels of each region."""
        return regions.masks.flatten(2).sum(dim=-1, dtype=torch.float32)


class RegionCenterOverlapBaseline(RegionPropertyBaseline):
    """Number of the central 2x2 pixels covered by the region."""

    name = "region_center_overlap"

    def score(
        self, batch: dict[str, torch.Tensor], regions: Regions
    ) -> torch.Tensor:
        """Count the central pixels covered by each region."""
        center = self.input_size // 2
        central = regions.masks[..., center - 1 : center + 1, center - 1 : center + 1]
        return central.flatten(2).sum(dim=-1, dtype=torch.float32)


class RegionCentroidOffsetBaseline(RegionPropertyBaseline):
    """Distance of the region centroid from the image center, in pixels."""

    name = "region_centroid_offset"

    def score(
        self, batch: dict[str, torch.Tensor], regions: Regions
    ) -> torch.Tensor:
        """Measure how far the centroid of each region sits from the image center."""
        row_counts = regions.masks.sum(dim=-1, dtype=torch.float32)  # (B, K, H)
        column_counts = regions.masks.sum(dim=-2, dtype=torch.float32)  # (B, K, W)

        height, width = regions.masks.shape[-2:]
        rows = torch.arange(height, device=regions.masks.device, dtype=torch.float32)
        columns = torch.arange(width, device=regions.masks.device, dtype=torch.float32)

        area = row_counts.sum(dim=-1).clamp(min=1.0)  # (B, K)

        mean_row = (row_counts * rows).sum(dim=-1) / area
        mean_column = (column_counts * columns).sum(dim=-1) / area

        center = (regions.masks.shape[-1] - 1) / 2
        return torch.hypot(mean_row - center, mean_column - center)


_BASELINES: dict[str, type[Baseline]] = {
    baseline_cls.name: baseline_cls
    for baseline_cls in (
        SpatialPriorBaseline,
        RegionAreaBaseline,
        RegionCenterOverlapBaseline,
        RegionCentroidOffsetBaseline,
    )
}
