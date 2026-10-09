"""Binary segmentation task."""

from typing import Any, override

import pandas as pd
import torch
import torch.nn.functional as F

from mlvbench.models._base import Model
from mlvbench.tasks._base import Evaluator, Task


class BinarySegmentationTask(Task):
    """Binary segmentation task.

    Predict whether each patch belongs to the foreground (segmentation > 0).
    """

    def __init__(self, patch_size: int | None = None) -> None:
        """Initialize the task.

        Args:
            patch_size: The patch size of the model. Alternatively, this can be
                specified later by calling `configure`. The
                [`Trainer`](mlvbench.trainer.Trainer) always calls `configure`, so the
                the patch size doesn't have to be specified for probe training.
        """
        self.patch_size = patch_size

    def configure(self, model: Model) -> None:
        """Configure the task for the given model."""
        self.patch_size = model.patch_size

    def prepare_target(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare the target for the probe.

        Args:
            batch: A batch from the dataset.

        Returns:
            A tuple of (target, weights) where target has shape `(B, N, 1)` and dtype
                `float32` with 1.0 where the patch belongs to the foreground and 0.0
                otherwise, and weight has shape `(B, N, 1)` and dtype `float32` with 0.0
                for patches that contain more than one segment label and 1.0 otherwise.
        """
        if self.patch_size is None:
            raise RuntimeError(
                "The patch size is not set. Pass `patch_size` to the constructor or "
                "call `configure` first."
            )

        segmentation = batch["segmentation"]

        assert segmentation.shape[-2] % self.patch_size == 0
        assert segmentation.shape[-1] % self.patch_size == 0
        segmentation = F.unfold(
            segmentation.float(),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        segmentation = segmentation.type(torch.uint8)  # B, P*P, N

        mode = segmentation.mode(dim=1)[0]  # B, N
        targets = (mode > 0).float().unsqueeze(-1)  # B, N, 1

        # Only consider "pure" patches where all pixels belong to the same segment.
        pure = segmentation.min(dim=1)[0] == segmentation.max(dim=1)[0]  # B, N

        # Combine with a mask of ignored pixels if provided.
        if "ignore_mask" in batch:
            ignore_mask = batch["ignore_mask"]  # B, 1, H, W (uint8, 1 = ignore)
            ignore_mask = F.unfold(
                ignore_mask.float(),
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            ignore_mask = ignore_mask.max(dim=1)[0] > 0
            weights = (pure & ~ignore_mask).float().unsqueeze(-1)
        else:
            weights = pure.float().unsqueeze(-1)

        return targets, weights

    def criterion(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Loss function for the task.

        Args:
            prediction: Probe output logits with shape `(B, N, 1)` and dtype `float32`.
            target: Binary target labels with shape `(B, N, 1)` and dtype `float32`.
            weight: Optional per-patch weights with shape `(B, N, 1)` and dtype
                `float32`. Patches with weight 0.0 are excluded from the loss.

        Returns:
            The scalar binary cross-entropy loss, a `float32` tensor of shape `()`.
        """
        return F.binary_cross_entropy_with_logits(prediction, target, weight=weight)

    @override
    def get_evaluator(
        self,
        save_predictions: int = 8,
    ) -> "BinarySegmentationEvaluator":
        """Return an evaluator for the task.

        Args:
            save_predictions: The maximum number of samples for which the evaluator
                keeps raw predictions. Set to 0 to disable.
        """
        if self.patch_size is None:
            raise RuntimeError(
                "The patch size is not set. Pass `patch_size` to the constructor or "
                "call `configure` first."
            )

        return BinarySegmentationEvaluator(self.patch_size, save_predictions)


class BinarySegmentationEvaluator(Evaluator):
    """Evaluator for binary segmentation tasks."""

    def __init__(
        self,
        patch_size: int,
        save_predictions: int = 8,
    ) -> None:
        """Initialize the evaluator.

        Args:
            patch_size: The patch size of the model, used to reconstruct the spatial
                layout of patches for visualization.
            save_predictions: The maximum number of samples for which to keep raw
                predictions. Samples are taken in the order they are passed to
                `update`. Set to 0 to disable.
        """
        self.patch_size = patch_size
        self.save_predictions = save_predictions
        self.results = []
        self.predictions = []

    def update(
        self,
        batch: dict[str, Any],
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> None:
        """Update the evaluator with predictions and targets for a single batch.

        Args:
            batch: A batch from the dataset. This evaluator expects a list of sample
                keys as `batch["__key__"]`.
            prediction: The prediction from the probe with shape `(B, N, 1)` and dtype
                `float32`. Values are logits.
            target: The target for the task with shape `(B, N, 1)` and dtype `float32`
                with binary values 0.0 and 1.0.
            weight: Optional per-patch weight tensor with shape `(B, N, 1)` and dtype
                `float32`. Patches with weight 0.0 are excluded from the metrics.
        """
        for i, key in enumerate(batch["__key__"]):
            self._update_single(
                key,
                prediction[i],
                target[i],
                weight[i] if weight is not None else None,
            )

            if len(self.predictions) < self.save_predictions:
                image_height, image_width = batch["image"].shape[-2:]
                target_height = image_height // self.patch_size
                target_width = image_width // self.patch_size

                self.predictions.append({
                    "__key__": key,
                    "image": batch["image"][i].cpu(),
                    "segmentation": batch["segmentation"][i].cpu(),
                    "target": target[i].view(target_height, target_width).cpu(),
                    "weight": (
                        None if weight is None
                        else weight[i].view(target_height, target_width).cpu()
                    ),
                    "prediction": prediction[i].view(target_height, target_width).cpu()
                })

    def _update_single(
        self,
        key: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> None:
        """Update with predictions and targets for a single sample."""
        if weight is not None:
            mask = weight.squeeze(-1) > 0
            prediction = prediction[mask]
            target = target[mask]

        bce = F.binary_cross_entropy_with_logits(prediction, target).item()
        accuracy = ((prediction > 0.0) == (target > 0.5)).float().mean().item()
        entry = {"__key__": key, "bce": bce, "accuracy": accuracy}

        self.results.append(entry)

    def finalize(self) -> tuple[dict[str, float], pd.DataFrame, list[dict[str, Any]]]:
        """Finalize the evaluation and return summary, results and predictions."""
        df = pd.DataFrame(self.results)
        summary = {
            "loss": df["bce"].mean(),
            "bce": df["bce"].mean(),
            "accuracy": df["accuracy"].mean(),
        }
        return summary, df, self.predictions
