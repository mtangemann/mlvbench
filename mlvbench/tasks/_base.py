"""Task interface."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import pandas as pd
import torch

from mlvbench.models._base import Model


class Task(ABC):
    """Base class for tasks.

    A task specifies the prediction target, the loss function and the evaluator for
    fitting probes.
    """

    def configure(self, model: Model) -> None:  # noqa: B027
        """Configure the task for the given model.

        Called by the trainer before any data preparation or probe fitting. Override to
        store model attributes (e.g., patch size) that are needed during target
        preparation or evaluation.

        Args:
            model: The model whose features will be probed.
        """
        pass

    @abstractmethod
    def prepare_target(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare the target for the probe.

        Args:
            batch: A batch from the dataset.

        Returns:
            A 2-tuple of (target, weight). target is a `float32` tensor whose shape
                depends on the task (e.g. `(B, N, 1)`). weight is None if all samples
                should be weighted equally, otherwise a `float32` tensor with the same
                shape as target holding per-element weights.
        """

    @abstractmethod
    def criterion(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Loss function for the task.

        Args:
            prediction: The prediction from the probe, a `float32` tensor whose shape
                depends on the task.
            target: The target for the task, a `float32` tensor whose shape depends on
                the task.
            weight: Optional per-element `float32` weight tensor with the same shape as
                target.

        Returns:
            The scalar loss for the task, a `float32` tensor of shape `()`.
        """

    @abstractmethod
    def get_evaluator(self, save_predictions: int = 8) -> "Evaluator":
        """Return an evaluator for the task.

        Args:
            save_predictions: The maximum number of samples for which the evaluator
                keeps raw predictions. Set to 0 to disable.
        """

    def has_prior(self) -> bool:
        """Return True if this task supports fitting a prior."""
        return False

    def fit_prior(
        self, train_dataloader: Iterable, device: torch.device
    ) -> torch.Tensor:
        """Fit the prior from the training data.

        Called by the trainer before probe training when has_prior() returns True.

        Args:
            train_dataloader: A non-repeating data loader over the training set.
            device: The device to use for the prior.
        """
        raise NotImplementedError


class Evaluator(ABC):
    """Base class for evaluators.

    An evaluator is used to compute and aggregate performance metrics over a dataset
    (subset). The following simplified code shows how the evaluator is used by the
    trainer:

    ```python
    evaluator = task.get_evaluator()

    for batch in dataset:
        ...
        evaluator.update(batch, prediction, target, weight)

    summary, results, predictions = evaluator.finalize()
    ```
    """

    @abstractmethod
    def update(
        self,
        batch: dict[str, Any],
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> None:
        """Update the evaluator with data from a single batch."""

    @abstractmethod
    def finalize(
        self,
    ) -> tuple[dict[str, float], pd.DataFrame, list[dict[str, Any]]]:
        """Finalize the evaluation and return the aggregated results.

        Returns:
            A 3-tuple of (summary, results, predictions). summary holds the aggregated
                metrics for the whole subset, results holds one row per sample, and
                predictions holds one record per stored sample as returned by
                `collect_predictions`. predictions may be an empty list.
        """
