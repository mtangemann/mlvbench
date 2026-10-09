"""Trainer for fitting probes to pretrained vision models.

The [`ProbeTrainer`][mlvbench.trainer.ProbeTrainer] is the core component that allows
fitting task-specific probes to pretrained vision encoders. Creating a `ProbeTrainer`
requires specifying task, probe type and data module. For example:

```python
data_module = MyDataModule(...)
trainer = ProbeTrainer(
    task: "binary_segmentation",
    probe: "linear",
    data_module: data_module,
    batch_size: 256,
    max_steps: 1000,
)
```

The trainer then allows fitting and evaluating probes for various models:

```python
model = build_model("...")
trainer.fit(model)

# Access the fitted probes via `model.probes`
print(f"Fitted {len(model.probes)} probes to the following layers:")
for probe in model.probes:
    print(probe.layer)

# Evaluate probes
test_loader = data_module.test_loader(batch_size=256)
trainer.evaluate(model, test_loader, "output/example/evaluation")
```

Have a look at the [`ProbeTrainer`][mlvbench.trainer.ProbeTrainer] class for more
options that allow customizing the training process.
"""

import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch

from mlvbench.datasets import DataModule
from mlvbench.models import Features, Model
from mlvbench.monitoring import ThroughputMonitor
from mlvbench.probes import Probe, build_probe
from mlvbench.tasks import Task, build_task

LOGGER = logging.getLogger(__name__)


class ProbeTrainer:
    """Trainer for fitting probes to pretrained vision models."""

    # IDEA Add autotuning option for batch size given effective batch size.
    # IDEA Use proper probe type registry.
    def __init__(
        self,
        data_module: DataModule,
        task: Task | str,
        probe: str,
        batch_size: int,
        max_steps: int,
        optimizer: Literal["adamw", "adam"] = "adamw",
        lr: float | list[float] = 1e-3,
        weight_decay: float | list[float] = 0.0,
        early_stopping_patience: int | None = None,
        effective_batch_size: int | None = None,
        num_workers: int = 0,
        val_every_n_steps: int = 200,
        log_every_n_steps: int = 50,
        enable_throughput_monitor: bool = True,
        probe_types: dict[str, type[Probe]] | None = None,
    ):
        """Initialize the trainer.

        Args:
            data_module: The data module providing training and validation data.
            task: The task to train the probe on. Can be a string (name of the task) or
                a Task object.
            probe: The name of the probe type to train (e.g. `"linear"`). Custom types
                must be registered via `probe_types`.
            batch_size: The batch size for all data loaders.
            max_steps: The maximum number of training steps.
            optimizer: The optimizer to use. Can be "adamw" or "adam".
            lr: The learning rate(s) to use. If a list, different probes are trained for
                each learning rate and the best configuration per layer is selected
                based on validation loss.
            weight_decay: The weight decay value(s) to use. If a list, different probes
                are trained for each weight decay value and the best configuration per
                layer is selected based on the validation loss.
            early_stopping_patience: If set, enable early stopping based on validation
                loss. The best probe state per layer is tracked and restored before the
                final evaluation, and training terminates once every layer has gone
                `early_stopping_patience` consecutive periodic validations without
                improvement.
            effective_batch_size: The effective batch size for gradient accumulation.
                If None, defaults to batch_size (no accumulation). Must be a multiple
                of batch_size.
            num_workers: Number of worker processes for the data loaders.
            val_every_n_steps: Evaluate the probe every N training steps and after
                training completes. If None, the probe is evaluated only after training
                completes.
            log_every_n_steps: Log the training progress every N training steps.
            enable_throughput_monitor: Whether to monitor throughput and memory usage
                during training and to write the recorded statistics to
                `throughput.json`. Disabling the throughput monitor avoids device
                synchronizations (see
                [`ThroughputMonitor`][mlvbench.monitoring.ThroughputMonitor]).
            probe_types: Optional mapping of custom probe type names to `Probe`
                subclasses, merged with the built-in types.
        """
        if effective_batch_size is None:
            effective_batch_size = batch_size
        if effective_batch_size % batch_size != 0:
            raise ValueError(
                f"Effective batch size {effective_batch_size} is not a multiple " +
                f"of the actual batch size {batch_size}"
            )

        self.data_module: DataModule = data_module
        self.task: Task = build_task(task) if isinstance(task, str) else task
        self.probe: str = probe
        self.batch_size: int = batch_size
        self.max_steps: int = max_steps
        self.optimizer:Literal["adamw", "adam"] = optimizer
        self.lr: list[float] = [lr] if isinstance(lr, (int, float)) else list(lr)
        self.weight_decay: list[float] = (
            [weight_decay]
            if isinstance(weight_decay, (int, float))
            else list(weight_decay)
        )
        self.early_stopping_patience: int | None = early_stopping_patience
        self.gradient_accumulation_steps: int = effective_batch_size // batch_size
        self.num_workers: int = num_workers
        self.val_every_n_steps: int = val_every_n_steps
        self.log_every_n_steps: int = log_every_n_steps
        self.enable_throughput_monitor: bool = enable_throughput_monitor
        self.probe_types: dict[str, type[Probe]] = probe_types or dict()

        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    def fit(
        self,
        model: Model,
        layers: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fit probes for the given model.

        Args:
            model: A fully set-up `Model` instance, already placed on its target device
                and optionally `torch.compile`d by the caller (see
                [`build_model`][mlvbench.models.build_model]). The trainer uses the
                model's device for training.
            layers: The model layer names to fit probes for. If None, all layers are
                used.

        Returns:
            The model with the trained probes attached (`model.probes`). The same model
                is also saved to `output_path / "model.pt"`.
        """
        if model.device.type == "cuda" and not model.compiled:
            LOGGER.warning(
                "The model's forward_features is not compiled and might run " +
                "significantly slower. Build the model with " +
                "build_model(..., compile=True) to enable compilation."
            )

        if layers is None:
            layers = model.layer_names

        self.task.configure(model)

        self.data_module.configure(model)
        self.data_module.prepare_data()
        self.data_module.setup()

        probes, throughput = self._fit_probes(model, layers)
        model.probes = probes

        return throughput

    def _fit_probes(
        self,
        model: Model,
        layers: list[str],
    ) -> tuple[list[Probe], list[dict[str, Any]]]:
        """Fit the probes to the training data."""
        LOGGER.info("Fitting probes...")
        start_time = time.perf_counter()

        model.eval()
        probes = self._setup_probes(model, layers)
        device = model.device
        step = 0

        train_loader = self.data_module.train_dataloader(
            batch_size=self.batch_size, num_workers=self.num_workers, device=device
        )
        val_loader = self.data_module.val_dataloader(
            batch_size=self.batch_size, num_workers=self.num_workers, device=device
        )

        monitor = ThroughputMonitor(
            device,
            record_every_n_steps=self.log_every_n_steps
            if self.enable_throughput_monitor
            else None,
        )

        for batch_index, batch in enumerate(monitor.iterate(train_loader)):
            monitor.record_samples(batch["image"].shape[0])

            with monitor.phase("features"), torch.no_grad():
                features = model.forward_features(batch["image"], layers=layers)

            step_optimizer = (batch_index + 1) % self.gradient_accumulation_steps == 0

            with monitor.phase("probes"):
                target, weight = self.task.prepare_target(batch)
                for probe in probes:
                    probe.update(features, target, weight, step_optimizer)

            if not step_optimizer:
                continue

            step += 1

            stats = monitor.step(step)

            if step % self.log_every_n_steps == 0:
                loss = sum(probe.loss_last_step for probe in probes) / len(probes)
                LOGGER.info(
                    "Step %d/%d | loss=%.4f%s",
                    step,
                    self.max_steps,
                    loss,
                    f" | {_format_throughput_stats(stats)}" if stats is not None else "",  # noqa: E501
                )

            if step % self.val_every_n_steps == 0 or step >= self.max_steps:
                self._validate(model, layers, val_loader, probes, step)

            if step >= self.max_steps:
                break

            if all(probe.should_stop() for probe in probes):
                LOGGER.info("Early stopping at step %d", step)
                break

        for probe in probes:
            probe.restore_best_state()

        probes = self._select_best_probes(probes)

        duration = time.perf_counter() - start_time
        LOGGER.info("Fitting probes completed in %.2fs", duration)

        return [trainable.probe for trainable in probes], monitor.records

    @torch.inference_mode()
    def _validate(
        self,
        model: Model,
        layers: list[str],
        val_loader: Iterable[dict],
        probes: list[_ProbeWithTrainingState],
        step: int,
    ) -> None:
        """Validate the probes during training.

        This is a cheap validation step performed during training to control overfitting
        and early stopping. Use `evaluate()` for a comprehensive evaluation.
        """
        LOGGER.info("Validating at step %d...", step)
        start_time = time.perf_counter()

        device = model.device

        for probe in probes:
            probe.eval()

        losses = [torch.tensor(0.0, device=device) for _ in probes]
        num_batches = len(val_loader)

        for batch in val_loader:
            features = model.forward_features(batch["image"], layers=layers)
            target, weight = self.task.prepare_target(batch)
            for probe, loss in zip(probes, losses, strict=True):
                prediction = probe.predict(features)
                batch_loss = self.task.criterion(prediction, target, weight)
                loss += batch_loss / num_batches

        for probe, loss in zip(probes, losses, strict=True):
            probe.record_val_loss(loss.item())
            probe.train()

        duration = time.perf_counter() - start_time
        LOGGER.info(
            "Validation completed in %.2fs | loss=%.4f (best %.4f)",
            duration,
            sum(loss.item() for loss in losses) / len(losses),
            min(probe.best_val_loss for probe in probes),
        )

    def _select_best_probes(
        self,
        probes: list[_ProbeWithTrainingState],
    ) -> list[_ProbeWithTrainingState]:
        """Select the best probe per layer based on validation loss.

        When only a single lr and weight_decay value are configured, returns all probes
        unchanged. Otherwise, picks the probe with the lowest validation loss for each
        layer.
        """
        if len(self.lr) == 1 and len(self.weight_decay) == 1:
            return probes

        best_per_layer: dict[str, _ProbeWithTrainingState] = {}
        for probe in probes:
            layer = probe.probe.layer
            if (
                layer not in best_per_layer
                or probe.best_val_loss < best_per_layer[layer].best_val_loss
            ):
                best_per_layer[layer] = probe

        return list(best_per_layer.values())

    @torch.inference_mode()
    def evaluate(
        self,
        model: Model,
        dataloader: Iterable[dict],
        output_path: Path,
        save_predictions: int = 8,
    ) -> None:
        """Evaluate the attached probes on a single dataset subset.

        This writes three files to the given output path:

        - `results.parquet`: Detailed results with one row per probe and sample.
        - `summary.csv`: Results averaged over the test set; one row per probe.
        - `predictions.pt`: A flat list of input images, targets and probe predictions.
            This allows visualizing results without having to restore
            the model.

        Args:
            model: The model with attached probes to evaluate.
            dataloader: The data loader to evaluate on.
            output_path: The path to save the results to.
            save_predictions: Maximum number of predictions to save in `predictions.pt`.
                Set to 0 to disable.
        """
        for probe in model.probes:
            probe.eval()

        evaluators = [
            self.task.get_evaluator(save_predictions) for _ in model.probes
        ]
        layers = list(dict.fromkeys(probe.layer for probe in model.probes))

        for batch in dataloader:
            features = model.forward_features(batch["image"], layers=layers)
            target, weight = self.task.prepare_target(batch)
            for probe, evaluator in zip(model.probes, evaluators, strict=True):
                prediction = probe(features)
                evaluator.update(batch, prediction, target, weight)

        results = []
        summary = []
        predictions: dict[str, dict[str, Any]] = {}

        for probe, evaluator in zip(model.probes, evaluators, strict=True):
            probe_summary, probe_results, probe_predictions = evaluator.finalize()

            summary.append(
                {
                    "layer": probe.layer,
                    **probe.metadata,
                    **probe_summary,
                }
            )

            probe_results["layer"] = probe.layer
            for key, value in probe.metadata.items():
                probe_results[key] = value
            results.append(probe_results)

            for record in probe_predictions:
                key = record["__key__"]

                # Store constant information only once (input image, target, ...)
                if key not in predictions:
                    predictions[key] = {
                        k: v for k, v in record.items() if k != "prediction"
                    }
                    predictions[key]["predictions"] = []

                # Predictions are stored for each probe individually
                predictions[key]["predictions"].append({
                    "layer": probe.layer,
                    **probe.metadata,
                    "prediction": record["prediction"],
                })

        output_path.mkdir(parents=True, exist_ok=True)

        # The parquet format stores data much more compactly than CSV, leading to 4-5x
        # smaller file sizes.
        pd.concat(results).to_parquet(output_path / "results.parquet")

        # The summary file is small. We use CSV so that we can easily inspect it.
        pd.DataFrame(summary).to_csv(output_path / "summary.csv", index=False)

        # Predictions are stored as a flat list of records so that more predictions can
        # be added later without reloading and rerunning the model.
        torch.save(predictions, output_path / "predictions.pt")

    def _setup_probes(
        self,
        model: Model,
        layers: list[str],
    ) -> list[_ProbeWithTrainingState]:
        """Set up one probe per layer / lr / weight decay combination."""
        probes = []
        for layer in layers:
            for lr in self.lr:
                for weight_decay in self.weight_decay:
                    metadata = {"lr": lr, "weight_decay": weight_decay, "step": 0}
                    probe = build_probe(
                        self.probe,
                        layer,
                        model.embed_dim,
                        bias="scalar",
                        metadata=metadata,
                        probe_types=self.probe_types,
                    ).to(model.device)
                    probes.append(
                        _ProbeWithTrainingState(
                            probe,
                            criterion=self.task.criterion,
                            optimizer=self.optimizer,
                            lr=lr,
                            weight_decay=weight_decay,
                            gradient_accumulation_steps=self.gradient_accumulation_steps,
                            early_stopping_patience=self.early_stopping_patience,
                        )
                    )
        return probes


class _ProbeWithTrainingState:
    """Probe with transient training state.

    The ProbeTrainer fits multiple probes simultaneously in order to reuse the extracted
    features. This class holds the optimizer, loss bookkeeping, and best-state tracking
    needed for each individual probe. This information is only used during training, all
    output information is stored the actual probe.
    """

    def __init__(
        self,
        probe: Probe,
        criterion: torch.nn.Module,
        optimizer: Literal["adamw", "adam"],
        lr: float,
        weight_decay: float,
        gradient_accumulation_steps: int = 1,
        early_stopping_patience: int | None = None,
    ):
        """Initialize the probe training state.

        Args:
            probe: The probe to train. The metadata is expected to carry a `step` key.
            criterion: Loss function to use for training.
            optimizer: Optimizer for the probe parameters ("adamw" or "adam").
            lr: Learning rate for the optimizer.
            weight_decay: Weight decay for the optimizer.
            gradient_accumulation_steps: Number of gradient accumulation steps.
            early_stopping_patience: Number of validation rounds without improvement
                before stopping early.
        """
        self.probe: Probe = probe
        self.criterion: torch.nn.Module = criterion
        self.optimizer: torch.optim.Optimizer = _build_optimizer(
            probe, optimizer, lr, weight_decay
        )
        self.gradient_accumulation_steps: int = gradient_accumulation_steps
        self.early_stopping_patience: int | None = early_stopping_patience

        self.loss_current_step: float = 0.0
        self.loss_last_step: float = float("nan")
        self.best_val_loss: float = float("inf")
        self.best_val_step: float = 0
        self.best_weights: dict[str, torch.Tensor] | None = None
        self.rounds_without_improvement: int = 0

        self.probe.train()

    def train(self) -> "_ProbeWithTrainingState":
        """Set the probe in training mode."""
        self.probe.train()
        return self

    def eval(self) -> "_ProbeWithTrainingState":
        """Set the probe in evaluation mode."""
        self.probe.eval()
        return self

    def predict(self, features: Features) -> torch.Tensor:
        """Predicts the target from the features."""
        return self.probe(features)

    def update(
        self,
        features: Features,
        target: torch.Tensor,
        weight: torch.Tensor | None,
        step_optimizer: bool = True,
    ) -> None:
        """Perform a single forward-backward pass and optionally step the optimizer.

        Args:
            features: Per-layer features extracted from the backbone.
            target: Target tensor with shape `(B, N, ...)`; dtype depends on the task.
            weight: Optional weight tensor with the same shape as target and dtype
                `float32`.
            step_optimizer: Whether to step the optimizer. Setting this to true only
                every N batches enables gradient accumulation.
        """
        prediction = self.predict(features)
        loss = self.criterion(prediction, target, weight)
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps
        loss.backward()
        self.loss_current_step += loss.item()

        if step_optimizer:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.loss_last_step = self.loss_current_step
            self.loss_current_step = 0.0
            self.probe.metadata["step"] += 1

    def record_val_loss(self, loss: float) -> None:
        """Record the validation loss at the current step.

        This information is used for early stopping and selecting the best probe state
        at the end of training.

        Args:
            loss: The validation loss.
        """
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.best_val_step = self.probe.metadata["step"]
            self.best_weights = {
                k: v.detach().clone().cpu() for k, v in self.probe.state_dict().items()
            }
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1

    def should_stop(self) -> bool:
        """Check if the probe can be stopped early based on the validation loss."""
        if self.early_stopping_patience is None:
            return False
        return self.rounds_without_improvement >= self.early_stopping_patience

    def restore_best_state(self) -> None:
        """Restores the probe state with the best validation loss."""
        if self.best_weights is not None:
            self.probe.load_state_dict(self.best_weights)
            self.probe.metadata["step"] = self.best_val_step


def _format_throughput_stats(stats: dict[str, Any]) -> str:
    """Format throughput monitor stats for logging.

    Args:
        stats: Statistics recorded by the
            [`ThroughputMonitor`][mlvbench.monitoring.ThroughputMonitor].

    Returns:
        A single-line summary of the throughput, the per-phase durations, and the GPU
            memory usage. Memory is omitted when training does not run on a GPU.
    """
    parts = [f"{stats['samples_per_sec']:.1f} samples/s"]

    timings = " ".join(
        f"{name.removeprefix('t_')}={value:.2f}s"
        for name, value in stats.items()
        if name.startswith("t_")
    )
    if timings:
        parts.append(timings)

    if stats["gpu_mem_peak_gb"] > 0:
        parts.append(
            f"mem={stats['gpu_mem_gb']:.2f}GB peak={stats['gpu_mem_peak_gb']:.2f}GB"
        )

    return " | ".join(parts)


def _build_optimizer(
    probe: torch.nn.Module,
    optimizer: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build the optimizer for a probe."""
    if optimizer == "adamw":
        return torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "adam":
        return torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
