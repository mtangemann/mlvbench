"""Run the probe training for a single model."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
import torch
from baselines import Baseline, iter_baselines

from mlvbench.datasets import (
    DataModule,
    FigureGroundConvexityDataModule,
    FigureGroundSurroundednessDataModule,
    FigureGroundSymmetryDataModule,
    PrecomputedMSRA10KDataModule,
)
from mlvbench.infrastructure import resolve_layers, save_metadata, save_model_info
from mlvbench.models import Model, build_model
from mlvbench.tasks import BinarySegmentationTask
from mlvbench.trainer import ProbeTrainer

LOGGER = logging.getLogger(__name__)


Condition = Literal["natural", "surroundedness", "convexity", "symmetry"]
_DATA_MODULES = {
    "natural": PrecomputedMSRA10KDataModule,
    "surroundedness": FigureGroundSurroundednessDataModule,
    "convexity": FigureGroundConvexityDataModule,
    "symmetry": FigureGroundSymmetryDataModule,
}


@dataclass
class Config:
    """Experiment configuration."""

    condition: Condition
    model: str
    pretrained: bool = True
    model_seed: int = 0
    layers: str | None = None
    batch_size: int = 256
    effective_batch_size: int = 256
    lr: float | list[float] = 0.001
    weight_decay: float | list[float] = 0.0
    max_steps: int = 2000
    num_workers: int = 0
    val_every_n_steps: int = 200
    early_stopping_patience: int | None = None
    precision: Literal["auto", "float32", "bfloat16"] = "auto"
    save_predictions: int = 8


def warmup(config: Config) -> None:
    """Prepare the model and data for the experiment."""
    model = build_model(
        config.model, pretrained=config.pretrained, seed=config.model_seed
    )

    # Probes trained on natural data are evaluated zero-shot for all other conditions,
    # so we need to prepare all data modules.
    if config.condition == "natural":
        conditions = _DATA_MODULES.keys()
    else:
        conditions = [config.condition]

    for condition in conditions:
        data_module = _DATA_MODULES[condition]()
        data_module.configure(model)
        data_module.prepare_data()


def run_model(config: Config, output_path: Path) -> None:
    """Train and evaluate the probe for a single configuration.

    Args:
        config: The configuration to run.
        output_path: The directory that holds all outputs of this job. Probe training
            outputs and the validation results are written to `output_path / "probe"`,
            evaluations on the test sets are save in `output_path / "evaluations" .
    """
    probe_path = output_path / "probe"
    probe_path.mkdir(parents=True, exist_ok=True)
    save_metadata(probe_path, asdict(config))

    data_module = _DATA_MODULES[config.condition]()
    trainer = ProbeTrainer(
        task="binary_segmentation",
        probe="linear",
        data_module=data_module,
        batch_size=config.batch_size,
        optimizer="adamw",
        lr=config.lr,
        weight_decay=config.weight_decay,
        max_steps=config.max_steps,
        val_every_n_steps=config.val_every_n_steps,
        early_stopping_patience=config.early_stopping_patience,
        effective_batch_size=config.effective_batch_size,
        num_workers=config.num_workers,
    )
    model = build_model(
        config.model,
        pretrained=config.pretrained,
        seed=config.model_seed,
        precision=config.precision,
        compile=True,
    )
    save_model_info(output_path, model)
    layers = resolve_layers(model.layer_names, config.layers)
    throughput = trainer.fit(model, layers=layers)
    model.save(probe_path / "model.pt")
    with open(probe_path / "throughput.json", "w") as file:
        json.dump(throughput, file)

    # The data module is configured, prepared and set up by `fit`, so we can build data
    # loaders from it here.
    dataloader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "device": model.device,
    }

    LOGGER.info(f"Evaluating '{config.condition}/val' ....")
    val_loader = data_module.val_dataloader(**dataloader_kwargs)
    trainer.evaluate(model, val_loader, probe_path, config.save_predictions)

    # Probes trained on natural data are evaluated zero-shot for all other conditions,
    # in all other cases we only evaluate in domain.
    if config.condition == "natural":
        eval_conditions = _DATA_MODULES.keys()
    else:
        eval_conditions = [config.condition]

    for condition in eval_conditions:
        # configure, prepare_data and setup are cheap when the data is prepared, so we
        # keep the code simple and recreate the training data module.
        data_module = _DATA_MODULES[condition]()
        data_module.configure(model)
        data_module.prepare_data()
        data_module.setup()

        test_subsets = [
            subset for subset in data_module.subsets() if subset.startswith("test")
        ]
        for subset in test_subsets:
            LOGGER.info(f"Evaluating '{condition}/{subset}' ....")
            subset_path = output_path / "evaluations" / f"{condition}_{subset}"
            subset_path.mkdir(parents=True, exist_ok=True)
            test_loader = data_module.test_dataloader(
                subset=subset, **dataloader_kwargs
            )
            trainer.evaluate(
                model, test_loader, subset_path, config.save_predictions
            )

    LOGGER.info("Done.")


def run_baselines(config: Config, results_path: Path) -> None:
    """Evaluate all baselines for a single configuration.

    Args:
        config: The configuration to run. All baselines will use the same input size and
            patch size as the specified model.
        results_path: The root directory for results (`<workspace>/results`).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(config.model, pretrained=False, seed=config.model_seed)

    for baseline in iter_baselines(model.input_size, model.patch_size):
        # Roughly mirror the naming pattern of timm checkpoints.
        name = f"baseline_{baseline.name}_patch{model.patch_size}_{model.input_size}"
        output_path = results_path / name / config.condition

        if (output_path / "done").exists():
            LOGGER.info(
                "Skipping baseline '%s' (%s): already done", name, config.condition
            )
            continue

        LOGGER.info("Evaluating baseline '%s' (%s) ...", name, config.condition)

        task = BinarySegmentationTask()
        task.configure(baseline)
        baseline.fit(_build_data_module(config.condition, baseline))

        for eval_condition in list(_DATA_MODULES):
            data_module = _build_data_module(eval_condition, baseline)
            test_subsets = [
                subset
                for subset in data_module.subsets()
                if subset.startswith("test")
            ]
            for subset in test_subsets:
                LOGGER.info("  '%s/%s' ...", eval_condition, subset)
                subset_path = output_path / "evaluations" / f"{eval_condition}_{subset}"
                test_loader = data_module.test_dataloader(
                    subset=subset, batch_size=config.batch_size, device=device
                )
                _evaluate_baseline_on_subset(
                    baseline, task, test_loader, subset_path, config.save_predictions
                )

        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "done", "w") as file:
            file.write("completed\n")


# TODO Reduce duplicated evaluation code (also in ProbeTrainer). Baselines and models...
#      are different, so the evaluation code is not 100% the same. But it should be
#      possible to substantially reduce duplication.
def _evaluate_baseline_on_subset(
    baseline: Baseline,
    task: BinarySegmentationTask,
    dataloader: Iterable[dict],
    output_path: Path,
    save_predictions: int,
) -> None:
    """Evaluate a baseline on a single subset and write the result files.

    Writes the output in the same format as
    [`ProbeTrainer.evaluate`][mlvbench.trainer.ProbeTrainer.evaluate], but with a
    `baseline` column instead of `layer` and probe metadata.
    """
    evaluator = task.get_evaluator(save_predictions=save_predictions)

    for batch in dataloader:
        target, weight = task.prepare_target(batch)
        prediction = baseline.predict(batch, target)
        evaluator.update(batch, prediction, target, weight)

    summary, results, predictions = evaluator.finalize()

    results["baseline"] = baseline.name

    nested: dict[str, dict] = {}
    for record in predictions:
        key = record["__key__"]
        nested[key] = {k: v for k, v in record.items() if k != "prediction"}
        nested[key]["predictions"] = [
            {"baseline": baseline.name, "prediction": record["prediction"]}
        ]

    output_path.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output_path / "results.parquet")
    pd.DataFrame([{"baseline": baseline.name, **summary}]).to_csv(
        output_path / "summary.csv", index=False
    )
    torch.save(nested, output_path / "predictions.pt")


def _build_data_module(
    condition: Condition,
    model_or_baseline: Model | Baseline,
) -> DataModule:
    data_module = _DATA_MODULES[condition]()
    data_module.configure(model_or_baseline)
    data_module.prepare_data()
    data_module.setup()
    return data_module


if __name__ == "__main__":
    import coloredlogs
    import tyro

    coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Extend the config with a default output path for the auto-generated CLI. Calling
    # this script directly is intended for debugging, run manual experiments in a
    # workspace through `launch.py`.
    @dataclass
    class _ExtendedConfig(Config):
        output_path: Path = Path(__file__).parent / "output" / "debug"

    config = tyro.cli(_ExtendedConfig)
    output_path = config.output_path
    config = Config(**{k: v for k, v in asdict(config).items() if k != "output_path"})

    run_model(config, output_path)
