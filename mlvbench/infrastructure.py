"""Infrastructure utilities."""

import json
import logging
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mlvbench.models import Model

LOGGER = logging.getLogger(__name__)


def resolve_layers(all_layers: list[str], layers_spec: str | None) -> list[str]:
    """Resolve a layer specification string to a list of layer names.

    Intended for experiment scripts that expose a `--layers` command line argument.
    The resolved list can be passed directly to
    [`ProbeTrainer.fit`][mlvbench.trainer.ProbeTrainer.fit].

    Args:
        all_layers: The full list of layer names from the model
            (`model.layer_names`).
        layers_spec: A specification of a list of layers, provided as a comma-separated
            string. Each value may be a non-negative index (e.g. 0), a negative index
            (e.g. -1), or a layer name (e.g. block.12). If None, all layers are
            returned.

    Returns:
        A deduplicated list of resolved layer names.
    """
    if layers_spec is None:
        return all_layers

    resolved: list[str] = []

    # maps resolved name -> original spec, for duplicate detection
    seen_before: dict[str, str] = {}

    for token in layers_spec.split(","):
        token = token.strip()
        if not token:
            continue

        try:
            index = int(token)
            layer_name = all_layers[index]
        except ValueError:
            if token not in all_layers:
                raise IndexError(
                    f"Layer '{token}' not found in model layers: {all_layers}"
                ) from None
            layer_name = token

        if layer_name in seen_before:
            LOGGER.warning(
                "Layer '%s' (specified as '%s') is a duplicate of '%s' and will " +
                "only be run once.",
                layer_name,
                token,
                seen_before[layer_name],
            )
        else:
            seen_before[layer_name] = token
            resolved.append(layer_name)

    return resolved


def save_metadata(
    output_path: Path,
    parameters: dict[str, Any],
) -> None:
    """Save the metadata for an experiment run."""
    metadata: dict[str, Any] = _get_run_metadata()

    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name()
    else:
        metadata["gpu_name"] = None

    metadata["parameters"] = parameters

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def save_model_info(output_path: Path, model: Model) -> None:
    """Save model info for an experiment run."""
    with open(output_path / "model_info.json", "w") as f:
        json.dump(model.info(), f, indent=2)


def _get_run_metadata() -> dict[str, str | bool | None]:
    """Get the metadata for a run."""
    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        **_git_metadata(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "pytorch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }


def _git_metadata() -> dict[str, str | bool | None]:
    """Returs git commit hash and dirty flag, or None values on failure."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        dirty = (
            subprocess.call(
                ["git", "diff", "--quiet", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            != 0
        )
    except subprocess.SubprocessError, FileNotFoundError:
        commit = None
        dirty = None

    return {
        "git_commit": commit,
        "git_dirty": dirty,
    }
