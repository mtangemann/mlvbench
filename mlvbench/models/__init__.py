"""Models.

This package provides a unified interface to pretrained Vision Transformers (ViTs) for
extracting intermediate features.

Models are referenced by a `"provider/model_name"` string (e.g.
`"timm/vit_base_patch16_224.dino"`). Use
[`build_model`][mlvbench.models.build_model] to instantiate one, and
[`list_models`][mlvbench.models.list_models] or
[`list_bundles`][mlvbench.models.list_bundles] to discover what is available. The
supported providers are `timm`, `franca`, `webssl`, and `diffusers`.

[`Model.forward_features`][mlvbench.models.Model.forward_features] takes a
`(B, 3, H, W)` image batch and returns a
[`Features`][mlvbench.models.Features] container keyed by layer name. ViT features are a
flat sequence of tokens — patch tokens plus optional special tokens such as cls or
register tokens, whose arrangement is described by a
[`TokenLayout`][mlvbench.models.TokenLayout]. `Features` provides named access to these
token types via `.patch()`, `.cls()`, and `.register()`:

```python
from mlvbench.models import build_model

model = build_model("timm/vit_base_patch16_224.dino")
features = model.forward_features(images)            # images: (B, 3, H, W)
patches = features.patch("block.11", format="BCHW")  # (B, C, H, W)
cls = features.cls("block.11")                       # (B, C)
```

Every model can also be built with randomly initialized weights instead of the
pretrained ones, which is useful as a control condition. The architecture and all
metadata stay the same; passing a seed makes the initialization reproducible:

```python
model = build_model("timm/vit_base_patch16_224.dino", pretrained=False, seed=0)
```

Trained [`Probe`s][mlvbench.probes.Probe] can be attached to a model to read out from
these features, and a model together with its probes can be round-tripped with
[`Model.save`][mlvbench.models.Model.save] and
[`load_model`][mlvbench.models.load_model]. See [`mlvbench.probes`][mlvbench.probes] for
the probing API.

See the symbol reference below for full details.
"""

import fnmatch
import logging
from pathlib import Path
from typing import Any, Literal

import torch

from mlvbench.probes import Probe, build_probe

from ._base import CHECKPOINT_FORMAT_VERSION, Features, Model, TokenLayout
from ._bundles import BUNDLES
from .diffusers import DiffusersModel, list_diffusers_models
from .franca import FrancaModel, list_franca_models
from .timm import TimmModel, list_timm_models
from .webssl import WebSSLModel, list_webssl_models

__all__ = [
    "Features",
    "Model",
    "TokenLayout",
    "build_model",
    "list_bundles",
    "list_models",
    "load_model",
]

LOGGER = logging.getLogger(__name__)


def build_model(
    name: str,
    pretrained: bool = True,
    precision: Literal["auto", "float32", "bfloat16"] = "auto",
    device: torch.device | str = "auto",
    compile: bool = False,
    seed: int = 0,
    **kwargs: Any,
) -> Model:
    """Build the model with the given name.

    Args:
        name: Model name in "provider/model_name" format.
        pretrained: If True (default), load the pretrained weights. If False, the same
            architecture is built with randomly initialized weights.
        precision: The compute precision of the model. `"auto"` keeps the checkpoint
            dtype, `"float32"` and `"bfloat16"` cast the weights accordingly.
        device: The device to load the model onto. `"auto"` selects CUDA if available
            and falls back to the CPU.
        compile: If True, compile the model's `forward_features` with `torch.compile`.
            Only the base model's feature extraction is compiled; attached probes are
            invoked separately and remain eager. The returned object stays a plain
            `Model` instance.
        seed: Seed for the random weight initialization when `pretrained=False`.
        **kwargs: Additional model-specific options forwarded to the model constructor.
            Passing an option that the selected model does not support raises a
            `TypeError`.

    Returns:
        The model instance.
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    provider, model_name = name.split("/", maxsplit=1)
    kwargs = {"pretrained": pretrained, "precision": precision, "seed": seed, **kwargs}
    match provider:
        case "timm":
            model = TimmModel(model_name, **kwargs)
        case "franca":
            model = FrancaModel(model_name, **kwargs)
        case "webssl":
            model = WebSSLModel(model_name, **kwargs)
        case "diffusers":
            model = DiffusersModel(model_name, **kwargs)
        case _:
            raise ValueError(f"Unsupported model: {name}")

    model = model.to(device)
    if compile:
        model.forward_features = torch.compile(model.forward_features)
    return model


def load_model(
    path: Path | str,
    device: torch.device | str = "auto",
    compile: bool = False,
    probe_types: dict[str, type[Probe]] | None = None,
) -> Model:
    """Load a model with probes from a checkpoint file.

    Args:
        path: Path to a checkpoint written by
            [`Model.save`](mlvbench.models.Model.save).
        device: The device to load the model and probes onto. "auto" selects CUDA if
            available and falls back to the CPU.
        compile: If True, compile the base model's `forward_features` with
            `torch.compile`. The attached probes remain eager.
        probe_types: Optional mapping of custom probe type names to `Probe` subclasses.
            Required if the checkpoint contains probes of a custom type.

    Returns:
        The reconstructed model with its probes attached.

    Raises:
        ValueError: If the checkpoint's format version is not supported by this
            mlvbench, or if a stored probe name is neither a built-in type nor present
            in `probe_types`.
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    format_version = checkpoint.get("mlvbench_checkpoint_format_version", 0)
    if format_version != CHECKPOINT_FORMAT_VERSION:
        created_with = checkpoint.get("mlvbench_version", "unknown")
        raise ValueError(
            f"Cannot load checkpoint: its format version ({format_version}) does not "
            f"match the version supported by this mlvbench "
            f"({CHECKPOINT_FORMAT_VERSION}). The checkpoint was created with mlvbench "
            f"{created_with}. Please regenerate it with the current mlvbench."
        )

    model = build_model(**checkpoint["config"], device=device, compile=compile)
    if checkpoint["state_dict"] is not None:
        model.load_state_dict(checkpoint["state_dict"])

    prior = checkpoint["prior"]
    if prior is not None:
        prior = prior.to(device)

    probes = [
        build_probe(**probe_state, probe_types=probe_types).to(device)
        for probe_state in checkpoint["probes"]
    ]

    model.probes = probes
    model.prior = prior
    return model


def list_bundles() -> list[str]:
    """Return all available model bundles."""
    return list(BUNDLES.keys())


def list_models(
    patterns: list[str] | None = None,
    bundle: str | None = None,
) -> list[str]:
    """List available models.

    Args:
        patterns: Patterns to filter model names, supporting glob-like wildcards. If
            provided, only models matching any of the patterns will be returned.
        bundle: Name of the model bundle to list models from. If provided, only models
            in the bundle will be returned.

    Returns:
        List of model names in "provider/model_name" format.
    """
    if bundle is not None:
        models = BUNDLES[bundle]
    else:
        models = (
            list_diffusers_models()
            + list_franca_models()
            + list_timm_models()
            + list_webssl_models()
        )

    if patterns is not None:
        models = [
            model
            for model in models
            if any(fnmatch.fnmatch(model, pattern) for pattern in patterns)
        ]

    return models
