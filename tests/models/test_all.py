"""Shared tests for all models.

These tests verify the common API that should be followed by all models. Adding a new
provider is a single entry in `PROVIDERS` below, after which all tests are run against a
representative model. Provider-specific behaviour is tested in dedicated test modules
(`test_timm.py`, `test_diffusers.py`, ...).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pytest
import torch

from mlvbench.models import Model, build_model, list_models
from mlvbench.models.diffusers import DiffusersModel, list_diffusers_models
from mlvbench.models.franca import FrancaModel, list_franca_models
from mlvbench.models.timm import TimmModel, list_timm_models
from mlvbench.models.webssl import WebSSLModel, list_webssl_models


@dataclass(frozen=True)
class Provider:
    """Describes a single provider for which to test the common model API.

    Attributes:
        prefix: The provider prefix, e.g. `"timm"`.
        Model: The class implementing the `Model` interface.
        list_models: The provider's `list_*_models()` function.
        example: A small representative checkpoint, fully qualified (e.g. `"timm/..."`).
        forward_precision: The precision used when running the forward pass on CUDA.
        forward_on_cpu: Whether the forward pass can run on CPU. Some adapters (Franca)
            only run on CUDA in our setup.
    """

    prefix: str
    Model: type[Model]
    list_models: Callable[[], list[str]]
    example: str
    forward_precision: Literal["auto", "float32", "bfloat16"] = "auto"
    forward_on_cpu: bool = True


PROVIDERS = [
    Provider(
        prefix="timm",
        Model=TimmModel,
        list_models=list_timm_models,
        example="timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    ),
    Provider(
        prefix="webssl",
        Model=WebSSLModel,
        list_models=list_webssl_models,
        example="webssl/dino300m_full2b_224",
    ),
    Provider(
        prefix="diffusers",
        Model=DiffusersModel,
        list_models=list_diffusers_models,
        example="diffusers/DiT-XL-2-256",
    ),
    Provider(
        prefix="franca",
        Model=FrancaModel,
        list_models=list_franca_models,
        example="franca/vitb14_In21k",
        forward_precision="bfloat16",  # xFormers doesn't support fp32 on blackwell
        forward_on_cpu=False,
    ),
]

# Parametrize every test in this module over the providers.
pytestmark = pytest.mark.parametrize(
    "provider", PROVIDERS, ids=[provider.prefix for provider in PROVIDERS]
)


def _build_model_for_forward(provider: Provider, device: torch.device) -> Model:
    """Build the sample model for a forward pass, skipping unsupported devices."""
    if device.type == "cpu" and not provider.forward_on_cpu:
        pytest.skip(f"{provider.prefix} forward pass requires CUDA.")
    precision = provider.forward_precision if device.type == "cuda" else "auto"
    return build_model(provider.example, precision=precision, device=device)


# -------------------------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------------------------


def test_list_models_is_nonempty(provider: Provider):
    """The provider's list function returns at least one model."""
    assert len(provider.list_models()) >= 1


def test_list_models_have_prefix(provider: Provider):
    """Every listed model name carries the provider prefix."""
    prefix = f"{provider.prefix}/"
    assert all(name.startswith(prefix) for name in provider.list_models())


def test_top_level_list_models_includes_provider(provider: Provider):
    """The top-level `list_models()` includes the provider's models."""
    provider_models = set(provider.list_models())
    all_models = set(list_models())
    assert provider_models.issubset(all_models)


# -------------------------------------------------------------------------------------
# Construction
# -------------------------------------------------------------------------------------


def test_unsupported_name_raises(provider: Provider):
    """Constructing with an unsupported model name raises `ValueError`."""
    with pytest.raises(ValueError, match="Unsupported model"):
        provider.Model("definitely-not-a-real-model")


def test_build_model(provider: Provider):
    """Models from the provider can be constructed by the global `build_model`."""
    model = build_model(provider.example)

    # build_model dispatches to the right adapter and the name round-trips.
    assert isinstance(model, provider.Model)
    assert model.name == provider.example


def test_build_model_without_pretrained_weights(provider: Provider):
    """Models can be built with randomly initialized weights.

    Detailed behaviour of the random initialization (reproducibility, round-trips) is
    covered in `test_random_init.py` using a single small model.
    """
    model = build_model(provider.example, pretrained=False)

    assert isinstance(model, provider.Model)
    assert model.name == provider.example
    assert model.pretrained is False
    assert model.seed == 0

    # The architecture is unchanged, so the metadata contract still holds.
    assert model.num_layers > 0
    assert len(model.layer_names) == model.num_layers
    assert model.embed_dim > 0
    assert model.patch_size > 0
    assert model.input_size > 0
    assert model.num_parameters > 0

    # The random initialization is reported by info() and preserved by config().
    assert model.info()["pretrained"] is False
    assert model.config()["pretrained"] is False
    assert model.config()["seed"] == 0


# -------------------------------------------------------------------------------------
# Metadata
# -------------------------------------------------------------------------------------


def test_metadata(provider: Provider):
    """`build_model` returns the adapter and exposes a self-consistent contract."""
    model = build_model(provider.example)

    # info() exposes every property without raising.
    info = model.info()
    assert info["name"] == provider.example

    # The abstract property contract is self-consistent.
    assert model.num_layers > 0
    assert len(model.layer_names) == model.num_layers
    assert len(set(model.layer_names)) == model.num_layers  # names are unique
    assert model.embed_dim > 0
    assert model.patch_size > 0
    assert model.input_size > 0
    assert model.num_parameters > 0

    # The static token layout is queryable without a forward pass: grid_size is unknown
    # and the group names are exposed both directly and via info().
    layout = model.token_layout
    assert layout.grid_size is None
    assert layout.names == info["token_groups"]
    assert all(name in layout for name in layout.names)
    assert "missing" not in layout

    # config() returns the build spec used by save()/load_model().
    config = model.config()
    assert config["name"] == provider.example
    assert config["precision"] == model.precision
    assert config["pretrained"] is True
    assert config["seed"] == 0


# -------------------------------------------------------------------------------------
# Feature Extraction
# -------------------------------------------------------------------------------------


def test_forward_features(provider: Provider, device: torch.device):
    """`forward_features` returns all layers with the documented shape/device/dtype."""
    model = _build_model_for_forward(provider, device)

    num_tokens = (model.input_size // model.patch_size) ** 2
    image = torch.randint(
        0,
        256,
        (1, 3, model.input_size, model.input_size),
        dtype=torch.uint8,
        device=device,
    )

    with torch.no_grad():
        features = model.forward_features(image)

    # Without a subset, all layers are returned.
    assert set(features.layers) == set(model.layer_names)

    for layer in model.layer_names:
        patch = features.patch(layer)
        # Shape (B, N, C), on the requested device, always float32.
        assert patch.shape == (1, num_tokens, model.embed_dim)
        assert patch.device.type == device.type
        assert patch.dtype == torch.float32


def test_forward_features_layer_subset(provider: Provider, device: torch.device):
    """`forward_features` honours an explicit subset of layers."""
    model = _build_model_for_forward(provider, device)

    selected = [model.layer_names[0], model.layer_names[-1]]
    image = torch.randint(
        0,
        256,
        (1, 3, model.input_size, model.input_size),
        dtype=torch.uint8,
        device=device,
    )

    with torch.no_grad():
        features = model.forward_features(image, layers=selected)

    assert set(features.layers) == set(selected)


def test_forward_features_accepts_float_input(provider: Provider, device: torch.device):
    """`forward_features` accepts float32 images in `[0, 1]` and returns float32."""
    model = _build_model_for_forward(provider, device)

    layer = model.layer_names[0]
    image_uint8 = torch.randint(
        0,
        256,
        (1, 3, model.input_size, model.input_size),
        dtype=torch.uint8,
        device=device,
    )
    image_float32 = image_uint8.to(torch.float32) / 255.0

    with torch.no_grad():
        features_uint8 = model.forward_features(image_uint8, layers=[layer])
        features_float32 = model.forward_features(image_float32, layers=[layer])

    assert torch.allclose(
        features_uint8._tokens[layer],
        features_float32._tokens[layer],
    )


def test_forward_features_layout_matches_static_layout(
    provider: Provider, device: torch.device
):
    """The layout carried by the features matches the model's static token layout."""
    model = _build_model_for_forward(provider, device)

    image = torch.randint(
        0,
        256,
        (1, 3, model.input_size, model.input_size),
        dtype=torch.uint8,
        device=device,
    )

    with torch.no_grad():
        features = model.forward_features(image, layers=[model.layer_names[0]])

    # The prefix groups are identical; the forward pass only adds the grid size.
    assert features.layout.prefix == model.token_layout.prefix
    assert features.layout.grid_size is not None


def test_forward_features_rejects_non_divisible_input(
    provider: Provider, device: torch.device
):
    """`forward_features` fails fast when the input size is not patch-divisible."""
    model = _build_model_for_forward(provider, device)

    # Make the spatial dimensions one pixel short of a full patch row/column.
    height = model.input_size + 1
    width = model.input_size + 1
    assert height % model.patch_size != 0
    image = torch.randint(
        0, 256, (1, 3, height, width), dtype=torch.uint8, device=device
    )

    with torch.no_grad(), pytest.raises(ValueError, match="not divisible by the patch"):
        model.forward_features(image)
