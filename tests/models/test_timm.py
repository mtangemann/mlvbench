"""Tests for timm models.

The shared adapter contract is covered in `test_all.py`. This module adds specific tests
for models from the timm library.
"""

import pytest
import torch
from einops import rearrange

from mlvbench.models import build_model

# We test a range of models covering different parameter counts, patch sizes and
# resolutions. Moreover, we include at least one checkpoint from each timm submodule.
# fmt: off
MODELS = [
    "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k",  # vision_transformer.py
    "timm/vit_base_patch32_224.orig_in21k",            # vision_transformer.py
    "timm/beit_base_patch16_224.in22k_ft_in22k",       # beit.py
    "timm/deit_tiny_patch16_224.fb_in1k",              # deit.py
    "timm/vit_pe_core_tiny_patch16_384.fb",            # eva.py
]
# fmt: on


@pytest.mark.parametrize("model_name", MODELS)
def test_forward_features_returns_features_with_correct_shape(model_name: str, device):
    """Test that each timm submodule builds and returns correctly shaped features."""
    model = build_model(model_name, device=device)

    with torch.no_grad():
        image = torch.randn(1, 3, model.input_size, model.input_size, device=device)
        features = model.forward_features(image)

    # All features should have shape (B, N, C) where N is the number of patch tokens,
    # and live on the requested device.
    assert set(features.layers) == set(model.layer_names)
    num_tokens = (model.input_size // model.patch_size) ** 2
    for layer in model.layer_names:
        feature = features.patch(layer)
        assert feature.shape == (1, num_tokens, model.embed_dim)
        assert feature.device.type == device.type


# One model per attention implementation used by the supported timm models.
# fmt: off
ATTENTION_MODELS = [
    "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k",  # layers.Attention
    "timm/vit_small_patch16_rope_224.naver_in1k",      # layers.AttentionRope
    "timm/beit_base_patch16_224.in22k_ft_in22k",       # beit.Attention, q/v bias
    "timm/eva02_tiny_patch14_224.mim_in22k",           # EvaAttention, fused, q/v bias
    "timm/vit_pe_core_tiny_patch16_384.fb",            # EvaAttention, unfused
]
# fmt: on


@pytest.mark.parametrize("model_name", ATTENTION_MODELS)
def test_queries_keys_values_match_attention_forward(model_name: str, monkeypatch):
    """Queries, keys and values match the tensors passed to the attention kernel."""
    model = build_model(model_name, device="cpu", precision="float32")
    for block in model._model.blocks:
        block.attn.fused_attn = True

    # Record the inputs of the fused attention kernel as the reference.
    captured = []
    original = torch.nn.functional.scaled_dot_product_attention

    def spy(q, k, v, *args, **kwargs):
        captured.append((q, k, v))
        return original(q, k, v, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)

    torch.manual_seed(0)
    image = torch.rand(2, 3, model.input_size, model.input_size)
    components = ("query", "key", "value")
    with torch.no_grad():
        features = model.forward_features(
            image, [f"block.{model.num_layers - 1}:{kind}" for kind in components]
        )

    # Rotary embeddings are applied to the patch queries and keys inside the attention
    # module, so only the prefix tokens are comparable for models using them.
    uses_rope = any(tag in model_name for tag in ("rope", "eva02", "pe_core"))
    num_prefix = model.token_layout.num_prefix_tokens
    for component, reference in zip(components, captured[-1], strict=True):
        reference = rearrange(reference, "B H N D -> B N (H D)").float()
        actual = features._tokens[f"block.{model.num_layers - 1}:{component}"]
        assert actual.shape == reference.shape
        if uses_rope and component != "value":
            actual, reference = actual[:, :num_prefix], reference[:, :num_prefix]
        torch.testing.assert_close(actual, reference)


def test_forward_features_mixes_block_outputs_and_attention_features():
    """Outputs and attention components can be requested together, in any order."""
    model = build_model(MODELS[0], device="cpu")
    layers = ["block.5:key", "block.2", "block.2:value", "block.5"]
    image = torch.rand(1, 3, model.input_size, model.input_size)

    with torch.no_grad():
        features = model.forward_features(image, layers)
        reference = model.forward_features(image, ["block.2", "block.5"])
        keys_only = model.forward_features(image, ["block.5:key"])

    assert features.layers == layers
    torch.testing.assert_close(features.patch("block.2"), reference.patch("block.2"))
    torch.testing.assert_close(features.patch("block.5"), reference.patch("block.5"))
    torch.testing.assert_close(
        keys_only.patch("block.5:key"), features.patch("block.5:key")
    )
    assert keys_only.layers == ["block.5:key"]

    grid = model.input_size // model.patch_size
    keys = features.patch("block.5:key", format="BCHW")
    assert keys.shape == (1, model.embed_dim, grid, grid)
    assert keys.dtype == torch.float32


def test_list_features_includes_queries_keys_and_values():
    """timm models expose "block", query, key and value features for every block."""
    model = build_model(MODELS[0], device="cpu")
    assert len(model.list_features()) == 4 * model.num_layers

    with pytest.raises(ValueError):
        model.forward_features(torch.rand(1, 3, 224, 224), ["block.11.attn"])
