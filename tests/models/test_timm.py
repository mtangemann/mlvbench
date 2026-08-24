"""Tests for timm models.

The shared adapter contract is covered in `test_all.py`. This module adds specific tests
for models from the timm library.
"""

import pytest
import torch

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
