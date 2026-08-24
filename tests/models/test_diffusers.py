"""Tests for the diffusers adapter.

The shared adapter contract is covered in `test_all.py`. This module adds specific tests
for models from the diffusers library.
"""

import torch

from mlvbench.models.diffusers import DiffusersModel


def test_config_captures_timestep():
    """config() extends the base spec with the denoising timestep."""
    model = DiffusersModel("DiT-XL-2-256", timestep=0.5)
    config = model.config()
    assert config["name"] == "diffusers/DiT-XL-2-256"
    assert config["timestep"] == 0.5


def test_forward_features_deterministic_at_t0(device):
    """Verify that timestep=0 (no noise) returns deterministic features."""
    model = DiffusersModel("DiT-XL-2-256", timestep=0).to(device)
    images = torch.randint(
        0,
        256,
        (1, 3, model.input_size, model.input_size),
        dtype=torch.uint8,
        device=device,
    )
    layer = model.layer_names[0]

    with torch.no_grad():
        feat1 = model.forward_features(images, layers=[layer])._tokens[layer]
        feat2 = model.forward_features(images, layers=[layer])._tokens[layer]

    assert torch.allclose(feat1, feat2)
