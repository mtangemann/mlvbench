"""Tests for randomly initialized models.

That every provider supports `pretrained=False` is covered in `test_all.py`. This module
tests the behaviour of the random initialization in detail, using a single small model
to keep the tests fast.
"""

import torch

from mlvbench.models import Model, build_model, load_model

MODEL = "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k"


def assert_same_weights(model: Model, other: Model) -> None:
    """Assert that two models have exactly the same weights."""
    state_dict = model.state_dict()
    other_state_dict = other.state_dict()
    assert state_dict.keys() == other_state_dict.keys()
    for key, value in state_dict.items():
        assert torch.equal(value, other_state_dict[key]), f"{key} differs"


def different_weights(model: Model, other: Model) -> bool:
    """Return whether any weight of the two models differs."""
    other_state_dict = other.state_dict()
    return any(
        not torch.equal(value, other_state_dict[key])
        for key, value in model.state_dict().items()
    )


def test_same_seed_gives_same_weights():
    """Building twice with the same seed yields identical weights."""
    model = build_model(MODEL, pretrained=False, seed=42)
    other = build_model(MODEL, pretrained=False, seed=42)

    assert_same_weights(model, other)


def test_different_seed_gives_different_weights():
    """Building with different seeds yields different weights."""
    model = build_model(MODEL, pretrained=False, seed=42)
    other = build_model(MODEL, pretrained=False, seed=43)

    assert different_weights(model, other)


def test_random_weights_differ_from_pretrained_weights():
    """A randomly initialized model does not carry the pretrained weights."""
    model = build_model(MODEL, pretrained=False, seed=42)
    pretrained = build_model(MODEL, pretrained=True)

    assert different_weights(model, pretrained)


def test_seed_does_not_affect_global_random_state():
    """Seeding the model initialization leaves the global random state untouched."""
    torch.manual_seed(0)
    expected = torch.rand(4)

    torch.manual_seed(0)
    build_model(MODEL, pretrained=False, seed=42)
    actual = torch.rand(4)

    assert torch.equal(actual, expected)


def test_seed_is_ignored_for_pretrained_models():
    """A seed is accepted but does not change the weights of a pretrained model."""
    model = build_model(MODEL, pretrained=True, seed=42)

    assert model.pretrained is True
    assert model.seed == 42
    assert_same_weights(model, build_model(MODEL, pretrained=True))


def test_config_round_trip():
    """Rebuilding a model from its config reproduces the same random weights."""
    model = build_model(MODEL, pretrained=False, seed=42)

    config = model.config()
    assert config["pretrained"] is False
    assert config["seed"] == 42

    assert_same_weights(model, build_model(**config))


def test_save_load_round_trip(tmp_path):
    """A saved randomly initialized model is restored with the same weights."""
    model = build_model(MODEL, pretrained=False, seed=42, device="cpu")
    model.save(tmp_path / "model.pt")

    loaded = load_model(tmp_path / "model.pt", device="cpu")

    assert loaded.pretrained is False
    assert loaded.seed == 42
    assert_same_weights(model, loaded)


def test_forward_features(device: torch.device):
    """A randomly initialized model extracts features like a pretrained one."""
    model = build_model(MODEL, pretrained=False, seed=42, device=device)

    image = torch.randint(
        0,
        256,
        (1, 3, model.input_size, model.input_size),
        dtype=torch.uint8,
        device=device,
    )

    with torch.no_grad():
        features = model.forward_features(image)

    assert set(features.layers) == set(model.layer_names)
    num_tokens = (model.input_size // model.patch_size) ** 2
    for layer in model.layer_names:
        patch = features.patch(layer)
        assert patch.shape == (1, num_tokens, model.embed_dim)
        assert patch.device.type == device.type
        assert patch.dtype == torch.float32
