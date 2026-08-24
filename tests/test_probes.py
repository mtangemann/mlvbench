"""Tests for probes."""

import pytest
import torch

from mlvbench.models import Features, TokenLayout
from mlvbench.probes import LinearProbe, build_probe


def test_linear_probe_parameter_count_with_bias():
    """Parameter count equals input_dim (weights) + 1 (bias)."""
    input_dim = 64
    probe = LinearProbe("block.0", input_dim, bias="scalar")
    n_params = sum(p.numel() for p in probe.parameters())
    assert n_params == input_dim + 1


def test_linear_probe_parameter_count_without_bias():
    """Parameter count equals input_dim (weights only) when bias="none"."""
    input_dim = 64
    probe = LinearProbe("block.0", input_dim, bias="none")
    n_params = sum(p.numel() for p in probe.parameters())
    assert n_params == input_dim


def test_linear_probe_parameter_count_spatial_bias():
    """Parameter count equals input_dim (weights) + num_patches (bias)."""
    input_dim = 64
    num_patches = 9
    probe = LinearProbe("block.0", input_dim, bias="spatial", num_patches=num_patches)
    n_params = sum(p.numel() for p in probe.parameters())
    assert n_params == input_dim + num_patches


def test_linear_probe_spatial_bias_initialized_from_prior():
    """A prior tensor selects a spatial bias initialized from it."""
    input_dim = 8
    prior = torch.randn(4, 1)
    probe = LinearProbe("block.0", input_dim, bias=prior)
    assert probe.bias_type == "spatial"
    assert torch.equal(probe.bias.detach(), prior)


def test_linear_probe_spatial_bias_requires_num_patches():
    """A string "spatial" bias without num_patches raises."""
    with pytest.raises(ValueError, match="num_patches"):
        LinearProbe("block.0", 8, bias="spatial")


def test_build_probe_unknown_type():
    """Building an unregistered probe type raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown probe type"):
        build_probe("does_not_exist", "block.0", input_dim=8)


def test_build_probe_custom_type():
    """A custom probe type is resolved via the probe_types mapping."""

    class _CustomProbe(LinearProbe):
        name = "custom"

    probe = build_probe(
        "custom", "block.0", input_dim=8, probe_types={"custom": _CustomProbe}
    )
    assert isinstance(probe, _CustomProbe)
    assert probe.name == "custom"


def _make_features(batch_size: int, input_dim: int) -> Features:
    """Build a Features container with a single layer on a 3x3 grid (9 patch tokens)."""
    tokens = {"block.0": torch.randn(batch_size, 9, input_dim)}
    return Features(tokens, TokenLayout(grid_size=(3, 3)))


@pytest.mark.parametrize(
    "probe",
    [
        LinearProbe("block.0", 8, bias="none"),
        LinearProbe("block.0", 8, bias="scalar"),
        LinearProbe("block.0", 8, bias="spatial", num_patches=9),
        LinearProbe("block.0", 8, bias=torch.randn(9, 1)),
    ],
)
def test_serialize_round_trip(probe):
    """A serialized probe is reconstructed with identical structure and outputs."""
    probe.eval()
    features = _make_features(batch_size=2, input_dim=8)
    with torch.no_grad():
        expected = probe(features)

    restored = build_probe(**probe.serialize())

    assert type(restored) is type(probe)
    assert restored.bias_type == probe.bias_type
    with torch.no_grad():
        actual = restored(features)
    assert torch.equal(actual, expected)
