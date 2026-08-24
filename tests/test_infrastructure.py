"""Tests for infrastructure utilities."""

import pytest

from mlvbench.infrastructure import resolve_layers

LAYERS = ["block.0", "block.1", "block.2", "block.3", "block.4"]


def test_none_returns_all_layers():
    """Return all layers when no layers specification is provided."""
    assert resolve_layers(LAYERS, None) == LAYERS


def test_single_name():
    """Resolve a single layer name."""
    assert resolve_layers(LAYERS, "block.2") == ["block.2"]


def test_multiple_names():
    """Resolve multiple layer names."""
    assert resolve_layers(LAYERS, "block.0,block.4") == ["block.0", "block.4"]


def test_positive_index():
    """Resolve a positive integer index."""
    assert resolve_layers(LAYERS, "0") == ["block.0"]


def test_last_positive_index():
    """Resolve the last valid positive index."""
    assert resolve_layers(LAYERS, "4") == ["block.4"]


def test_negative_index():
    """Resolve a negative integer index."""
    assert resolve_layers(LAYERS, "-1") == ["block.4"]


def test_negative_index_not_last():
    """Resolve a negative index that is not -1."""
    assert resolve_layers(LAYERS, "-2") == ["block.3"]


def test_mixed_indices_and_names():
    """Resolve a mix of indices and names."""
    assert resolve_layers(LAYERS, "0,1,-1") == ["block.0", "block.1", "block.4"]


def test_preserves_order():
    """Preserves the order of the specified layers."""
    assert resolve_layers(LAYERS, "block.4,block.0,block.2") == [
        "block.4",
        "block.0",
        "block.2",
    ]


def test_duplicate_names_deduplicates(caplog):
    """Deduplicates duplicate layer names and emits a warning."""
    result = resolve_layers(LAYERS, "block.0,block.0")
    assert result == ["block.0"]
    assert "duplicate" in caplog.text.lower()


def test_duplicate_index_and_name_deduplicates(caplog):
    """Deduplicates when the same layer is specified by index and by name."""
    result = resolve_layers(LAYERS, "0,block.0")
    assert result == ["block.0"]
    assert "duplicate" in caplog.text.lower()


def test_duplicate_negative_and_positive_index_deduplicates(caplog):
    """Deduplicates when a layer is specified via both a positive and negative index."""
    result = resolve_layers(LAYERS, "4,-1")
    assert result == ["block.4"]
    assert "duplicate" in caplog.text.lower()


def test_out_of_range_positive_index_raises():
    """Raises ValueError for an out-of-range positive index."""
    with pytest.raises(IndexError):
        resolve_layers(LAYERS, "5")


def test_out_of_range_negative_index_raises():
    """Raises ValueError for an out-of-range negative index."""
    with pytest.raises(IndexError):
        resolve_layers(LAYERS, "-6")


def test_unknown_layer_name_raises():
    """Raises ValueError for an unknown layer name."""
    with pytest.raises(IndexError, match="block.99"):
        resolve_layers(LAYERS, "block.99")
