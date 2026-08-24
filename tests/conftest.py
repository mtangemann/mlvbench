"""Shared pytest fixtures and hooks for device handling."""

import pytest
import torch

# Avaible devices for running tests
_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(params=_DEVICES, ids=_DEVICES)
def device(request) -> torch.device:
    """Parametrize a test over available devices.

    ```python
    # Will be run on all available devices.
    def test_model(device: str):
        model = Model().to(device)
        ...
    """
    return torch.device(request.param)


def pytest_collection_modifyitems(config, items):
    """Skip tests marked `requires_cuda` when no CUDA device is available."""
    if torch.cuda.is_available():
        return
    skip_no_cuda = pytest.mark.skip(reason="No CUDA device available.")
    for item in items:
        if "requires_cuda" in item.keywords:
            item.add_marker(skip_no_cuda)
