"""Tests for the MSRA-10K dataset."""

import pytest
import torch

from mlvbench.datasets.msra10k import _MSRA10K_URL, MSRA10KDataModule
from mlvbench.download import DOWNLOADS_PATH, _get_cache_key


@pytest.fixture(scope="module")
def datamodule() -> MSRA10KDataModule:
    if not (DOWNLOADS_PATH / _get_cache_key(_MSRA10K_URL)).exists():
        pytest.skip("MSRA-10K has not been downloaded.")
    datamodule = MSRA10KDataModule()
    datamodule.image_size = 224
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def test_segmentation_masks_are_binary(datamodule):
    """All segmentation masks contain only the values 0 and 1."""
    non_binary = []
    num_samples = 0
    for dataset in [
        datamodule.train_dataset(),
        datamodule.val_dataset(),
        datamodule.test_dataset(),
    ]:
        for sample in dataset:
            num_samples += 1
            values = torch.unique(sample["segmentation"]).tolist()
            if not set(values).issubset({0, 1}):
                non_binary.append(sample["__key__"])

    assert not non_binary, (
        f"{len(non_binary)} of {num_samples} masks are non-binary, "
        f"e.g. {non_binary[:10]}"
    )
