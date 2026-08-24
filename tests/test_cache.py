"""Tests for mlvbench.cache."""

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from mlvbench.cache import (
    PrecomputedDataset,
    create_cached_resource,
    finalize_cached_resource,
    get_cached_resource,
    precompute_dataset,
)


class _FakeDataset(Dataset):
    """Minimal dataset that returns fixed numpy arrays per sample."""

    def __init__(self, num_samples: int, keys: list[str]):
        self._num_samples = num_samples
        self._keys = keys
        self._size = (16, 19)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if index >= self._num_samples:
            raise IndexError(index)
        rng = np.random.default_rng(index)
        return {
            key: rng.integers(0, 256, (*self._size, 3), dtype=np.uint8)
            for key in self._keys
        }


@pytest.fixture(autouse=True)
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect all cache operations to a temporary directory."""
    monkeypatch.setenv("MLVBENCH_CACHE_PATH", str(tmp_path))
    return tmp_path


class TestGetCachedResource:
    """Tests for get_cached_resource."""

    def test_returns_none_for_missing_entry(self):
        result = get_cached_resource("name", "1.0.0", {"x": 1})
        assert result is None

    def test_returns_none_for_incomplete_entry(self):
        # Temporary directory created by create does not count as complete.
        create_cached_resource("name", "1.0.0", {"x": 1})
        result = get_cached_resource("name", "1.0.0", {"x": 1})
        assert result is None

    def test_returns_data_path_after_finalize(self):
        path = create_cached_resource("name", "1.0.0", {"x": 1})
        finalize_cached_resource(path)
        result = get_cached_resource("name", "1.0.0", {"x": 1})
        assert result is not None
        assert result.is_dir()


class TestCreateCachedResource:
    """Tests for create_cached_resource."""

    def test_returns_data_path(self):
        path = create_cached_resource("name", "1.0.0", {"x": 1})
        assert path.name == "data"
        assert path.is_dir()

    def test_writes_metadata_json(self):
        path = create_cached_resource("name", "1.0.0", {"x": 1})
        assert (path.parent / "metadata.json").exists()

    def test_metadata_contains_parameters(self):
        parameters = {"image_size": 224, "seed": 42}
        path = create_cached_resource("name", "1.0.0", parameters)
        with open(path.parent / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["parameters"] == parameters

    def test_warns_if_resource_exists(self):
        parameters = {"x": 1}
        path = create_cached_resource("name", "1.0.0", parameters)
        finalize_cached_resource(path)
        with pytest.warns(UserWarning, match="already exists"):
            create_cached_resource("name", "1.0.0", parameters)

    def test_does_not_delete_existing_resource(self):
        parameters = {"x": 1}
        path = create_cached_resource("name", "1.0.0", parameters)
        (path / "marker.txt").write_text("original")
        finalize_cached_resource(path)

        with pytest.warns(UserWarning):
            create_cached_resource("name", "1.0.0", parameters)

        result = get_cached_resource("name", "1.0.0", parameters)
        assert (result / "marker.txt").read_text() == "original"

    def test_returns_unique_tmp_path_each_call(self):
        # Two concurrent workers get distinct tmp directories.
        path1 = create_cached_resource("name", "1.0.0", {"x": 1})
        path2 = create_cached_resource("name", "1.0.0", {"x": 1})
        assert path1 != path2


class TestFinalizeCachedResource:
    """Tests for finalize_cached_resource."""

    def test_makes_resource_accessible(self):
        path = create_cached_resource("name", "1.0.0", {"x": 1})
        finalize_cached_resource(path)
        assert get_cached_resource("name", "1.0.0", {"x": 1}) is not None

    def test_renames_to_final_dir(self, cache_dir: Path):
        parameters = {"x": 1}
        tmp_path = create_cached_resource("name", "1.0.0", parameters)
        final_path = finalize_cached_resource(tmp_path)
        # Tmp directory is gone; final directory exists.
        assert not tmp_path.parent.exists()
        assert final_path.exists()
        assert "_tmp_" not in final_path.parent.name

    def test_warns_and_discards_if_target_exists(self):
        parameters = {"x": 1}
        path1 = create_cached_resource("name", "1.0.0", parameters)
        (path1 / "marker.txt").write_text("first")
        finalize_cached_resource(path1)

        with pytest.warns(UserWarning, match="already exists"):
            path2 = create_cached_resource("name", "1.0.0", parameters)

        (path2 / "marker.txt").write_text("second")

        with pytest.warns(UserWarning, match="already exists"):
            finalize_cached_resource(path2)

        # Tmp dir is removed; first writer's data is preserved.
        assert not path2.parent.exists()
        result = get_cached_resource("name", "1.0.0", parameters)
        assert (result / "marker.txt").read_text() == "first"

    def test_metadata_finalization_fields(self):
        path = create_cached_resource("name", "1.0.0", {"x": 1})
        (path / "a.bin").write_bytes(b"hello")
        (path / "b.bin").write_bytes(b"world!")
        t_before = time.time()
        final_path = finalize_cached_resource(path)
        t_after = time.time()

        with open(final_path.parent / "metadata.json") as f:
            metadata = json.load(f)

        assert "created_at" in metadata
        assert t_before <= metadata["finalized_at"] <= t_after
        assert metadata["duration"] >= 0
        assert metadata["num_files"] == 2
        assert metadata["total_size"] == 11  # len("hello") + len("world!")

    def test_raises_if_path_outside_cache_dir(
        self, tmp_path_factory: pytest.TempPathFactory
    ):
        outside = tmp_path_factory.mktemp("outside")
        with pytest.raises(ValueError, match="not within the cache directory"):
            finalize_cached_resource(outside)


class TestPathDeterminism:
    """Tests for stable, deterministic path construction."""

    def test_get_is_deterministic(self):
        path1 = create_cached_resource("name", "1.0.0", {"x": 1})
        path1 = finalize_cached_resource(path1)
        path2 = get_cached_resource("name", "1.0.0", {"x": 1})
        assert path1 == path2

    def test_path_differs_for_different_params(self):
        path1 = create_cached_resource("name", "1.0.0", {"image_size": 224})
        path1 = finalize_cached_resource(path1)
        path2 = create_cached_resource("name", "1.0.0", {"image_size": 518})
        path2 = finalize_cached_resource(path2)
        assert path1 != path2

    def test_path_differs_for_different_name(self):
        path1 = create_cached_resource("name1", "1.0.0", {"x": 1})
        path1 = finalize_cached_resource(path1)
        path2 = create_cached_resource("name2", "1.0.0", {"x": 1})
        path2 = finalize_cached_resource(path2)
        assert path1 != path2

    def test_path_differs_for_different_version(self):
        path1 = create_cached_resource("name", "1.0.0", {"x": 1})
        path1 = finalize_cached_resource(path1)
        path2 = create_cached_resource("name", "2.0.0", {"x": 1})
        path2 = finalize_cached_resource(path2)
        assert path1 != path2

    def test_params_order_does_not_affect_path(self):
        path1 = create_cached_resource("name", "1.0.0", {"a": 1, "b": 2})
        path1 = finalize_cached_resource(path1)
        path2 = get_cached_resource("name", "1.0.0", {"b": 2, "a": 1})
        assert path1 == path2

    def test_path_structure(self, cache_dir: Path):
        path = create_cached_resource("name", "3.0.0", {"z": 9})
        path = finalize_cached_resource(path)
        # path is <cache_dir>/name/3.0.0/<hash>/data
        assert path.name == "data"
        assert path.parent.parent.name == "3.0.0"
        assert path.parent.parent.parent.name == "name"
        assert path.parent.parent.parent.parent == cache_dir


class TestPrecomputedDataset:
    """Tests for PrecomputedDataset and precompute_dataset."""

    @pytest.fixture()
    def dataset_path(self, tmp_path: Path) -> Path:
        """Precompute a two-feature dataset and return its path."""
        dataset = _FakeDataset(num_samples=4, keys=["image_consistent", "segmentation"])
        precompute_dataset(dataset, tmp_path / "dataset", progress_bar=False)
        return tmp_path / "dataset"

    def test_len(self, dataset_path: Path):
        dataset = PrecomputedDataset(dataset_path)
        assert len(dataset) == 4

    def test_no_feature_map_returns_all_keys(self, dataset_path: Path):
        dataset = PrecomputedDataset(dataset_path)
        sample = dataset[0]
        assert set(sample.keys()) == {"__key__", "image_consistent", "segmentation"}

    def test_feature_map_renames_keys(self, dataset_path: Path):
        dataset = PrecomputedDataset(
            dataset_path, feature_map={"image_consistent": "image"}
        )
        sample = dataset[0]
        assert set(sample.keys()) == {"__key__", "image"}

    def test_feature_map_values_match_original_data(self, dataset_path: Path):
        dataset_original = PrecomputedDataset(dataset_path)
        dataset_mapped = PrecomputedDataset(
            dataset_path, feature_map={"image_consistent": "image"}
        )
        image_original = dataset_original[2]["image_consistent"]
        image_mapped = dataset_mapped[2]["image"]
        assert torch.equal(image_original, image_mapped)

    def test_multiple_keys_remapped(self, dataset_path: Path):
        dataset = PrecomputedDataset(
            dataset_path,
            feature_map={
                "image_consistent": "image",
                "segmentation": "label",
            },
        )
        sample = dataset[0]
        assert set(sample.keys()) == {"__key__", "image", "label"}
