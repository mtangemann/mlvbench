"""Caching utilities for reusable files.

[`mlvbench.cache`][] provides utilities for caching files and reusing them across
experiments. The following example illustrates how the cache is typically used:

```python
from pathlib import Path

from mlvbench.cache import (
    get_cached_resource,
    create_cached_resource,
    finalize_cached_resource,
)

def my_data(seed: int = 2) -> Path:
    # A cached resource is identified by a name, version and the parameters that where
    # used to generate the resource:
    name = "my_data"
    version = "1"
    parameters = { "seed": seed }

    # If the resource with the given parameters already exists, get_cached_resource will
    # return its path.
    path = get_cached_resource(name, version, parameters)
    if path is not None:
        return path

    # If the resource doesn't exist yet, create_cached_resource will create a temporary
    # directory where we can prepare the data.
    path = create_cached_resource(name, version, parameters)

    # Prepare the data at the given path
    ...

    # Once all files have been generated, finalize_cached_resource will move the data to
    # its final location. This method takes care of race conditions: If a different
    # process has generated the same resource in the meantime, the newly created data
    # will be discarded and the cached data is used.
    path = finalize_cached_resource(path)

    return path
```

The location of the cache defaults to "~/.cache/mlvbench/cache" and can be configured
via the `MLVBENCH_CACHE_PATH` environment variable.
"""

import hashlib
import json
import os
import shutil
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset
from tqdm import tqdm


def _get_cache_path() -> Path:
    cache_path = Path(os.environ.get("MLVBENCH_CACHE_PATH", "~/.cache/mlvbench/cache"))
    return cache_path.expanduser().resolve()


def get_cached_resource(
    name: str,
    version: str,
    parameters: dict[str, Any],
) -> Path | None:
    """Return the data path of a complete cache entry, or None if not ready.

    Args:
        name: Name of the cached resource.
        version: Version string of the resource.
        parameters: Parameter dict that identifies this particular resource.

    Returns:
        Path to the `data/` subdirectory if the entry is complete, else None.
    """
    path = _get_cached_resource_path(name, version, parameters) / "data"
    return path if path.exists() else None


def create_cached_resource(
    name: str,
    version: str,
    parameters: dict[str, Any],
) -> Path:
    """Create a new cache entry and return a temporary data path.

    This will create a new cached resource at a temporary location. Call
    [`finalize_cached_resource`][mlvbench.cache.finalize_cached_resource] to move it to
    its final location and make the data available for future runs.

    Args:
        name: Name of the cached resource.
        version: Version string of the resource.
        parameters: Parameter dict that identifies this particular resource.

    Returns:
        Path to the directory where resource data should be written.
    """
    final_root = _get_cached_resource_path(name, version, parameters)

    if final_root.exists():
        warnings.warn(
            f"Cache entry '{name}' ({version}) already exists at {final_root}. " +
            "Call get_cached_resource() first to reuse it.",
            stacklevel=2,
        )

    tmp_id = uuid.uuid4().hex[:8]
    tmp_root = final_root.parent / f"{final_root.name}_tmp_{tmp_id}"
    data_path = tmp_root / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    with open(tmp_root / "metadata.json", "w") as f:
        json.dump({"parameters": parameters, "created_at": time.time()}, f, indent=2)

    return data_path


def finalize_cached_resource(path: Path) -> Path:
    """Atomically finalize a cached resource by moving it to its final path.

    If the target already exists (e.g., another worker finalized first), this emits a
    warning and deletes the newly created data.

    Args:
        path: The data path returned by
            [`create_cached_resource`][mlvbench.cache.create_cached_resource].

    Returns:
        The updated data path.

    Raises:
        ValueError: If `path` is not located within the cache directory.
    """
    cache_path = _get_cache_path()

    if not path.is_relative_to(cache_path):
        raise ValueError(f"Path {path} is not within the cache directory {cache_path}.")

    tmp_root = path.parent
    hash_str = tmp_root.name.split("_tmp_")[0]
    final_root = tmp_root.parent / hash_str

    metadata_path = tmp_root / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    finalized_at = time.time()
    data_files = list(path.rglob("*"))
    metadata["finalized_at"] = finalized_at
    metadata["duration"] = finalized_at - metadata["created_at"]
    metadata["num_files"] = sum(1 for p in data_files if p.is_file())
    metadata["total_size"] = sum(p.stat().st_size for p in data_files if p.is_file())

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    try:
        tmp_root.rename(final_root)
    except OSError:
        warnings.warn(
            f"Cache entry at {final_root} already exists. " +
            "Discarding the newly created resource.",
            stacklevel=2,
        )
        shutil.rmtree(tmp_root)

    return final_root / "data"


def _get_cached_resource_path(
    name: str,
    version: str,
    parameters: dict[str, Any],
) -> Path:
    """Return the path to the cached resource."""
    payload = {"name": name, "version": version, "parameters": parameters}
    encoded = json.dumps(payload, sort_keys=True).encode()
    hash_str = hashlib.sha256(encoded).hexdigest()[:8]
    return _get_cache_path() / name / version / hash_str


def precompute_dataset(
    dataset: Dataset,
    output_path: Path,
    progress_bar: bool = True,
) -> "PrecomputedDataset":
    """Precompute the dataset and store it as np.memmap.

    Args:
        dataset: The dataset to precompute.
        output_path: The path to store the precomputed dataset.
        progress_bar: Whether to show a progress bar.

    Returns:
        A PrecomputedDataset object.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    num_samples = len(dataset)

    buffers = dict()
    metadata = dict()
    keys = []

    for key, value in dataset[0].items():
        if key == "__key__":
            continue

        buffers[key] = np.memmap(
            output_path / f"{key}.memmap",
            dtype=value.dtype,
            mode="w+",
            shape=(num_samples, *value.shape),
        )

        metadata[key] = {
            "shape": buffers[key].shape,
            "dtype": buffers[key].dtype.name,
        }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if progress_bar:
        dataset = tqdm(dataset, desc="Precomputing dataset", total=num_samples)

    for sample_index, sample in enumerate(dataset):
        for key, value in sample.items():
            if key == "__key__":
                keys.append(value)
            else:
                buffers[key][sample_index] = value

    if len(keys) > 0:
        with open(output_path / "keys", "w") as f:
            f.write("\n".join(keys))

    return PrecomputedDataset(output_path)


class PrecomputedDataset(Dataset):
    """Precomputed dataset."""

    def __init__(
        self,
        path: Path,
        feature_map: dict[str, str] | None = None,
        mmap: bool = True,
    ):
        """Initialize the dataset.

        Args:
            path: Path to the precomputed dataset directory.
            feature_map: Optional mapping from stored feature names to output
                feature names, e.g. `{"image_consistent": "image"}`.  When
                provided, only the keys present in the map are loaded; all
                other stored features are ignored.
            mmap: If True (default), buffers are memory-mapped from disk. If
                False, the entire arrays are loaded into RAM at construction
                time.
        """
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.mmap = mmap
        self.buffers = dict()
        for key, info in metadata.items():
            if feature_map is not None:
                if key not in feature_map:
                    continue
                out_key = feature_map[key]
            else:
                out_key = key
            buffer = np.memmap(
                path / f"{key}.memmap",
                dtype=info["dtype"],
                mode="r",
                shape=info["shape"],
            )
            if not mmap:
                buffer = torch.from_numpy(np.array(buffer))
                buffer = rearrange(buffer, "B H W C -> B C H W")
            self.buffers[out_key] = buffer

        if (path / "keys").exists():
            with open(path / "keys", "r") as f:
                self.keys = f.read().splitlines()
        else:
            self.keys = None

    def __len__(self):
        """Return the number of samples in the dataset."""
        return next(iter(self.buffers.values())).shape[0]

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        """Return the sample at the given index.

        Returns:
            A dictionary mapping each feature name to its value for this sample. Each
                value has the shape and dtype of the corresponding cached feature with
                the leading sample dimension removed.
        """
        sample = dict()

        for key, buffer in self.buffers.items():
            if isinstance(buffer, torch.Tensor):
                sample[key] = buffer[index]
            else:
                sample[key] = torch.from_numpy(buffer[index]).permute(2, 0, 1)

        if self.keys is not None:
            sample["__key__"] = self.keys[index]
        else:
            sample["__key__"] = str(index)

        return sample


def _human_size(num_bytes: int) -> str:
    """Return a human-readable string for a byte count."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def _human_duration(seconds: float) -> str:
    """Return a human-readable string for a duration in seconds."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


if __name__ == "__main__":
    import argparse
    import sys
    from datetime import datetime

    import tabulate

    parser = argparse.ArgumentParser(description="Manage cached resources.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List cached resources.")
    list_parser.add_argument(
        "--sort",
        choices=["name", "size", "created", "duration", "files"],
        default="name",
        help="Sort order (default: name)",
    )

    remove_parser = subparsers.add_parser("remove", help="Remove a cached resource.")
    remove_parser.add_argument("hash", help="Hash of the cache entry to remove.")

    subparsers.add_parser("prune", help="Remove all incomplete resources.")

    args = parser.parse_args()

    cache_path = _get_cache_path()

    if args.command == "prune":
        incomplete_paths = (
            [
                path
                for path in cache_path.rglob("*")
                if path.is_dir() and "_tmp_" in path.name
            ]
            if cache_path.exists()
            else []
        )
        for path in incomplete_paths:
            shutil.rmtree(path)
            print(f"Removed {path}")  # noqa: T201
        print(f"\nRemoved {len(incomplete_paths)} incomplete resource(s).")  # noqa: T201

    elif args.command == "remove":
        if cache_path.exists():
            matches = list(cache_path.glob(f"*/*/{args.hash}"))
        else:
            matches = []
        if len(matches) == 0:
            print(f"No cache entry found with hash '{args.hash}'.", file=sys.stderr)  # noqa: T201
            sys.exit(1)
        for path in matches:
            shutil.rmtree(path)
            print(f"Removed {path}")  # noqa: T201

    elif args.command == "list":
        rows = []

        if cache_path.exists():
            for name_dir in cache_path.iterdir():
                if not name_dir.is_dir():
                    continue
                for version_dir in name_dir.iterdir():
                    if not version_dir.is_dir():
                        continue
                    for hash_dir in version_dir.iterdir():
                        if not hash_dir.is_dir() or "_tmp_" in hash_dir.name:
                            continue
                        if not (hash_dir / "data").exists():
                            continue
                        metadata_file = hash_dir / "metadata.json"
                        if not metadata_file.exists():
                            continue
                        with open(metadata_file) as f:
                            meta = json.load(f)
                        rows.append(
                            {
                                "hash": hash_dir.name,
                                "name": name_dir.name,
                                "version": version_dir.name,
                                "parameters": json.dumps(
                                    meta["parameters"], sort_keys=True
                                ),
                                "created": meta.get("created_at"),
                                "duration": meta.get("duration"),
                                "num_files": meta.get("num_files"),
                                "size": meta.get("total_size"),
                            }
                        )

        sort_keys = {
            "name": lambda row: (row["name"], row["version"], row["parameters"]),
            "size": lambda row: row["size"],
            "created": lambda row: row["created"],
            "duration": lambda row: row["duration"],
            "files": lambda row: row["num_files"],
        }
        rows.sort(key=sort_keys[args.sort])

        table = [
            [
                row["hash"],
                row["name"],
                row["version"],
                row["parameters"],
                datetime.fromtimestamp(row["created"]).strftime("%Y-%m-%d %H:%M"),
                _human_duration(row["duration"]),
                row["num_files"],
                _human_size(row["size"]),
            ]
            for row in rows
        ]
        headers = [
            "Hash",
            "Name",
            "Version",
            "Parameters",
            "Created",
            "Duration",
            "Files",
            "Size",
        ]
        total_files = sum(row["num_files"] for row in rows)
        total_size = sum(row["size"] for row in rows)
        footer = ["", "", "", "", "", "", total_files, _human_size(total_size)]
        print(  # noqa: T201
            tabulate.tabulate(
                table + [tabulate.SEPARATING_LINE, footer],
                headers=headers,
                tablefmt="simple",
            )
        )

        incomplete_paths = (
            [
                path
                for path in cache_path.rglob("*")
                if path.is_dir() and "_tmp_" in path.name
            ]
            if cache_path.exists()
            else []
        )
        if len(incomplete_paths) > 0:
            incomplete_files = sum(
                1
                for incomplete_path in incomplete_paths
                for path in incomplete_path.rglob("*")
                if path.is_file()
            )
            incomplete_size = sum(
                path.stat().st_size
                for incomplete_path in incomplete_paths
                for path in incomplete_path.rglob("*")
                if path.is_file()
            )
            print(  # noqa: T201
                f"\n{len(incomplete_paths)} incomplete resource(s): " +
                f"{incomplete_files} files, {_human_size(incomplete_size)}"
            )
        else:
            print("\nNo incomplete resources found.")  # noqa: T201
