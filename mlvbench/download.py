"""Download and manage files from the internet.

The location of the downloads defaults to "~/.cache/mlvbench/downloads" and can be
configured via the `MLVBENCH_DOWNLOADS_PATH` environment variable.
"""

import argparse
import base64
import hashlib
import json
import logging
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import coloredlogs
from tabulate import tabulate

LOGGER = logging.getLogger(__name__)

DOWNLOADS_PATH = Path(
    os.environ.get("MLVBENCH_DOWNLOADS_PATH", "~/.cache/mlvbench/downloads")
).expanduser().resolve()


def download(url: str) -> Path:
    """Download the file from the given URL.

    If the file was already download earlier, the cached file will be returned.

    Parameters:
        url: The URL to download.

    Returns:
        The path to the downloaded file.
    """
    output_path = DOWNLOADS_PATH / _get_cache_key(url)

    LOGGER.info(f"Downloading {url} to {output_path} ...")

    if output_path.exists():
        metadata = _read_metadata(output_path)

        if "completed_at" not in metadata:
            LOGGER.info("Removing incomplete download")
            shutil.rmtree(output_path)

        else:
            LOGGER.info("Using cached download")
            metadata["last_used_at"] = datetime.now(UTC).isoformat()
            _write_metadata(output_path, metadata)
            return _get_output_file(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "url": url,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _write_metadata(output_path, metadata)

    if url.startswith("https://drive.google.com/uc?id="):
        subprocess.run(["gdown", url], cwd=output_path, check=True)
    else:
        subprocess.run(["curl", "-fLO", url], cwd=output_path, check=True)

    output_file = _get_output_file(output_path)

    completed_at = datetime.now(UTC).isoformat()
    metadata["completed_at"] = completed_at
    metadata["last_used_at"] = completed_at
    metadata["size"] = output_file.stat().st_size
    _write_metadata(output_path, metadata)

    return output_file


def _read_metadata(output_path: Path) -> dict[str, Any]:
    """Read the metadata of a download."""
    metadata_path = output_path / "metadata.yaml"

    if not metadata_path.exists():
        return dict()

    try:
        with open(metadata_path, "r") as metadata_file:
            return json.load(metadata_file)

    except json.JSONDecodeError:
        return dict()


def _write_metadata(output_path: Path, metadata: dict[str, Any]) -> None:
    """Write the metadata of a download."""
    with open(output_path / "metadata.yaml", "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def _get_cache_key(url: str) -> str:
    # We want a short key that doesn't take much space in CLI output. Using base64
    # instead of the typical hex encoding allows for more entropy at the same length.
    sha256 = hashlib.sha256(url.encode()).digest()
    return base64.urlsafe_b64encode(sha256).decode()[:8]


def _get_output_file(output_path: Path) -> Path:
    return next(
        iter(
            child
            for child in output_path.iterdir()
            if child.name not in ["metadata.yaml", "README.md"]
        )
    )


def _cli_ls():
    downloads = []

    for path in DOWNLOADS_PATH.iterdir():
        if not path.is_dir():
            continue

        key = path.name
        metadata = _read_metadata(path)

        completed_at = metadata.get("completed_at")
        done = "✓" if completed_at is not None else " "
        url = metadata.get("url", "")

        if "size" in metadata:
            size_human = _size_to_human(metadata["size"])
        else:
            size_human = "?"

        downloads.append((
            key,
            done,
            size_human,
            _timestamp_to_human(completed_at),
            _timestamp_to_human(metadata.get("last_used_at")),
            url,
        ))

    if len(downloads) > 0:
        table = tabulate(
            downloads,
            tablefmt="plain",
            colalign=("left", "left", "right", "left", "left", "left"),
        )

        print(table)  # noqa: T201


def _cli_rm(keys: list[str]):
    for key in keys:
        path = DOWNLOADS_PATH / key
        if path.exists():
            shutil.rmtree(path)
        else:
            print(f"No such download: {key}")  # noqa: T201


def _size_to_human(size: int) -> str:
    """Convert a size in bytes to a human-readable string."""
    for unit in ["B", "K", "M", "G", "T"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} T"


def _timestamp_to_human(timestamp: str | None) -> str:
    """Convert an ISO 8601 timestamp to a human-readable string."""
    if timestamp is None:
        return ""
    return datetime.fromisoformat(timestamp).astimezone().strftime("%Y-%m-%d %H:%M")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download files to the project's download directory."
    )
    parser.add_argument(
        "command_or_urls",
        nargs="*",
        help="Command or URL(s) to download or remove",
    )
    args = parser.parse_args()

    coloredlogs.install(level=logging.INFO)

    if args.command_or_urls[0] == "ls":
        _cli_ls()

    elif args.command_or_urls[0] == "rm":
        _cli_rm(args.command_or_urls[1:])

    else:
        for url in args.command_or_urls:
            download(url)
