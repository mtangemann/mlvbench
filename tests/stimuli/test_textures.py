"""Tests for mlvbench.stimuli.textures."""

import time
from pathlib import Path

import numpy as np
import PIL.Image
import pytest
from torch.utils.data import Dataset

from mlvbench.stimuli.textures import _prepare_dtd, texturize


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


class _ColorTextures(Dataset):
    """Minimal texture dataset returning uniformaly colored PIL Images."""

    def __init__(self, colors: list[tuple[int, int, int]], texture_size: int):
        self.images = [
            PIL.Image.new("RGB", (texture_size, texture_size), c) for c in colors
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


class TestTexturize:
    @pytest.fixture
    def color_textures_32(self):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        return _ColorTextures(colors, texture_size=32)

    @pytest.fixture
    def color_textures_64(self):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        return _ColorTextures(colors, texture_size=64)

    @pytest.fixture
    def region_map(self):
        """A self-contained 32x32 two-region map (top and bottom halves)."""
        region_map = np.zeros((32, 32, 1), dtype=np.uint8)
        region_map[16:, :, 0] = 1
        return region_map

    def test_output_shape(self, color_textures_32, region_map):
        """Output shape is (H, W, 3)."""
        result = texturize(region_map, color_textures_32, _rng())
        assert result.shape == (32, 32, 3)

    def test_output_shape_for_different_texture_sizes(
        self, color_textures_64, region_map
    ):
        """Output shape is (H, W, 3)."""
        result = texturize(region_map, color_textures_64, _rng())
        assert result.shape == (32, 32, 3)

    def test_output_dtype(self, color_textures_32, region_map):
        """Output dtype is uint8."""
        result = texturize(region_map, color_textures_32, _rng())
        assert result.dtype == np.uint8

    def test_each_region_filled_with_one_color(self, color_textures_32, region_map):
        """Every pixel in a region shares the same color (solid-color textures)."""
        result = texturize(region_map, color_textures_32, _rng())
        for label in np.unique(region_map[..., 0]):
            mask = region_map[..., 0] == label
            colors_in_region = result[mask]
            assert np.all(colors_in_region == colors_in_region[0])

    def test_different_regions_get_different_colors(
        self, color_textures_32, region_map
    ):
        """Different regions receive different textures."""
        result = texturize(region_map, color_textures_32, _rng())
        region_colors = [
            tuple(result[region_map[..., 0] == label][0])
            for label in np.unique(region_map[..., 0])
        ]
        assert len(set(region_colors)) == len(region_colors)

    def test_raises_when_too_few_textures(self, region_map):
        """ValueError when the dataset has fewer items than unique regions."""
        color_textures_32 = _ColorTextures([(255, 0, 0)], 32)
        with pytest.raises(ValueError):
            texturize(region_map, color_textures_32, _rng())

    def test_same_seed_produces_same_result(self, color_textures_32, region_map):
        """Same RNG seed yields identical output."""
        result_a = texturize(region_map, color_textures_32, _rng(0))
        result_b = texturize(region_map, color_textures_32, _rng(0))
        np.testing.assert_array_equal(result_a, result_b)

    def test_different_seeds_produce_different_results(
        self, color_textures_32, region_map
    ):
        """Different seeds assign textures differently (with high probability)."""
        result_a = texturize(region_map, color_textures_32, _rng(0))
        result_b = texturize(region_map, color_textures_32, _rng(1))
        assert not np.array_equal(result_a, result_b)

    def test_consistent_textures_across_different_region_counts(self):
        """First min(N,M) textures are identical when called with the same RNG state.

        Two calls with N and M regions respectively must assign the same texture to
        region label k for all k < min(N, M).
        """
        # 3-region map: labels 0, 1, 2 in horizontal stripes
        region_map_3 = np.zeros((30, 30, 1), dtype=np.uint8)
        region_map_3[10:20, :, 0] = 1
        region_map_3[20:, :, 0] = 2

        # 2-region map: labels 0 and 1 in horizontal halves
        region_map_2 = np.zeros((30, 30, 1), dtype=np.uint8)
        region_map_2[15:, :, 0] = 1

        colors = [(i * 20, i * 10, 255 - i * 20) for i in range(10)]
        textures = _ColorTextures(colors, texture_size=30)

        result_3 = texturize(region_map_3, textures, _rng(42))
        result_2 = texturize(region_map_2, textures, _rng(42))

        for label in (0, 1):  # min(2, 3) = 2 shared labels
            color_n = tuple(result_3[region_map_3[..., 0] == label][0])
            color_m = tuple(result_2[region_map_2[..., 0] == label][0])
            assert color_n == color_m, (
                f"Region {label}: texture differs between N=3 ({color_n}) "
                f"and M=2 ({color_m}) calls with the same RNG state"
            )


class TestPrepareDtd:
    def test_file_timestamps_are_current_after_extraction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """All file timestamps should be updated during preparation.

        Background: On some clusters, files in scratch storage are automatically deleted
        after a certain time period. Extracting the DTD tar archive naively will restore
        the original file timestamps, so that the files might get cleaned up
        immediately. This test ensures that the file timestamps are set to the
        extraction time.
        """
        monkeypatch.setenv("MLVBENCH_CACHE_PATH", str(tmp_path))

        t1 = time.time()
        path = _prepare_dtd()
        t2 = time.time()

        for file in path.rglob("*"):
            if file.is_file():
                mtime = file.stat().st_mtime
                assert t1 <= mtime <= t2, (
                    f"{file.name}: mtime {mtime:.3f} not in [{t1:.3f}, {t2:.3f}]"
                )
