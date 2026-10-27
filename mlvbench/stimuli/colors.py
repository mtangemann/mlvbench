"""Color fill utilities."""

import numpy as np
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab

# Candidates are drawn from the RNG in fixed-size chunks so that the candidate stream
# depends only on the RNG state, never on the requested number of colors.
_CHUNK_SIZE = 64

# `lab2rgb` clips out-of-gamut results, so a CIELAB color survives the round trip
# through sRGB unchanged exactly if it is in gamut. In-gamut colors round-trip with an
# error of ~1e-13, the smallest out-of-gamut error is ~1e-4, so this threshold separates
# the two cases with a wide margin.
_GAMUT_TOLERANCE = 1e-6


def sample_colors(
    num_colors: int,
    rng: np.random.Generator,
    lightness_range: tuple[float, float] = (20.0, 90.0),
    chroma_range: tuple[float, float] = (-60.0, 60.0),
    min_delta_e: float = 25.0,
    max_attempts: int = 1000,
) -> np.ndarray:
    """Sample perceptually distinct colors uniformly from a box in CIELAB space.

    Candidates are drawn uniformly from the given `L*`, `a*` and `b*` ranges and are
    accepted if they lie inside the sRGB gamut and are at least `min_delta_e`
    (CIEDE2000) away from every color accepted before.

    Args:
        num_colors: Number of colors to sample.
        rng: NumPy random number generator for reproducibility. Two calls with the same
            RNG state but different numbers of colors N and M return the same first
            `min(N, M)` colors.
        lightness_range: Range of the `L*` coordinate to sample from.
        chroma_range: Range of the `a*` and `b*` coordinates to sample from.
        min_delta_e: Minimum CIEDE2000 distance between any two sampled colors. This
            bounds how many colors can be sampled at all: with the default box and
            distance, up to about eight colors are reachable, which is well above the
            two or three regions the stimuli use.
        max_attempts: Maximum number of candidates to draw before giving up.

    Returns:
        RGB colors as uint8 NumPy array of shape `(num_colors, 3)`.

    Raises:
        ValueError: If `num_colors` is negative, or if `max_attempts` candidates were
            drawn without finding `num_colors` acceptable colors.
    """
    if num_colors < 0:
        raise ValueError(f"num_colors must be non-negative, got {num_colors}")

    accepted_lab: list[np.ndarray] = []
    accepted_rgb: list[np.ndarray] = []
    attempts = 0

    while len(accepted_lab) < num_colors:
        if attempts >= max_attempts:
            raise ValueError(
                f"Could not sample {num_colors} colors with a minimum distance of " +
                f"{min_delta_e} within {max_attempts} attempts"
            )

        # Drawing the three coordinates with one call each keeps the candidate stream
        # independent of the chunk contents.
        candidates = np.stack(
            [
                rng.uniform(*lightness_range, _CHUNK_SIZE),
                rng.uniform(*chroma_range, _CHUNK_SIZE),
                rng.uniform(*chroma_range, _CHUNK_SIZE),
            ],
            axis=-1,
        )

        # The gamut test is vectorized over the whole chunk, the distance test has to be
        # sequential because it depends on the colors accepted so far.
        candidates_rgb = lab2rgb(candidates)
        in_gamut = (
            np.abs(candidates - rgb2lab(candidates_rgb)).max(axis=-1) < _GAMUT_TOLERANCE
        )

        for candidate, candidate_rgb, candidate_in_gamut in zip(
            candidates, candidates_rgb, in_gamut, strict=True
        ):
            attempts += 1

            if not candidate_in_gamut:
                continue

            if accepted_lab:
                distances = deltaE_ciede2000(
                    candidate[np.newaxis], np.stack(accepted_lab)
                )
                if distances.min() < min_delta_e:
                    continue

            accepted_lab.append(candidate)
            accepted_rgb.append(candidate_rgb)

            if len(accepted_lab) == num_colors:
                break

    if not accepted_rgb:
        return np.zeros((0, 3), dtype=np.uint8)

    return np.round(np.stack(accepted_rgb) * 255).astype(np.uint8)


def colorize(
    region_map: np.ndarray,
    rng: np.random.Generator,
    **kwargs,
) -> np.ndarray:
    """Fill each region in a region map with a unique random color.

    This is the flat-color counterpart of
    [`texturize`][mlvbench.stimuli.textures.texturize] and shares its contract: regions
    are filled in the order of their labels, so two calls with the same RNG state but
    different numbers of regions N and M assign the same color to the first
    `min(N, M)` regions.

    Args:
        region_map: Integer-labeled region map of shape `(H, W, 1)` and dtype `uint8`.
        rng: NumPy random number generator for reproducibility.
        **kwargs: Additional keyword arguments passed on to [`sample_colors`][].

    Returns:
        RGB image as uint8 NumPy array of shape `(H, W, 3)`.
    """
    H, W = region_map.shape[:2]
    labels = np.unique(region_map[..., 0])

    colors = sample_colors(len(labels), rng, **kwargs)
    image = np.zeros((H, W, 3), dtype=np.uint8)

    for label, color in zip(labels, colors, strict=True):
        image[region_map[..., 0] == label] = color

    return image
