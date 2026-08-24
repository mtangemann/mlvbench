"""Shape generation utilities."""

import cv2
import numpy as np
import numpy.typing as npt
from scipy.interpolate import splev, splprep


def split_figure(
    segmentation: np.ndarray,
    rng: np.random.Generator,
    *,
    min_region_size: float = 0.0,
) -> np.ndarray:
    """Split the foreground region of a binary segmentation into two sub-regions.

    A random line divides the foreground pixels (label 1 in the input) into two
    sub-regions. Background pixels (label 0) are left unchanged.

    Args:
        segmentation: Binary segmentation map of shape `(H, W, 1)` and dtype `uint8`
            with values 0 (background) and 1 (foreground). The foreground region will
            be split into two sub-regions using a random line.
        rng: NumPy random number generator used for reproducibility.
        min_region_size: Minimum size of each foreground sub-region as a fraction
            of the total foreground area. Must be in [0.0, 0.5]. When 0.0
            (the default) any non-empty split is accepted.

    Returns:
        Region map as uint8 numpy array of shape `(H, W, 1)` with values 0 (background),
            1 (foreground sub-region A), and 2 (foreground sub-region B).

    Raises:
        ValueError: If `min_region_size` is outside [0.0, 0.5].
    """
    if not (0.0 <= min_region_size <= 0.5):
        raise ValueError(
            f"min_region_size must be in [0.0, 0.5], got {min_region_size}"
        )

    H, W = segmentation.shape[:2]
    foreground = segmentation[..., 0] > 0
    foreground_size = int(foreground.sum())

    xs, ys = np.meshgrid(np.arange(W), np.arange(H))

    while True:
        pivot_x = rng.uniform(0, W)
        pivot_y = rng.uniform(0, H)
        angle = rng.uniform(0, 2 * np.pi)
        nx, ny = np.cos(angle), np.sin(angle)
        signed_distance = (xs - pivot_x) * nx + (ys - pivot_y) * ny

        region1 = foreground & (signed_distance > 0)
        region2 = foreground & (signed_distance <= 0)

        region1_size = int(region1.sum())
        region2_size = int(region2.sum())

        if region1_size == 0 or region2_size == 0:
            continue
        if min(region1_size, region2_size) / foreground_size >= min_region_size:
            break

    result = np.zeros((H, W, 1), dtype=np.uint8)
    result[region1, 0] = 1
    result[region2, 0] = 2
    return result


def random_idsprite(image_size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a segmentation map for a random idsprite.

    The shape generation is adapted from the
    [idsprites](https://github.com/sbdzdz/idsprites/) library (Dziadzio et al., 2023).

    Args:
        image_size: Side length of the square image in pixels.
        rng: NumPy random number generator for reproducibility.

    Returns:
        Region map as uint8 NumPy array of shape `(image_size, image_size, 1)`
            with values 0 (background) and 1 (foreground shape).
    """
    while True:
        vertices = _sample_vertex_positions(rng)
        shape = (
            _interpolate(vertices)
            if rng.random() < 0.5
            else _interpolate(vertices, k=1)
        )
        try:
            shape = _position_shape(shape, image_size, area=0.5, rng=rng)
            break
        except ValueError:
            continue

    regions = _render_shape(shape, image_size)
    return regions[..., np.newaxis]


# Adapted from https://github.com/sbdzdz/idsprites/blob/96c0210a0124dad9283073206f3fb13be29b252b/idsprites/infinite_dsprites.py#L202-L230
def _sample_vertex_positions(
    rng: np.random.Generator,
    min_verts: int = 5,
    max_verts: int = 8,
    radius_std: float = 0.4,
    angle_std: float = 0.5,
):
    """Sample the positions of the vertices of a polygon.

    Args:
        rng: NumPy random number generator for reproducibility.
        min_verts: Minimum number of vertices (inclusive).
        max_verts: Maximum number of vertices (inclusive).
        radius_std: Standard deviation of the polar radius when sampling the vertices.
        angle_std: Standard deviation of the polar angle when sampling the vertices.

    Returns:
        An array of shape `(2, num_verts)` and dtype `float64`.
    """
    num_verts = rng.integers(min_verts, max_verts + 1)
    rs = rng.normal(1.0, radius_std, num_verts)
    rs = np.clip(rs, 0.1, 1.9)

    epsilon = 1e-6
    circle_sector = np.pi / num_verts - epsilon
    intervals = np.linspace(0, 2 * np.pi, num_verts, endpoint=False)
    thetas = rng.normal(0.0, circle_sector * angle_std, num_verts)
    thetas = np.clip(thetas, -circle_sector, circle_sector) + intervals

    vertices = [
        [r * np.cos(theta), r * np.sin(theta)]
        for r, theta in zip(rs, thetas, strict=True)
    ]
    vertices = np.array(vertices).T
    return vertices


# Adapted from https://github.com/sbdzdz/idsprites/blob/96c0210a0124dad9283073206f3fb13be29b252b/idsprites/infinite_dsprites.py#L232-L247
def _interpolate(
    verts: npt.NDArray,
    k: int = 3,
    num_spline_points: int = 1000,
):
    """Interpolate a set of vertices with a spline.

    Args:
        verts: An array of shape `(2, num_verts)` and dtype `float64`.
        k: The degree of the spline.
        num_spline_points: The number of points to sample from the spline.

    Returns:
        An array of shape `(2, num_spline_points)` and dtype `float64`.
    """
    verts = np.column_stack((verts, verts[:, 0]))
    spline_params, u, *_ = splprep(verts, s=0, per=1, k=k)
    u_new = np.linspace(u.min(), u.max(), num_spline_points)
    x, y = splev(u_new, spline_params, der=0)
    return np.array([x, y])


def _position_shape(
    shape: npt.NDArray,
    image_size: int,
    area: float,
    rng: np.random.Generator,
) -> npt.NDArray:
    """Position a shape in the image."""
    angle = rng.uniform(0, 2 * np.pi)
    shape = _rotate_shape(shape, angle)

    xmin, xmax = shape[0, :].min(), shape[0, :].max()
    ymin, ymax = shape[1, :].min(), shape[1, :].max()
    width = xmax - xmin
    height = ymax - ymin
    scale = min((image_size - 2) / width, (image_size - 2) / height)
    shape = shape - np.array([[xmin], [ymin]])
    shape = shape * scale
    shape = shape + np.array([[1], [1]])

    mask = _render_shape(shape, image_size)
    native_area = mask.sum() / image_size**2
    scale = np.sqrt(area / native_area)

    if scale > 1.0:
        raise ValueError(f"Scale {scale} is greater than 1.0")

    shape = shape * scale

    xmin, xmax = shape[0, :].min(), shape[0, :].max()
    ymin, ymax = shape[1, :].min(), shape[1, :].max()
    width = xmax - xmin
    height = ymax - ymin

    offset_x = rng.uniform(0, image_size - 1 - width)
    offset_y = rng.uniform(0, image_size - 1 - height)
    shape = shape + np.array([[offset_x], [offset_y]])

    return shape


# Adapted from https://github.com/sbdzdz/idsprites/blob/96c0210a0124dad9283073206f3fb13be29b252b/idsprites/infinite_dsprites.py#L309-L323
def _rotate_shape(shape: npt.NDArray, angle: float) -> npt.NDArray:
    """Rotate a shape by a given angle.

    Args:
        shape: An array of shape `(2, num_points)` and dtype `float64`.
        angle: The angle in radians.

    Returns:
        The rotated shape, an array of shape `(2, num_points)` and dtype `float64`.
    """
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    return rotation_matrix @ shape


def _render_shape(shape: npt.NDArray, image_size: int) -> npt.NDArray:
    """Render a shape as a binary mask.

    Args:
        shape: The shape to render, an array of shape `(2, num_points)` and dtype
            `float64`.
        image_size: The size of the image.

    Returns:
        The binary mask as a uint8 numpy array of shape `(image_size, image_size)`.
    """
    shape = shape.T.astype(np.int32)
    canvas = np.zeros((image_size, image_size, 3)).astype(np.int32)
    cv2.fillPoly(img=canvas, pts=[shape], color=(255, 255, 255), lineType=cv2.LINE_AA)
    canvas = (canvas[:, :, 0] > 0).astype(np.uint8)
    return canvas
