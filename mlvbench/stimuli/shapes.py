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


def random_idsprite(
    image_size: int,
    rng: np.random.Generator,
    *,
    area: float = 0.5,
    covers_center: bool | None = None,
    center_radius: float = 0.02,
    margin: float = 0.0,
    max_attempts: int = 200,
) -> np.ndarray:
    """Generate a segmentation map for a random idsprite.

    The shape generation is adapted from the
    [idsprites](https://github.com/sbdzdz/idsprites/) library (Dziadzio et al., 2023).

    Shapes are sampled until one can be scaled and placed to satisfy `area`,
    `covers_center` and `margin`.

    Args:
        image_size: Side length of the square image in pixels.
        rng: NumPy random number generator for reproducibility.
        area: Target area of the shape as a fraction of the image area. Measured on the
            rendered mask, so the returned map matches this closely (within ~0.1% of the
            image area) rather than only in terms of the underlying polygon.
        covers_center: Whether the shape must cover the center of the image (`True`),
            must avoid it (`False`), or is placed without a constraint (`None`). The
            center is the disc of radius `center_radius`, which has to be covered
            completely or missed completely.
        center_radius: Radius of the central disc, as a fraction of `image_size`.
        margin: Minimum distance between the shape and the image border, as a fraction
            of `image_size`. With the default of 0.0 the shape may touch the border.
        max_attempts: Maximum number of shapes to sample before giving up.

    Returns:
        Region map as uint8 NumPy array of shape `(image_size, image_size, 1)`
            with values 0 (background) and 1 (foreground shape).

    Raises:
        ValueError: If `area` is not in (0.0, 1.0), or if no shape satisfying the
            constraints was found within `max_attempts`.
    """
    if not (0.0 < area < 1.0):
        raise ValueError(f"area must be in (0.0, 1.0), got {area}")

    center = _central_disc(image_size, center_radius)

    for _ in range(max_attempts):
        vertices = _sample_vertex_positions(rng)
        shape = (
            _interpolate(vertices)
            if rng.random() < 0.5
            else _interpolate(vertices, k=1)
        )
        try:
            shape = _position_shape(
                shape,
                image_size,
                area=area,
                rng=rng,
                covers_center=covers_center,
                center=center,
                margin=margin,
            )
        except ValueError:
            continue

        regions = _render_shape(shape, image_size)
        return regions[..., np.newaxis]

    raise ValueError(
        f"Could not generate a shape with area {area}, covers_center={covers_center} "
        f"and margin {margin} within {max_attempts} attempts"
    )


def _central_disc(image_size: int, radius: float) -> npt.NDArray[np.bool_]:
    """Return a mask of the central disc with the given radius.

    Args:
        image_size: Side length of the square image in pixels.
        radius: Radius of the disc, as a fraction of `image_size`.

    Returns:
        Boolean mask of shape `(image_size, image_size)`. The mask is never empty: it
            always contains at least the pixel closest to the image center.
    """
    center = (image_size - 1) / 2
    coordinates = np.arange(image_size) - center
    rows, columns = np.meshgrid(coordinates, coordinates, indexing="ij")
    disc = np.hypot(rows, columns) <= radius * image_size

    if not disc.any():
        disc[int(round(center)), int(round(center))] = True

    return disc


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
    *,
    covers_center: bool | None = None,
    center: npt.NDArray[np.bool_] | None = None,
    margin: float = 0.0,
    placement_attempts: int = 40,
) -> npt.NDArray:
    """Scale a shape to the requested area and place it in the image.

    Args:
        shape: The shape to place, an array of shape `(2, num_points)`.
        image_size: Side length of the square image in pixels.
        area: Target area of the rendered shape, as a fraction of the image area.
        rng: NumPy random number generator for reproducibility.
        covers_center: Whether the shape must cover `center` completely (`True`), must
            miss it completely (`False`), or is placed without a constraint (`None`).
        center: Boolean mask of the central region, required unless `covers_center`
            is `None`.
        margin: Minimum distance between the shape and the image border, as a fraction
            of `image_size`.
        placement_attempts: Maximum number of offsets to try before giving up on this
            shape.

    Returns:
        The scaled and translated shape, an array of shape `(2, num_points)`.

    Raises:
        ValueError: If the shape cannot be scaled to `area` without growing beyond the
            image, if it does not leave room for `margin`, or if no offset satisfying
            `covers_center` was found.
    """
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

    shape = _scale_to_area(shape, image_size, area)

    # Move the shape into the corner, so that the offsets below control the distance to
    # the image border exactly.
    shape = shape - np.array([[shape[0, :].min()], [shape[1, :].min()]])
    width = shape[0, :].max()
    height = shape[1, :].max()

    low = margin * image_size
    high_x = image_size - 1 - low - width
    high_y = image_size - 1 - low - height

    if high_x <= low or high_y <= low:
        raise ValueError(
            f"Shape of area {area} does not leave room for a margin of {margin}"
        )

    if covers_center is None:
        offset = np.array([[rng.uniform(low, high_x)], [rng.uniform(low, high_y)]])
        return shape + offset

    if center is None:
        raise ValueError("center is required when covers_center is not None")

    for _ in range(placement_attempts):
        offset = np.array([[rng.uniform(low, high_x)], [rng.uniform(low, high_y)]])
        placed = shape + offset
        covered = _render_shape(placed, image_size)[center] > 0
        if covered.all() if covers_center else not covered.any():
            return placed

    raise ValueError(
        f"Could not place a shape of area {area} with covers_center={covers_center} "
        f"within {placement_attempts} attempts"
    )


def _scale_to_area(
    shape: npt.NDArray,
    image_size: int,
    area: float,
    iterations: int = 3,
) -> npt.NDArray:
    """Scale a shape so that its rendered area matches the target.

    Args:
        shape: The shape to scale, an array of shape `(2, num_points)` positioned in the
            top left corner of the image at its maximum size.
        image_size: Side length of the square image in pixels.
        area: Target area of the rendered shape, as a fraction of the image area.
        iterations: Number of refinement steps.

    Returns:
        The scaled shape, an array of shape `(2, num_points)`.

    Raises:
        ValueError: If reaching `area` would require growing the shape beyond the image,
            or if the shape renders empty.
    """
    scale = 1.0

    for _ in range(iterations):
        current_area = _render_shape(shape * scale, image_size).sum() / image_size**2

        if current_area <= 0.0:
            raise ValueError("Shape renders as an empty mask")

        scale *= np.sqrt(area / current_area)

        if scale > 1.0:
            raise ValueError(f"Scale {scale} is greater than 1.0")

    return shape * scale


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
