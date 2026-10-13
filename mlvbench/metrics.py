"""Evaluation metrics."""

import torch


def intersection_over_union(prediction: torch.Tensor, target: torch.Tensor) -> float:
    r"""Compute the intersection over union (IoU) of two binary masks.

    $$
    \mathrm{IoU} = \frac{|P \cap T|}{|P \cup T|}
    $$

    If both the prediction and target are empty ($|P \cup T| = 0$), this returns 1.0
    since the prediction is correct.

    Args:
        prediction: The predicted mask, a `bool` tensor of arbitrary shape.
        target: The target mask, a `bool` tensor of the same shape as `prediction`.

    Returns:
        The IoU computed over all elements. Returns 1.0 if both masks are empty.
    """
    if prediction.dtype != torch.bool or target.dtype != torch.bool:
        raise TypeError(
            f"Expected boolean masks, got {prediction.dtype} and {target.dtype}."
        )
    if prediction.shape != target.shape:
        raise ValueError(
            "Expected masks of the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}."
        )

    intersection = (prediction & target).sum().item()
    union = (prediction | target).sum().item()

    if union == 0:
        return 1.0

    return intersection / union
