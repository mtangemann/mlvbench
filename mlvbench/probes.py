"""Probes.

Probes are small models that readout information from intermediate features of a
pretrained vision model.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from mlvbench.models import Features


class Probe(nn.Module, ABC):
    """Abstract base class for probes.

    Attributes:
        name: Name of the probe type.
        layer: Name of the backbone layer the probe reads from.
        metadata: Free-form dict of training/experiment information attached to the
            probe. Persisted in full when the probe is saved.
    """

    def __init__(
        self,
        layer: str,
        input_dim: int,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the probe.

        Args:
            layer: Name of the model layer the probe reads from.
            input_dim: Number of dimensions of the input features.
            metadata: Free-form dict of training/experiment information.
        """
        super().__init__()
        self.layer: str = layer
        self.metadata: dict[str, Any] = metadata if metadata is not None else {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the probe type."""

    def select_features(self, features: "Features") -> torch.Tensor:
        """Select the part of the feature representation used by the probe.

        Defaults to the patch tokens of `self.layer` with shape `(B, N, C)`. Override to
        read from a different part of the representation (e.g. the cls token or a
        different token format).

        Args:
            features: The features extracted from the backbone.

        Returns:
            The selected feature tensor.
        """
        return features.patch(self.layer)

    @abstractmethod
    def forward(self, features: "Features") -> torch.Tensor:
        """Apply the probe to the given features.

        The probe selects the part of the representation it reads from via
        [`select_features`][mlvbench.probes.Probe.select_features].

        Args:
            features: The features extracted from the backbone.

        Returns:
            The prediction from the probe as torch.Tensor. Shape and dtype very per
                probe.
        """

    @abstractmethod
    def serialize(self) -> dict[str, Any]:
        """Return all information needed to reconstruct the probe.

        This returns a self-describing dictionary that includes both the torch
        `state_dict` and all config options. The probe can be rebuilt from this
        representation via [`build_probe`][mlvbench.probes.build_probe].

        Returns:
            A dictionary describing the probe, with detached CPU tensors for `weights`.
        """


def build_probe(
    name: str,
    layer: str,
    input_dim: int,
    metadata: dict[str, Any] | None = None,
    state_dict: dict[str, Any] | None = None,
    probe_types: dict[str, type[Probe]] | None = None,
    **kwargs: Any,
) -> Probe:
    """Build the probe with the given name and arguments.

    Args:
        name: Name of the probe type (e.g. `"linear"`).
        layer: Name of the model layer the probe reads from.
        input_dim: Number of dimensions of the input features.
        metadata: Free-form metadata to attach to the probe.
        state_dict: Optional state dict for restoring pretrained probes.
        probe_types: Optional mapping of custom probe type names to `Probe` subclasses,
            merged with the built-in types. Each class must be constructible as
            `Cls(layer, input_dim, metadata=metadata, **kwargs)`.
        **kwargs: Additional keyword arguments forwarded to the probe constructor (e.g.
            `bias` and `num_patches`).

    Raises:
        ValueError: If `name` is not a known built-in or custom probe type.
    """
    types = {**BUILTIN_PROBE_TYPES, **(probe_types or {})}
    if name not in types:
        raise ValueError(f"Unknown probe type: {name}")
    probe = types[name](layer, input_dim, metadata=metadata, **kwargs)
    if state_dict is not None:
        probe.load_state_dict(state_dict)
    return probe


class LinearProbe(Probe):
    """Linear probe.

    A linear projection of the input features to a scalar, optionally with a learnable
    bias:

    $$
        y = W x + b
    $$
    """

    def __init__(
        self,
        layer: str,
        input_dim: int,
        bias: Literal["none", "scalar", "spatial"] | torch.Tensor = "scalar",
        num_patches: int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize the probe.

        Args:
            layer: Name of the model layer the probe reads from.
            input_dim: Number of dimensions of the input features.
            bias: The kind of learnable bias added to the output:

                - `"none"`: no bias.
                - `"scalar"`: a single learnable scalar shared across all patches.
                - `"spatial"`: a learnable per-patch bias with shape `(num_patches, 1)`,
                    initialized to zero. Requires `num_patches`.
                - a `torch.Tensor` (a prior with shape `(N, 1)`): a per-patch bias
                    initialized from the tensor; equivalent to `"spatial"` with
                    `num_patches = N`.

            num_patches: Number of patch tokens of the input features. Required when
                using a `"spatial"` bias and ignored otherwise.
            metadata: Free-form metadata to attach to the probe.
        """
        super().__init__(layer=layer, input_dim=input_dim, metadata=metadata)

        if isinstance(bias, torch.Tensor):
            self.bias_type = "spatial"
            self.bias = nn.Parameter(bias)

        elif bias == "spatial":
            if num_patches is None:
                raise ValueError("num_patches is required for a spatial bias.")
            self.bias_type = "spatial"
            self.bias = nn.Parameter(torch.zeros(num_patches, 1))

        elif bias in ("none", "scalar"):
            self.bias_type = bias
            self.register_parameter("bias", None)

        else:
            raise ValueError(f"Unknown bias type: {bias}")

        # We only use the standard bias of nn.Linear for a scalar bias, otherwise we
        # don't use a bias at all or a custom spatial bias.
        self.projection = nn.Linear(input_dim, 1, bias=(self.bias_type == "scalar"))

    @property
    def name(self) -> str:
        """Name of the probe type (`"linear"`)."""
        return "linear"

    def forward(self, features: "Features") -> torch.Tensor:
        """Forward pass.

        Args:
            features: The features extracted from the model. The probe uses the patch
                tokens of `self.layer`, expecting shape `(B, N, C)` and dtype `float32`.

        Returns:
            The output with shape `(B, N, 1)` and dtype `float32`.
        """
        out = self.projection(self.select_features(features))
        if self.bias is not None:
            out = out + self.bias
        return out

    def serialize(self) -> dict[str, Any]:
        """Serialize the probe.

        See [`Probe.serialize`][mlvbench.probes.Probe.serialize].
        """
        return {
            "name": self.name,
            "layer": self.layer,
            "input_dim": self.projection.in_features,
            "bias": self.bias_type,
            "num_patches": self.bias.shape[0] if self.bias is not None else None,
            "metadata": self.metadata,
            "state_dict": self.state_dict(),
        }


BUILTIN_PROBE_TYPES: dict[str, type[Probe]] = {
    "linear": LinearProbe,
}
