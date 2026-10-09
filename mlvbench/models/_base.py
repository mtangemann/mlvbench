"""Base class for models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import torch
from einops import rearrange

from mlvbench.probes import Probe

# Version of the checkpoint layout written by `Model.save`. Bump this whenever the
# structure of the saved dictionary changes in an incompatible way; `load_model` checks
# it and refuses checkpoints it does not understand. This is distinct from the
# `mlvbench_version` provenance stamp, which records the package version that created
# the checkpoint and is never used for compatibility decisions.
CHECKPOINT_FORMAT_VERSION = 2


def _mlvbench_version() -> str:
    """Return the installed mlvbench package version, or `"unknown"`."""
    try:
        return version("mlvbench")
    except PackageNotFoundError:
        return "unknown"


class Model(ABC, torch.nn.Module):
    """Base class for models."""

    def __init__(
        self,
        pretrained: bool = True,
        precision: Literal["auto", "float32", "bfloat16"] = "auto",
        seed: int = 0,
    ):
        """Initialize the model.

        Args:
            pretrained: If True (default), load the pretrained weights. If False, the
                architecture is built with randomly initialized weights.
            precision: The compute precision of the model. `"auto"` keeps the dtype of
                the loaded checkpoint, `"float32"` and `"bfloat16"` cast the weights to
                the respective dtype.
            seed: Seed for the random weight initialization when `pretrained=False`.
        """
        super().__init__()
        if precision not in ("auto", "float32", "bfloat16"):
            raise ValueError(
                f"Unsupported precision: {precision!r}. "
                "Expected 'auto', 'float32' or 'bfloat16'."
            )
        self._precision = precision
        self._pretrained = pretrained
        self._seed = seed

        self.probes: list[Probe] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the model."""

    @property
    @abstractmethod
    def url(self) -> str:
        """URL pointing to more information about the model.

        Ideally a model card (e.g. on Hugging Face) documenting details such as the
        citation and license. If no model card exists, this may instead link to the git
        repository or paper describing the model.
        """

    @property
    def num_parameters(self) -> int:
        """The number of parameters of the model."""
        return sum(p.numel() for p in self.parameters())

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """The number of layers in the model."""

    @property
    @abstractmethod
    def layer_names(self) -> list[str]:
        """The names of the model layers.

        Returns:
            A list of layer names, e.g. ["block.0", "block.1", "block.2"].
        """

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """The number of dimensions for each token."""

    @property
    @abstractmethod
    def patch_size(self) -> int:
        """The patch size of the model."""

    @property
    @abstractmethod
    def input_size(self) -> int:
        """The native input image size the model was trained on."""

    @property
    @abstractmethod
    def token_layout(self) -> "TokenLayout":
        """The static token layout of the model.

        Describes which prefix (special) token groups the model offers, such as cls or
        register tokens, without running a forward pass. `grid_size` is `None` because
        it depends on the input resolution; the layout carried by the
        [`Features`][mlvbench.models.Features] returned by
        [`forward_features`][mlvbench.models.Model.forward_features] fills it in.

        Use this to query a model's token types ahead of time, for example:

        ```python
        "cls" in model.token_layout        # whether a cls token is available
        model.token_layout.names           # e.g. ["cls", "register"]
        ```
        """

    @property
    def precision(self) -> Literal["auto", "float32", "bfloat16"]:
        """The requested compute precision of the model."""
        return self._precision

    @property
    def pretrained(self) -> bool:
        """Whether the model was built with pretrained weights.

        If False, the architecture is identical but the weights are randomly
        initialized.
        """
        return self._pretrained

    @property
    def seed(self) -> int:
        """The seed used for the random weight initialization."""
        return self._seed

    def info(self) -> dict[str, Any]:
        """Return information about the model."""
        return {
            "name": self.name,
            "url": self.url,
            "num_parameters": self.num_parameters,
            "num_layers": self.num_layers,
            "embed_dim": self.embed_dim,
            "patch_size": self.patch_size,
            "input_size": self.input_size,
            "token_groups": self.token_layout.names,
            "precision": self.precision,
            "pretrained": self.pretrained,
            "seed": self.seed,
        }

    def config(self) -> dict[str, Any]:
        """Return the keyword arguments required to rebuild this model.

        The returned mapping is passed to [`build_model`][mlvbench.models.build_model]
        to reconstruct an equivalent model, and is what
        [`Model.save`][mlvbench.models.Model.save] persists so that a saved model can be
        restored with [`load_model`][mlvbench.models.load_model].

        The base implementation captures `name`, `precision`, `pretrained` and `seed`.
        Adapters with extra constructor options should override this and
        extend the returned mapping, so that those options survive a save/load
        round-trip:

        ```python
        def config(self) -> dict[str, Any]:
            return {**super().config(), "timestep": self._timestep}
        ```
        """
        return {
            "name": self.name,
            "pretrained": self.pretrained,
            "precision": self.precision,
            "seed": self.seed,
        }

    @property
    def device(self) -> torch.device:
        """The device used by the model."""
        return next(iter(self.parameters())).device

    @property
    def dtype(self) -> torch.dtype:
        """The compute dtype of the model weights."""
        return next(iter(self.parameters())).dtype

    @property
    def compiled(self) -> bool:
        """Whether `forward_features` has been compiled with `torch.compile`.

        Detected via the marker `torch.compile` leaves on the wrapped callable, so it
        reflects the actual state regardless of how compilation was applied (e.g. by
        [`build_model`][mlvbench.models.build_model]).
        """
        return hasattr(self.forward_features, "_torchdynamo_orig_callable")

    def _apply_precision(self) -> None:
        """Cast the model weights to the requested precision.

        Called by subclasses once their submodules are built. `"auto"` keeps the
        checkpoint dtype; `"float32"` and `"bfloat16"` cast the weights accordingly.
        """
        match self._precision:
            case "auto":
                return
            case "float32":
                self.to(torch.float32)
            case "bfloat16":
                self.to(torch.bfloat16)

    def _grid_size(self, images: torch.Tensor) -> tuple[int, int]:
        """Compute the patch grid size for the given images.

        Validates that the spatial dimensions are divisible by the patch size, so that
        callers fail fast with a clear message instead of hitting an opaque shape error
        inside the backbone or when reshaping patch tokens to `BCHW`.

        Args:
            images: Input images with shape `(..., H, W)`.

        Returns:
            The `(H // patch_size, W // patch_size)` patch grid size.

        Raises:
            ValueError: If the height or width is not divisible by the patch size.
        """
        height, width = images.shape[-2], images.shape[-1]
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Input size {height}x{width} is not divisible by the patch size "
                f"{self.patch_size}. Resize the images so that both dimensions are a "
                "multiple of the patch size."
            )
        return height // self.patch_size, width // self.patch_size

    @abstractmethod
    def forward_features(
        self,
        images: torch.Tensor,
        layers: list[str] | None = None,
    ) -> Features:
        """Extract intermediate features for the images.

        Args:
            images: Input images with shape `(B, 3, H, W)` and dtype `uint8` or
                `float32`.
            layers: The list of layer names to extract features from. If None, features
                from all layers are extracted.

        Returns:
            Per-layer features extracted from the model, always with dtype `float32`
                regardless of the model's compute precision. The
                [`Features`][mlvbench.models.Features] container provides named access
                to the different token types, for example:

                ```python
                features.patch("block.11")                 # (B, H*W, C)
                features.patch("block.11", format="BCHW")  # (B, C, H, W)
                features.cls("block.11")                   # (B, C)
                ```
        """

    def forward_probes_from_features(self, features: "Features") -> list[torch.Tensor]:
        """Apply each attached probe to the given features.

        Args:
            features: Per-layer features as returned by
                [`forward_features`][mlvbench.models.Model.forward_features].

        Returns:
            A list of predictions in the order of `self.probes`.
        """
        return [probe(features) for probe in self.probes]

    def forward_probes(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Extract features and apply the attached probes.

        Args:
            images: Input images with shape `(B, 3, H, W)` and dtype `uint8` or
                `float32`.

        Returns:
            A list of predictions in the order of `self.probes`.
        """
        layers = list(dict.fromkeys(probe.layer for probe in self.probes))
        features = self.forward_features(images, layers=layers)
        return self.forward_probes_from_features(features)

    def save(self, path: Path, include_weights: bool = False) -> None:
        """Save this model's state so it can be restored with `load_model`.

        Stores the build configuration (so the pretrained backbone is rebuilt by name
        via [`config`][mlvbench.models.Model.config]) and all attached probes.
        Backbone weights are *not* stored by default — they are
        reconstructed from the pretrained checkpoint, which keeps the file small.

        Each probe stores its own `name`, so a model may hold probes of different types.
        Custom probe types (those not built in) must be supplied via the `probe_types`
        argument of [`load_model`][mlvbench.models.load_model] to rebuild the probes.

        Args:
            path: Destination file path.
            include_weights: If True, additionally store the backbone weights (its
                `state_dict`). Use this when the weights have been modified after
                loading; otherwise the pretrained checkpoint is sufficient.
        """
        checkpoint = {
            "mlvbench_checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "mlvbench_version": _mlvbench_version(),
            "config": self.config(),
            "probes": [probe.serialize() for probe in self.probes],
            "state_dict": self.state_dict() if include_weights else None,
        }
        torch.save(checkpoint, path)


@dataclass(frozen=True)
class TokenLayout:
    """Describes the token layout in Vision Transformers features.

    Vision Transformers process a flat sequence of tokens that includes the spatial
    patch tokens, and optionally special tokens (cls, register, ...). The TokenLayout
    describes how the tokens are arranged in the sequences.

    For example, if a model uses 1 cls token, 4 register tokens followed by patch tokens
    on a 10x10 grid, we exepct 105 tokens in total. This arrangement is described by the
    following layout:

    ```python
    TokenLayout(
        prefix=(("cls", 1), ("register", 4)),
        grid_size=(10, 10),
    )
    ```

    Attributes:
        prefix: Ordered prefix-token groups as ``(name, count)`` pairs, e.g.
            `(("cls", 1), ("register", 4))`. Empty if the model has no special tokens.
        grid_size: The `(H, W)` spatial grid size of the patch tokens, used to reshape
            patch tokens to `BCHW`. `None` if unknown.
    """

    prefix: tuple[tuple[str, int], ...] = ()
    grid_size: tuple[int, int] | None = None

    @property
    def num_prefix_tokens(self) -> int:
        """The total number of prefix (special) tokens across all groups."""
        return sum(count for _, count in self.prefix)

    @property
    def names(self) -> list[str]:
        """The names of the prefix-token groups, in token order.

        Returns:
            The group names, e.g. `["cls", "register"]`. Empty if the model has no
                special tokens.
        """
        return [name for name, _ in self.prefix]

    def __contains__(self, name: str) -> bool:
        """Whether the layout declares a prefix-token group with the given name.

        Enables membership tests such as `"cls" in model.token_layout`.

        Args:
            name: The name of a prefix-token group, e.g. `"cls"` or `"register"`.

        Returns:
            `True` if a group with this name is declared, `False` otherwise.
        """
        return any(group == name for group, _ in self.prefix)

    def span(self, name: str) -> tuple[int, int]:
        """Return the `[start, end)` index range of a prefix group.

        Args:
            name: The name of the prefix-token group (e.g., `"cls"` or `"register"`),

        Returns:
            The start (inclusive) and end (exclusive) token indices of the group.

        Raises:
            KeyError: If no prefix group with this name is declared.
        """
        offset = 0
        for group_name, count in self.prefix:
            if group_name == name:
                return offset, offset + count
            offset += count
        available = [group_name for group_name, _ in self.prefix]
        raise KeyError(
            f"Token group '{name}' is not available for this model. "
            f"Available groups: {available}."
        )


class Features:
    """Per-layer model features.

    Features from Vision Transformers are a sequence of tokens that include spatial
    patch tokens and may include special tokens (e.g., cls or register tokens). This
    class is a lightweight container for model features that simplifies accessing the
    different token types returned by a ViT.
    """

    def __init__(self, tokens: dict[str, torch.Tensor], layout: TokenLayout):
        """Initialize the container.

        Args:
            tokens: Mapping from layer name to the full token sequence with shape
                `(B, N_total, C)`, arranged as described by `layout`.
            layout: The token layout shared across all layers.
        """
        self._tokens = dict(tokens)
        self.layout = layout

    @property
    def layers(self) -> list[str]:
        """The names of the layers held by this container."""
        return list(self._tokens)

    def patch(self, layer: str, format: Literal["BNC", "BCHW"] = "BNC") -> torch.Tensor:
        """Return the patch tokens of a layer.

        Args:
            layer: The layer name.
            format: The output format, either `"BNC"` (default) for a flat sequence or
                `"BCHW"` for a spatial map.

        Returns:
            The patch tokens in the requested format.

        Raises:
            ValueError: If `format` is unknown, or `"BCHW"` is requested but the grid
                size is unknown or inconsistent with the token count.
        """
        patch = self._tokens[layer][:, self.layout.num_prefix_tokens :, :]

        if format == "BNC":
            return patch

        if format == "BCHW":
            if self.layout.grid_size is None:
                raise ValueError("Grid size is unknown; cannot reshape to BCHW.")
            height, width = self.layout.grid_size
            _, num_tokens, _ = patch.shape
            if num_tokens != height * width:
                raise ValueError(
                    f"Patch token count {num_tokens} does not match grid size "
                    f"{height}x{width}."
                )
            return rearrange(patch, "B (H W) C -> B C H W", H=height, W=width)

        raise ValueError(f"Unknown format '{format}', expected 'BNC' or 'BCHW'.")

    def subsample_patches(self, index: torch.Tensor) -> "Features":
        """Return a copy with the patch tokens gathered along the given index.

        The same indices are applied to every layer. Prefix (special) tokens are kept
        unchanged, while the patch tokens are gathered, so accessors such as
        [`cls`][mlvbench.models.Features.cls] keep working.

        Args:
            index: Patch-token indices with shape `(B, K)`, indexing into the patch
                tokens (i.e. excluding any prefix tokens).

        Returns:
            A new Features holding the same prefix tokens and the gathered patch tokens.
                The grid layout is dropped (`grid_size` is None), since the subsampled
                patch tokens no longer form a spatial grid.
        """
        batch_size, num_selected = index.shape
        batch_index = torch.arange(batch_size, device=index.device)[:, None].expand(
            -1, num_selected
        )
        num_prefix = self.layout.num_prefix_tokens

        tokens = {}
        for layer, sequence in self._tokens.items():
            prefix = sequence[:, :num_prefix, :]
            patch = sequence[:, num_prefix:, :][batch_index, index]
            tokens[layer] = torch.cat([prefix, patch], dim=1)

        layout = TokenLayout(prefix=self.layout.prefix, grid_size=None)
        return Features(tokens, layout)

    def special(self, layer: str, name: str) -> torch.Tensor:
        """Return a named group of special (prefix) tokens of a layer.

        Args:
            layer: The layer name.
            name: The name of the prefix-token group, e.g. `"cls"` or `"register"`.

        Returns:
            The tokens of the group with shape `(B, k, C)`, where `k` is the group size.

        Raises:
            KeyError: If the tokens do not include a group with that name.
        """
        start, end = self.layout.span(name)
        return self._tokens[layer][:, start:end, :]

    def cls(self, layer: str) -> torch.Tensor:
        """Return the cls (class) token of a layer.

        Args:
            layer: The layer name.

        Returns:
            The cls token with shape `(B, C)`.

        Raises:
            KeyError: If the model has no cls token.
        """
        return self.special(layer, "cls").squeeze(1)

    def register(self, layer: str) -> torch.Tensor:
        """Return the register tokens of a layer.

        Args:
            layer: The layer name.

        Returns:
            The register tokens with shape `(B, R, C)`.

        Raises:
            KeyError: If the model has no register tokens.
        """
        return self.special(layer, "register")
