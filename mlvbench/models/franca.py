"""Franca models (Venkataramanan et al. 2026).

https://arxiv.org/abs/2507.14137
https://github.com/valeoai/Franca
"""

from typing import Literal

import torch
from torchvision import transforms

from mlvbench.models._base import Features, Model, TokenLayout

# Model names follow the naming convention in the official repository, but the franca_
# prefix is dropped in favour of the "franca/" provider prefix.
# https://github.com/valeoai/Franca/blob/52653cdd2f94fc7e4dd12655cf326b181a48091d/franca/hub/backbones.py#L21-L28
_SUPPORTED_MODELS = {
    "vitb14_In21k": {
        "model": "franca_vitb14",
        "weights": "IN21K",
        "use_rasa_head": False,
    },
    "vitb14_In21k_rasa": {
        "model": "franca_vitb14",
        "weights": "IN21K",
        "use_rasa_head": True,
    },
    "vitl14_In21k": {
        "model": "franca_vitl14",
        "weights": "IN21K",
        "use_rasa_head": False,
    },
    "vitl14_In21k_rasa": {
        "model": "franca_vitl14",
        "weights": "IN21K",
        "use_rasa_head": True,
    },
    "vitg14_In21k": {
        "model": "franca_vitg14",
        "weights": "IN21K",
        "use_rasa_head": False,
    },
    "vitg14_In21k_rasa": {
        "model": "franca_vitg14",
        "weights": "IN21K",
        "use_rasa_head": True,
    },
    "vitl14_Laion600M": {
        "model": "franca_vitl14",
        "weights": "LAION",
        "use_rasa_head": False,
    },
    "vitl14_Laion600M_rasa": {
        "model": "franca_vitl14",
        "weights": "LAION",
        "use_rasa_head": True,
    },
    "vitg14_Laion600M": {
        "model": "franca_vitg14",
        "weights": "LAION",
        "use_rasa_head": False,
    },
    "vitg14_Laion600M_rasa": {
        "model": "franca_vitg14",
        "weights": "LAION",
        "use_rasa_head": True,
    },
    "vitb14_Dinov2_In21k": {
        "model": "franca_vitb14",
        "weights": "DINOV2_IN21K",
        "use_rasa_head": False,
    },
    "vitb14_Dinov2_In21k_rasa": {
        "model": "franca_vitb14",
        "weights": "DINOV2_IN21K",
        "use_rasa_head": True,
    },
    "vitl14_Dinov2_In21k": {
        "model": "franca_vitl14",
        "weights": "DINOV2_IN21K",
        "use_rasa_head": False,
    },
    "vitl14_Dinov2_In21k_rasa": {
        "model": "franca_vitl14",
        "weights": "DINOV2_IN21K",
        "use_rasa_head": True,
    },
}


def list_franca_models() -> list[str]:
    """List available Franca models."""
    return [f"franca/{name}" for name in _SUPPORTED_MODELS.keys()]


class FrancaModel(Model):
    """Franca (Venkataramanan et al. 2026)."""

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        precision: Literal["auto", "float32", "bfloat16"] = "auto",
        seed: int = 0,
    ):
        """Initialize the model.

        Args:
            name: Model name without the `franca/` prefix. Have a look at
                `list_franca_models()` for available models.
            pretrained: Whether to load the pretrained weights. See `Model` for details.
                The torch.hub implementation of Franca doesn' support
            precision: The compute precision of the model. See `Model` for details.
            seed: Seed for the random weight initialization. See `Model` for details.
        """
        super().__init__(pretrained=pretrained, precision=precision, seed=seed)

        if name not in _SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {name}")

        kwargs = _SUPPORTED_MODELS[name]

        # The Franca torch.hub factory always loads pretrained RASA weights even when
        # passing `pretrained=False`.
        # https://github.com/valeoai/Franca/blob/52653cdd2f94fc7e4dd12655cf326b181a48091d/franca/hub/backbones.py#L133-L151
        if kwargs["use_rasa_head"] and not pretrained:
            raise ValueError(
                "The Franca RASA head is only supported when using pretrained models. "
                "Either use a model without RASA head or set `pretrained=True`."
            )

        self._name = name
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self._model = torch.hub.load(
                "valeoai/Franca", pretrained=pretrained, **kwargs
            )
        self._model.eval()

        # The models loaded from torch hub don't include the input transforms. We
        # manually create the transforms as described in the Franca README, but exclude
        # the resizing and cropping transforms.
        self._transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        self._apply_precision()

    @property
    def name(self) -> str:
        """The full name of the model."""
        return f"franca/{self._name}"

    @property
    def url(self) -> str:
        """URL of the official Franca repository.

        Franca does not provide per-model cards. We link to the repository instead,
        which documents the models and their license.
        """
        return "https://github.com/valeoai/Franca"

    @property
    def num_layers(self) -> int:
        """The number of layers in the model."""
        return len(self._model.blocks)

    @property
    def layer_names(self) -> list[str]:
        """The names of the model layers."""
        return [f"blocks.{i}" for i in range(self.num_layers)]

    @property
    def embed_dim(self) -> int:
        """The number of dimensions for each token."""
        return self._model.embed_dim

    @property
    def patch_size(self) -> int:
        """The patch size of the model."""
        return self._model.patch_size

    @property
    def input_size(self) -> int:
        """The native input image size the model was trained on."""
        # According to the Franca README, all models were trained on 224x224 images.
        return 224

    @property
    def token_layout(self) -> TokenLayout:
        """The static token layout of the model.

        All supported Franca checkpoints are trained without register tokens
        (`num_register_tokens=0`), so the only special token is the clss token.
        """
        return TokenLayout(prefix=(("cls", 1),), grid_size=None)

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
            Per-layer features extracted from the model.
        """
        # We expect Tensors as input, so we don't need the ToTensor() transform.
        # However, we still need to convert uint8 tensors to float.
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        images = self._transform(images)
        images = images.to(self.dtype)

        grid_size = self._grid_size(images)

        if layers is not None:
            layer_indices = [self.layer_names.index(layer) for layer in layers]
        else:
            layer_indices = list(range(self.num_layers))

        # get_intermediate_layers returns a (patch_tokens, class_token) pair per
        # requested block. The supported checkpoints have no register tokens, so the
        # full sequence is simply [cls, patch] (see token_layout).
        features = self._model.get_intermediate_layers(
            images, layer_indices, return_class_token=True
        )

        tokens = {}
        for index, (patch, cls) in zip(layer_indices, features, strict=True):
            token = torch.cat([cls.unsqueeze(1), patch], dim=1)
            tokens[f"blocks.{index}"] = token.to(torch.float32)

        layout = TokenLayout(prefix=self.token_layout.prefix, grid_size=grid_size)
        return Features(tokens, layout)
