"""Web-SSL models (Fan et al. 2025).

https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Scaling_Language-Free_Visual_Representation_Learning_ICCV_2025_paper.html
https://github.com/facebookresearch/webssl
"""

from typing import Literal

import torch
import torchvision.transforms.v2 as transforms
from transformers import AutoImageProcessor, Dinov2Config, Dinov2Model

from mlvbench.models._base import Features, Model, TokenLayout

# Model names are adapted from the HuggingFace checkpoint, dropping the "webssl-" prefix
# and converting hyphens to underscores.
_SUPPORTED_MODELS = [
    "dino300m_full2b_224",
    "dino1b_full2b_224",
    "dino2b_full2b_224",
    "dino3b_full2b_224",
    "dino5b_full2b_224",
    "dino7b_full8b_224",
    "dino7b_full8b_378",
    "dino7b_full8b_518",
    "dino300m_light2b_224",
    "dino2b_light2b_224",
    "dino2b_heavy2b_224",
    "dino3b_light2b_224",
    "dino3b_heavy2b_224",
    "mae300m_full2b_224",
    "mae700m_full2b_224",
    "mae1b_full2b_224",
    "mae2b_full2b_224",
    "mae3b_full2b_224",
]


def list_webssl_models() -> list[str]:
    """List available Web-SSL models."""
    return [f"webssl/{name}" for name in _SUPPORTED_MODELS]


class WebSSLModel(Model):
    """Web-SSL model (Fan et al. 2025)."""

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        precision: Literal["auto", "float32", "bfloat16"] = "auto",
        seed: int = 0,
    ):
        """Initialize the model.

        Args:
            name: Model name without the `webssl/` prefix. Have a look at
                `list_webssl_models()` for available models.
            precision: The compute precision of the model. See `Model` for details.
            pretrained: Whether to load the pretrained weights. See `Model` for details.
            seed: Seed for the random weight initialization. See `Model` for details.
        """
        super().__init__(pretrained=pretrained, precision=precision, seed=seed)

        if name not in _SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {name}")

        self._name = name

        huggingface_name = f"facebook/webssl-{name.replace('_', '-')}"

        # We remove all resizing and cropping transforms from the processor, and only
        # keep the normalization. The processor only provides the input preprocessing
        # and is used for randomly initialized models as well.
        processor = AutoImageProcessor.from_pretrained(huggingface_name)
        assert processor.image_processor_type == "BitImageProcessor"
        assert processor.backend == "torchvision"
        self._transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=processor.image_mean, std=processor.image_std
                ),
            ]
        )

        if self.pretrained:
            self._model = Dinov2Model.from_pretrained(huggingface_name)
        else:
            # Building the model from the config alone gives the same architecture with
            # randomly initialized weights.
            config = Dinov2Config.from_pretrained(huggingface_name)
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self._model = Dinov2Model(config)
        self._model.eval()

        self._apply_precision()

    @property
    def name(self) -> str:
        """The full name of the model."""
        return f"webssl/{self._name}"

    @property
    def url(self) -> str:
        """URL of the model card on Hugging Face."""
        huggingface_name = f"facebook/webssl-{self._name.replace('_', '-')}"
        return f"https://huggingface.co/{huggingface_name}"

    @property
    def num_layers(self) -> int:
        """The number of layers in the model."""
        return self._model.config.num_hidden_layers

    @property
    def layer_names(self) -> list[str]:
        """The names of the model layers."""
        return [f"encoder.layer.{i}" for i in range(self.num_layers)]

    @property
    def embed_dim(self) -> int:
        """The number of dimensions for each token."""
        return self._model.config.hidden_size

    @property
    def patch_size(self) -> int:
        """The patch size of the model."""
        return self._model.config.patch_size

    @property
    def input_size(self) -> int:
        """The native input image size the model was trained on."""
        return self._model.config.image_size

    @property
    def token_layout(self) -> TokenLayout:
        """The static token layout of the model.

        The DINOv2 token sequence is `[cls, register..., patch...]`. Register tokens are
        only present in the `-reg` variants (`num_register_tokens > 0`).
        """
        num_register = getattr(self._model.config, "num_register_tokens", 0) or 0
        prefix: tuple[tuple[str, int], ...] = (("cls", 1),)
        if num_register:
            prefix += (("register", num_register),)
        return TokenLayout(prefix=prefix, grid_size=None)

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
        if layers is None:
            layer_indices = list(range(self.num_layers))
        else:
            layer_indices = [self.layer_names.index(layer) for layer in layers]

        if images.dtype == torch.uint8:
            images = images.float() / 255.0

        images = self._transform(images)
        images = images.to(self.dtype)

        grid_size = self._grid_size(images)

        output = self._model(images, output_hidden_states=True)

        # output.hidden_states is a list of one entry per layer plus the embedding
        # output at index 0. We thus need to add 1 to the layer indices to get the
        # correct hidden state.
        tokens = {
            f"encoder.layer.{index}": output.hidden_states[index + 1].to(torch.float32)
            for index in layer_indices
        }

        layout = TokenLayout(prefix=self.token_layout.prefix, grid_size=grid_size)
        return Features(tokens, layout)
