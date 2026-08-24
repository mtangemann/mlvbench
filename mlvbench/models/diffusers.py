"""Models from the diffusers library.

https://github.com/huggingface/diffusers
"""

from typing import Any, Literal

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel

from mlvbench.models._base import Features, Model, TokenLayout

# We only support the DiT XL models. Review the model class below when adding more
# models to make sure they are used correctly.
_SUPPORTED_MODELS = [
    "DiT-XL-2-256",
    "DiT-XL-2-512",
]


def list_diffusers_models() -> list[str]:
    """List available diffusers models."""
    return [f"diffusers/{name}" for name in _SUPPORTED_MODELS]


class DiffusersModel(Model):
    """Model from the diffusers library."""

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        precision: Literal["auto", "float32", "bfloat16"] = "auto",
        seed: int = 0,
        timestep: float = 0.0,
    ):
        """Initialize the model.

        Args:
            name: Model name without the `diffusers/` prefix. Have a look at
                `list_diffusers_models()` for available models.
            pretrained: Whether to load the pretrained weights. See `Model` for details.
                For `pretrained=False`, both the autoencoder and the transformer are
                randomly initialized, so no pretrained weights are involved at all.
            precision: The compute precision of the model. See `Model` for details.
            seed: Seed for the random weight initialization. See `Model` for details.
            timestep: Denoising timestep in `[0, 1]` used for the forward pass. No noise
                is added for `timestep=0` (default). For `timestep>0`, noise is added
                following a linear schedule.
        """
        super().__init__(pretrained=pretrained, precision=precision, seed=seed)

        if name not in _SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {name}.")

        self._name = name
        self._timestep = timestep

        vae_name = "stabilityai/sd-vae-ft-mse"
        huggingface_name = f"facebook/{self._name}"

        if self.pretrained:
            # VAE for encoding images to latent space.
            self._vae = AutoencoderKL.from_pretrained(vae_name)

            # Transformer for denoising latents.
            self._transformer = DiTTransformer2DModel.from_pretrained(
                huggingface_name, subfolder="transformer"
            )
        else:
            # Building both components from their configs alone gives the same
            # architectures with randomly initialized weights.
            vae_config = AutoencoderKL.load_config(vae_name)
            transformer_config = DiTTransformer2DModel.load_config(
                huggingface_name, subfolder="transformer"
            )
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self._vae = AutoencoderKL.from_config(vae_config)
                self._transformer = DiTTransformer2DModel.from_config(
                    transformer_config
                )

        self._vae.eval()
        self._transformer.eval()

        self._apply_precision()

    @property
    def name(self) -> str:
        """The full name of the model."""
        return f"diffusers/{self._name}"

    @property
    def url(self) -> str:
        """URL of the model card on Hugging Face."""
        huggingface_name = f"facebook/{self._name}"
        return f"https://huggingface.co/{huggingface_name}"

    @property
    def num_layers(self) -> int:
        """The number of transformer blocks."""
        return self._transformer.config.num_layers

    @property
    def layer_names(self) -> list[str]:
        """The names of the transformer blocks."""
        return [f"block.{i}" for i in range(self.num_layers)]

    @property
    def embed_dim(self) -> int:
        """The hidden dimension of each token."""
        config = self._transformer.config
        return config.num_attention_heads * config.attention_head_dim

    @property
    def patch_size(self) -> int:
        """The effective pixel-space patch size.

        The effective patch size is the product of the autoencoder downsampling factor
        and the latent space patch size used by the transformer.
        """
        return 8 * self._transformer.config.patch_size

    @property
    def input_size(self) -> int:
        """The native input image resolution."""
        return int(self._name.rsplit("-", 1)[1])

    @property
    def token_layout(self) -> TokenLayout:
        """The static token layout of the model.

        DiT operates purely on patch (latent) tokens and has no special tokens, so the
        prefix is empty.
        """
        return TokenLayout(prefix=(), grid_size=None)

    def config(self) -> dict[str, Any]:
        """Return the keyword arguments required to rebuild this model.

        Extends the base configuration with the denoising `timestep`, so that it
        survives a save/load round-trip.
        """
        return {**super().config(), "timestep": self._timestep}

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

        # The autoencoder expects pixel values normalized to [-1, 1]
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        images = images * 2.0 - 1.0
        images = images.to(self.dtype)

        grid_size = self._grid_size(images)

        # Encode to latent space using the mode (mean of the posterior) for
        # deterministic output.
        latent = self._vae.encode(images).latent_dist.mode()
        latent = latent * self._vae.config.scaling_factor

        if self._timestep > 0:
            # Add noise using a linear schedule.
            noise = torch.randn_like(latent)
            latent = (1.0 - self._timestep) * latent + self._timestep * noise

        # Add hooks to capture the outputs of each transformer block.
        features: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_index: int):
            def hook(
                module: torch.nn.Module, input: tuple, output: torch.Tensor
            ) -> None:
                if isinstance(output, tuple):
                    features[layer_index] = output[0].detach()
                else:
                    features[layer_index] = output.detach()

            return hook

        for layer_index in layer_indices:
            module = self._transformer.transformer_blocks[layer_index]
            handle = module.register_forward_hook(make_hook(layer_index))
            handles.append(handle)

        try:
            batch_size = images.shape[0]
            device = self.device
            timestep = torch.full(
                (batch_size,),
                int(self._timestep * 1000),  # The DiT encodes timestamps in [0, 1000]
                device=device,
                dtype=torch.long,
            )
            # Index 1000 is the unconditional (null) class used for classifier-free
            # guidance of ImageNet DiT models.
            class_labels = torch.full(
                (batch_size,), 1000, device=device, dtype=torch.long
            )
            self._transformer(latent, timestep=timestep, class_labels=class_labels)

        finally:
            # Remove hooks after the forward pass.
            for handle in handles:
                handle.remove()

        tokens = {f"block.{i}": features[i].to(torch.float32) for i in layer_indices}

        layout = TokenLayout(prefix=self.token_layout.prefix, grid_size=grid_size)
        return Features(tokens, layout)
