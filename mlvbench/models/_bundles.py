"""Predefined bundles of models."""

# Tiny set of ViT-B models that span different training objectives. All models use a
# patch size of 16 and a resolution of 224, except for DINOv3 which uses an input
# resolution of 256.
_STANDARD5 = [
    "timm/vit_base_patch16_224.augreg_in21k_ft_in1k",
    "timm/vit_base_patch16_siglip_224.v2_webli",
    "timm/beit3_base_patch16_224.in22k_ft_in1k",
    "timm/vit_base_patch16_dinov3.lvd1689m",
    "timm/vit_base_patch16_224.mae",
]


# Small set of models that span different training objectives. For each model, a
# ViT-B and ViT-L variant is included. All models use a patch size of 16 and a
# resolution of 224, and a resolution of 256 for SigLIP ViT-L and DINOv3.
_STANDARD10 = [
    *_STANDARD5,
    "timm/vit_large_patch16_224.augreg_in21k_ft_in1k",
    "timm/vit_large_patch16_siglip_256.v2_webli",
    "timm/beit3_large_patch16_224.in22k_ft_in1k",
    "timm/vit_large_patch16_dinov3.lvd1689m",
    "timm/vit_large_patch16_224.mae",
]


# Medium set of diverse models. Extends the standard10 bundle with additional models,
# using ViT-B and ViT-L variants.
_STANDARD25 = [
    *_STANDARD10,
    # DeiT III
    "timm/deit3_base_patch16_224.fb_in22k_ft_in1k",
    "timm/deit3_large_patch16_224.fb_in22k_ft_in1k",
    # FlexiViT
    "timm/flexivit_base.1200ep_in1k",
    "timm/flexivit_large.1200ep_in1k",
    # CLIP
    "timm/vit_base_patch16_clip_224.openai",
    "timm/vit_large_patch14_clip_224.openai",
    # EVA-02
    "timm/eva02_base_patch14_224.mim_in22k",
    "timm/eva02_large_patch14_224.mim_in22k",
    # Perception Encoder
    "timm/vit_pe_core_base_patch16_224.fb",
    "timm/vit_pe_core_large_patch14_336.fb",
    # BEiT
    "timm/beit_base_patch16_224.in22k_ft_in22k",
    "timm/beit_large_patch16_224.in22k_ft_in22k",
    # DINO
    "timm/vit_base_patch16_224.dino",
    # DINOv2
    "timm/vit_base_patch14_reg4_dinov2.lvd142m",
    "timm/vit_large_patch14_reg4_dinov2.lvd142m",
]


BUNDLES = {
    "standard5": _STANDARD5,
    "standard10": _STANDARD10,
    "standard25": _STANDARD25,
}
