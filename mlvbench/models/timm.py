"""Models from the timm library."""

from typing import Literal

import timm
import torch
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms

from mlvbench.models._base import Features, Model, TokenLayout

# For now, we only support selected ViT models. We rely on the forward_intermediates()
# method to extract features, any model that provides this method should work in
# principle. Extending to all models should be straightforward using forward hooks.
# fmt: off
_SUPPORTED_MODELS = [
    # Original ViT models from Dosovitskiy et al. (2021)
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/vision_transformer.py#L1873
    "vit_base_patch16_224.orig_in21k_ft_in1k",
    "vit_base_patch16_384.orig_in21k_ft_in1k",
    "vit_large_patch32_384.orig_in21k_ft_in1k",
    "vit_base_patch32_224.orig_in21k",
    "vit_base_patch16_224.orig_in21k",
    "vit_large_patch32_224.orig_in21k",
    "vit_large_patch16_224.orig_in21k",
    "vit_huge_patch14_224.orig_in21k",

    # AugReg models from the "How to train your ViT?" paper
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/vision_transformer.py#L1813
    "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "vit_base_patch8_224.augreg2_in21k_ft_in1k",
    "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "vit_tiny_patch16_384.augreg_in21k_ft_in1k",
    "vit_small_patch32_224.augreg_in21k_ft_in1k",
    "vit_small_patch32_384.augreg_in21k_ft_in1k",
    "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_small_patch16_384.augreg_in21k_ft_in1k",
    "vit_base_patch32_224.augreg_in21k_ft_in1k",
    "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "vit_base_patch32_384.augreg_in21k_ft_in1k",
    "vit_base_patch16_384.augreg_in21k_ft_in1k",
    "vit_base_patch8_224.augreg_in21k_ft_in1k",
    "vit_large_patch16_224.augreg_in21k_ft_in1k",
    "vit_large_patch16_384.augreg_in21k_ft_in1k",
    "vit_small_patch16_224.augreg_in1k",
    "vit_small_patch16_384.augreg_in1k",
    "vit_base_patch32_224.augreg_in1k",
    "vit_base_patch32_384.augreg_in1k",
    "vit_base_patch16_224.augreg_in1k",
    "vit_base_patch16_384.augreg_in1k",
    "vit_tiny_patch16_224.augreg_in21k",
    "vit_small_patch32_224.augreg_in21k",
    "vit_small_patch16_224.augreg_in21k",
    "vit_base_patch32_224.augreg_in21k",
    "vit_base_patch16_224.augreg_in21k",
    "vit_base_patch8_224.augreg_in21k",
    "vit_large_patch16_224.augreg_in21k",

    # ImageNet-21K-P trained models by MIIL
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/vision_transformer.py#L2046
    "vit_base_patch16_224_miil.in21k",
    "vit_base_patch16_224_miil.in21k_ft_in1k",

    # Models trained using the SAM optimizer
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/vision_transformer.py#L1968
    "vit_base_patch32_224.sam_in1k",
    "vit_base_patch16_224.sam_in1k",

    # ViT variant trained on ImageNet with ROPE
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/eva.py#L1667
    "vit_small_patch16_rope_224.naver_in1k",
    "vit_base_patch16_rope_224.naver_in1k",
    "vit_large_patch16_rope_224.naver_in1k",
    "vit_small_patch16_rope_mixed_224.naver_in1k",
    "vit_base_patch16_rope_mixed_224.naver_in1k",
    "vit_large_patch16_rope_mixed_224.naver_in1k",
    "vit_small_patch16_rope_ape_224.naver_in1k",
    "vit_base_patch16_rope_ape_224.naver_in1k",
    "vit_large_patch16_rope_ape_224.naver_in1k",
    "vit_small_patch16_rope_mixed_ape_224.naver_in1k",
    "vit_base_patch16_rope_mixed_ape_224.naver_in1k",
    "vit_large_patch16_rope_mixed_ape_224.naver_in1k",

    # DeiT models trained on ImageNet-1K
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/deit.py#L155
    "deit_tiny_patch16_224.fb_in1k",
    "deit_small_patch16_224.fb_in1k",
    "deit_base_patch16_224.fb_in1k",
    "deit_base_patch16_384.fb_in1k",
    "deit_tiny_distilled_patch16_224.fb_in1k",
    "deit_small_distilled_patch16_224.fb_in1k",
    "deit_base_distilled_patch16_224.fb_in1k",
    "deit_base_distilled_patch16_384.fb_in1k",

    # DeiT III models trained on ImageNet-1K or pretrained on ImageNet-21K and
    # fine-tuned on ImageNet-1K
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/deit.py#L187
    "deit3_small_patch16_224.fb_in1k",
    "deit3_small_patch16_384.fb_in1k",
    "deit3_medium_patch16_224.fb_in1k",
    "deit3_base_patch16_224.fb_in1k",
    "deit3_base_patch16_384.fb_in1k",
    "deit3_large_patch16_224.fb_in1k",
    "deit3_large_patch16_384.fb_in1k",
    "deit3_huge_patch14_224.fb_in1k",
    "deit3_small_patch16_224.fb_in22k_ft_in1k",
    "deit3_small_patch16_384.fb_in22k_ft_in1k",
    "deit3_medium_patch16_224.fb_in22k_ft_in1k",
    "deit3_base_patch16_224.fb_in22k_ft_in1k",
    "deit3_base_patch16_384.fb_in22k_ft_in1k",
    "deit3_large_patch16_224.fb_in22k_ft_in1k",
    "deit3_large_patch16_384.fb_in22k_ft_in1k",
    "deit3_huge_patch14_224.fb_in22k_ft_in1k",

    # FlexiViT models trained on ImageNet-1K or ImageNet-21K
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/vision_transformer.py#L2385
    "flexivit_small.300ep_in1k",
    "flexivit_small.600ep_in1k",
    "flexivit_small.1200ep_in1k",
    "flexivit_base.300ep_in1k",
    "flexivit_base.600ep_in1k",
    "flexivit_base.1200ep_in1k",
    "flexivit_base.300ep_in21k",
    "flexivit_base.1000ep_in21k",
    "flexivit_base.patch16_in21k",
    "flexivit_base.patch30_in21k",
    "flexivit_large.300ep_in1k",
    "flexivit_large.600ep_in1k",
    "flexivit_large.1200ep_in1k",

    # CLIP models trained on OpenAI's WIT-400M dataset
    "vit_base_patch32_clip_224.openai",
    "vit_base_patch16_clip_224.openai",
    "vit_large_patch14_clip_224.openai",
    "vit_large_patch14_clip_336.openai",

    # CLIP models trained on OpenAI's WIT-400M dataset and finetuned on ImageNet
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/vision_transformer.py#L2105
    "vit_base_patch32_clip_224.openai_ft_in12k_in1k",
    "vit_base_patch32_clip_384.openai_ft_in12k_in1k",
    "vit_base_patch16_clip_224.openai_ft_in12k_in1k",
    "vit_base_patch16_clip_384.openai_ft_in12k_in1k",
    "vit_large_patch14_clip_224.openai_ft_in12k_in1k",
    "vit_large_patch14_clip_336.openai_ft_in12k_in1k",
    "vit_base_patch32_clip_224.openai_ft_in1k",
    "vit_base_patch16_clip_224.openai_ft_in1k",
    "vit_base_patch16_clip_384.openai_ft_in1k",
    "vit_large_patch14_clip_224.openai_ft_in1k",
    "vit_base_patch16_clip_224.openai_ft_in12k",
    "vit_large_patch14_clip_224.openai_ft_in12k",

    # CLIP models trained on LAION-2B
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/vision_transformer.py#L2183
    "vit_base_patch32_clip_224.laion2b",
    "vit_base_patch16_clip_224.laion2b",
    "vit_large_patch14_clip_224.laion2b",
    "vit_huge_patch14_clip_224.laion2b",
    "vit_giant_patch14_clip_224.laion2b",
    "vit_gigantic_patch14_clip_224.laion2b",

    # CLIP models trained on LAION-2B and finetuned on ImageNet
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/vision_transformer.py#L2073
    "vit_base_patch32_clip_224.laion2b_ft_in12k_in1k",
    "vit_base_patch32_clip_384.laion2b_ft_in12k_in1k",
    "vit_base_patch32_clip_448.laion2b_ft_in12k_in1k",
    "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
    "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k",
    "vit_large_patch14_clip_224.laion2b_ft_in12k_in1k",
    "vit_large_patch14_clip_336.laion2b_ft_in12k_in1k",
    "vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k",
    "vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k",
    "vit_base_patch32_clip_224.laion2b_ft_in1k",
    "vit_base_patch16_clip_224.laion2b_ft_in1k",
    "vit_base_patch16_clip_384.laion2b_ft_in1k",
    "vit_large_patch14_clip_224.laion2b_ft_in1k",
    "vit_large_patch14_clip_336.laion2b_ft_in1k",
    "vit_huge_patch14_clip_224.laion2b_ft_in1k",
    "vit_huge_patch14_clip_336.laion2b_ft_in1k",
    "vit_base_patch16_clip_224.laion2b_ft_in12k",
    "vit_large_patch14_clip_224.laion2b_ft_in12k",
    "vit_huge_patch14_clip_224.laion2b_ft_in12k",

    # EVA models finetuned on ImageNet
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/vision_transformer.py#L2362
    "eva_large_patch14_196.in22k_ft_in22k_in1k",
    "eva_large_patch14_336.in22k_ft_in22k_in1k",
    "eva_large_patch14_196.in22k_ft_in1k",
    "eva_large_patch14_336.in22k_ft_in1k",
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/eva.py#L1389
    "eva_giant_patch14_336.m30m_ft_in22k_in1k",
    "eva_giant_patch14_560.m30m_ft_in22k_in1k",

    # EVA-02, optionally finetuned on ImageNet
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/eva.py#L1401
    "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
    "eva02_large_patch14_448.mim_in22k_ft_in22k_in1k",
    "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
    "eva02_tiny_patch14_336.mim_in22k_ft_in1k",
    "eva02_small_patch14_336.mim_in22k_ft_in1k",
    "eva02_base_patch14_448.mim_in22k_ft_in1k",
    "eva02_large_patch14_448.mim_in22k_ft_in1k",
    "eva02_large_patch14_448.mim_m38m_ft_in1k",
    "eva02_base_patch14_448.mim_in22k_ft_in22k",
    "eva02_large_patch14_448.mim_in22k_ft_in22k",
    "eva02_large_patch14_448.mim_m38m_ft_in22k",
    "eva02_tiny_patch14_224.mim_in22k",
    "eva02_small_patch14_224.mim_in22k",
    "eva02_base_patch14_224.mim_in22k",
    "eva02_large_patch14_224.mim_in22k",
    "eva02_large_patch14_224.mim_m38m",

    # EVA01-CLIP and EVA02-CLIP, optionally finetuned on ImageNet
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/eva.py#L1379
    "eva_giant_patch14_224.clip_ft_in1k",
    "eva_giant_patch14_336.clip_ft_in1k",
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/eva.py#L1489
    "eva_giant_patch14_clip_224.laion400m",
    "eva_giant_patch14_clip_224.merged2b",
    "eva02_base_patch16_clip_224.merged2b",
    "eva02_large_patch14_clip_224.merged2b",
    "eva02_large_patch14_clip_336.merged2b",
    "eva02_enormous_patch14_clip_224.laion2b",
    "eva02_enormous_patch14_clip_224.laion2b_plus",
    "eva02_enormous_patch14_clip_224.pretrain",

    # SigLIP models trained on webli
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/vision_transformer.py#L2490
    "vit_base_patch16_siglip_224.webli",
    "vit_base_patch16_siglip_256.webli",
    "vit_base_patch16_siglip_256.webli_i18n",
    "vit_base_patch16_siglip_384.webli",
    "vit_base_patch16_siglip_512.webli",
    "vit_large_patch16_siglip_256.webli",
    "vit_large_patch16_siglip_384.webli",
    "vit_so400m_patch14_siglip_224.webli",
    "vit_so400m_patch14_siglip_378.webli",
    "vit_so400m_patch14_siglip_384.webli",
    "vit_so400m_patch16_siglip_256.webli_i18n",

    # SigLIP models trained on webli and finetuned on ImageNet
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/vision_transformer.py#L2766
    "vit_so400m_patch14_siglip_378.webli_ft_in1k",

    # SigLIP 2 models trained on webli
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/vision_transformer.py#L2483
    "vit_base_patch32_siglip_256.v2_webli",
    "vit_base_patch16_siglip_224.v2_webli",
    "vit_base_patch16_siglip_256.v2_webli",
    "vit_base_patch16_siglip_384.v2_webli",
    "vit_base_patch16_siglip_512.v2_webli",
    "vit_large_patch16_siglip_256.v2_webli",
    "vit_large_patch16_siglip_384.v2_webli",
    "vit_large_patch16_siglip_512.v2_webli",
    "vit_so400m_patch14_siglip_224.v2_webli",
    "vit_so400m_patch14_siglip_378.v2_webli",
    "vit_so400m_patch16_siglip_256.v2_webli",
    "vit_so400m_patch16_siglip_384.v2_webli",
    "vit_so400m_patch16_siglip_512.v2_webli",
    "vit_giantopt_patch16_siglip_256.v2_webli",
    "vit_giantopt_patch16_siglip_384.v2_webli",

    # BEiT-3 models trained on text and image datasets (pt) and optionally finetuned on
    # ImageNet
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/vision_transformer.py#L2986
    "beit3_base_patch16_224.in22k_ft_in1k",
    "beit3_base_patch16_224.indomain_in22k_ft_in1k",
    "beit3_large_patch16_224.in22k_ft_in1k",
    "beit3_large_patch16_224.indomain_in22k_ft_in1k",
    "beit3_base_patch16_224.pt",
    "beit3_base_patch16_224.indomain_pt",
    "beit3_large_patch16_224.pt",
    "beit3_large_patch16_224.indomain_pt",

    # Perception Encoder Core models trained on custom vision/text data
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/eva.py#L1565
    "vit_pe_core_tiny_patch16_384.fb",
    "vit_pe_core_small_patch16_384.fb",
    "vit_pe_core_base_patch16_224.fb",
    "vit_pe_core_large_patch14_336.fb",
    "vit_pe_core_gigantic_patch14_448.fb",

    # DINO self-supervised models
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/vision_transformer.py#L1976
    "vit_small_patch16_224.dino",
    "vit_small_patch8_224.dino",
    "vit_base_patch16_224.dino",
    "vit_base_patch8_224.dino",

    # DINOv2 self-supervised models, optionally with registers
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/vision_transformer.py#L1994
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
    "vit_large_patch14_dinov2.lvd142m",
    "vit_giant_patch14_dinov2.lvd142m",
    "vit_small_patch14_reg4_dinov2.lvd142m",
    "vit_base_patch14_reg4_dinov2.lvd142m",
    "vit_large_patch14_reg4_dinov2.lvd142m",
    "vit_giant_patch14_reg4_dinov2.lvd142m",

    # DINOv3 self-supervised models
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/eva.py#L1729
    "vit_small_patch16_dinov3.lvd1689m",
    "vit_small_patch16_dinov3_qkvb.lvd1689m",
    "vit_small_plus_patch16_dinov3.lvd1689m",
    "vit_small_plus_patch16_dinov3_qkvb.lvd1689m",
    "vit_base_patch16_dinov3.lvd1689m",
    "vit_base_patch16_dinov3_qkvb.lvd1689m",
    "vit_large_patch16_dinov3.lvd1689m",
    "vit_large_patch16_dinov3_qkvb.lvd1689m",
    "vit_large_patch16_dinov3.sat493m",
    "vit_large_patch16_dinov3_qkvb.sat493m",
    "vit_huge_plus_patch16_dinov3.lvd1689m",
    "vit_huge_plus_patch16_dinov3_qkvb.lvd1689m",
    "vit_7b_patch16_dinov3.lvd1689m",
    "vit_7b_patch16_dinov3.sat493m",

    # MAEs trained on ImageNet
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/vision_transformer.py#L2445
    "vit_base_patch16_224.mae",
    "vit_large_patch16_224.mae",
    "vit_huge_patch14_224.mae",

    # BEiT models pretrained on ImageNet-22K and optionally fine-tuned on ImageNet-1K
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/beit.py#L853
    "beit_base_patch16_224.in22k_ft_in22k",
    "beit_base_patch16_224.in22k_ft_in22k_in1k",
    "beit_base_patch16_384.in22k_ft_in22k_in1k",
    "beit_large_patch16_224.in22k_ft_in22k",
    "beit_large_patch16_224.in22k_ft_in22k_in1k",
    "beit_large_patch16_384.in22k_ft_in22k_in1k",
    "beit_large_patch16_512.in22k_ft_in22k_in1k",

    # BEiT v2 models pretrained on ImageNet-1K and optionally fine-tuned on ImageNet-22K
    # or ImageNet-1K
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/beit.py#L885
    "beitv2_base_patch16_224.in1k_ft_in1k",
    "beitv2_base_patch16_224.in1k_ft_in22k",
    "beitv2_base_patch16_224.in1k_ft_in22k_in1k",
    "beitv2_large_patch16_224.in1k_ft_in1k",
    "beitv2_large_patch16_224.in1k_ft_in22k",
    "beitv2_large_patch16_224.in1k_ft_in22k_in1k",

    # AIMv2 models pretrained on DFN-2B with autoregressive image modeling
    # https://github.com/huggingface/pytorch-image-models/blob/6e3fdda39508db30766f9d9e6ec32380ebee8b8c/timm/models/vision_transformer.py#L2917
    "aimv2_large_patch14_224.apple_pt",
    "aimv2_large_patch14_224.apple_pt_dist",
    "aimv2_large_patch14_336.apple_pt",
    "aimv2_large_patch14_336.apple_pt_dist",
    "aimv2_large_patch14_448.apple_pt",
    "aimv2_huge_patch14_224.apple_pt",
    "aimv2_huge_patch14_336.apple_pt",
    "aimv2_huge_patch14_448.apple_pt",
    "aimv2_1b_patch14_224.apple_pt",
    "aimv2_1b_patch14_336.apple_pt",
    "aimv2_1b_patch14_448.apple_pt",
    "aimv2_3b_patch14_224.apple_pt",
    "aimv2_3b_patch14_336.apple_pt",
    "aimv2_3b_patch14_448.apple_pt",
]
# fmt: on


def list_timm_models() -> list[str]:
    """List available timm models.

    Returns:
        List of model names with "timm/" prefix.
    """
    return [f"timm/{name}" for name in _SUPPORTED_MODELS]


class TimmModel(Model):
    """Model from the timm library."""

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        precision: Literal["auto", "float32", "bfloat16"] = "auto",
        seed: int = 0,
    ):
        """Initialize the model.

        Args:
            name: Model name without the `timm/` prefix. Have a look at `list_models()`
                for available models.
            pretrained: Whether to load the pretrained weights. See `Model` for details.
            precision: The compute precision of the model. See `Model` for details.
            seed: Seed for the random weight initialization. See `Model` for details.
        """
        super().__init__(pretrained=pretrained, precision=precision, seed=seed)

        if name not in _SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {name}")

        self._name = name

        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self._model = timm.create_model(name, pretrained=self.pretrained)
        self._model.eval()

        config = resolve_data_config(self._model.pretrained_cfg)
        self._transform = create_transform(**config)

        # Remove the resizing and cropping transforms for dense prediction tasks.
        self._transform.transforms = [
            transform
            for transform in self._transform.transforms
            if not isinstance(transform, transforms.Resize)
            and not isinstance(transform, transforms.CenterCrop)
        ]

        self._apply_precision()

    @property
    def name(self) -> str:
        """The full name of the model."""
        return f"timm/{self._name}"

    @property
    def url(self) -> str:
        """URL of the model card on Hugging Face."""
        return f"https://huggingface.co/timm/{self._name}"

    @property
    def num_layers(self) -> int:
        """The number of layers in the model."""
        return len(self._model.blocks)

    @property
    def layer_names(self) -> list[str]:
        """The names of the model layers."""
        return [f"block.{i}" for i in range(self.num_layers)]

    @property
    def embed_dim(self) -> int:
        """The number of dimensions for each token."""
        return self._model.embed_dim

    @property
    def patch_size(self) -> int:
        """The patch size of the model."""
        return self._model.patch_embed.patch_size[0]

    @property
    def input_size(self) -> int:
        """The native input image size the model was trained on."""
        return self._model.pretrained_cfg["input_size"][1]

    @property
    def token_layout(self) -> TokenLayout:
        """The static token layout of the model.

        The prefix order matches the token sequence produced by the model, i.e. the
        class token (if any), followed by the distillation token (if any), followed by
        the register tokens (if any). Models without special tokens (e.g. SigLIP, which
        uses global average pooling) have an empty prefix.
        """
        # References for token order:
        # https://github.com/huggingface/pytorch-image-models/blob/fbe27d6c935a00e357e13f2f74043645769bd5fc/timm/models/vision_transformer.py#L1041
        # https://github.com/huggingface/pytorch-image-models/blob/fbe27d6c935a00e357e13f2f74043645769bd5fc/timm/models/deit.py#L98
        # https://github.com/huggingface/pytorch-image-models/blob/fbe27d6c935a00e357e13f2f74043645769bd5fc/timm/models/eva.py#L875
        prefix: list[tuple[str, int]] = []

        if getattr(self._model, "cls_token", None) is not None:
            prefix.append(("cls", 1))

        if getattr(self._model, "dist_token", None) is not None:
            prefix.append(("distill", 1))

        reg_token = getattr(self._model, "reg_token", None)
        if reg_token is not None:
            prefix.append(("register", reg_token.shape[1]))

        # Double-check that the number of prefix token adds up
        num_prefix_tokens = sum(count for (_, count) in prefix)
        assert num_prefix_tokens == self._model.num_prefix_tokens

        return TokenLayout(prefix=tuple(prefix), grid_size=None)

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
        # The MaybeToTensor transform in timm does not convert uint8 tensors to float,
        # so we need to handle this case manually.
        if images.dtype == torch.uint8:
            images = images.float() / 255.0

        images = self._transform(images)
        images = images.to(self.dtype)

        grid_size = self._grid_size(images)

        if layers is not None:
            indices = [self.layer_names.index(layer) for layer in layers]
        else:
            indices = list(range(self.num_layers))

        features = self._model.forward_intermediates(
            images,
            indices=indices,
            return_prefix_tokens=True,
            stop_early=True,
            intermediates_only=True,
            output_fmt="NLC",
        )

        # With return_prefix_tokens, each entry is a (patch, prefix) tuple for models
        # with special tokens, or a plain patch tensor otherwise (e.g. SigLIP). We store
        # the full sequence [prefix, patch] and let the layout describe the split.
        tokens = {}
        for index, feature in zip(indices, features, strict=True):
            if isinstance(feature, tuple):
                patch, prefix = feature
                feature = torch.cat([prefix, patch], dim=1)
            tokens[f"block.{index}"] = feature.to(torch.float32)

        layout = TokenLayout(prefix=self.token_layout.prefix, grid_size=grid_size)
        return Features(tokens, layout)
