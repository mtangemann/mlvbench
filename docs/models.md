# Supported Models

MLV-Bench supports Vision Transformer (ViT) models from multiple sources. This document lists all available models and links to further information. To use a model, call [`build_model`](API/mlvbench.models/#mlvbench.models.build_model) with the `<provider>/<name>` identifiers listed below.

## AIMv2
[Fini et al. (2024)](https://arxiv.org/abs/2411.14402)

Apple ViTs pretrained with a causal autoregressive objective over image patches conditioned on multimodal (image + text) inputs from DFN-2B, with optional teacher distillation (`_dist`).

- [`timm/aimv2_1b_patch14_224.apple_pt`](https://huggingface.co/timm/aimv2_1b_patch14_224.apple_pt)
- [`timm/aimv2_1b_patch14_336.apple_pt`](https://huggingface.co/timm/aimv2_1b_patch14_336.apple_pt)
- [`timm/aimv2_1b_patch14_448.apple_pt`](https://huggingface.co/timm/aimv2_1b_patch14_448.apple_pt)
- [`timm/aimv2_3b_patch14_224.apple_pt`](https://huggingface.co/timm/aimv2_3b_patch14_224.apple_pt)
- [`timm/aimv2_3b_patch14_336.apple_pt`](https://huggingface.co/timm/aimv2_3b_patch14_336.apple_pt)
- [`timm/aimv2_3b_patch14_448.apple_pt`](https://huggingface.co/timm/aimv2_3b_patch14_448.apple_pt)
- [`timm/aimv2_huge_patch14_224.apple_pt`](https://huggingface.co/timm/aimv2_huge_patch14_224.apple_pt)
- [`timm/aimv2_huge_patch14_336.apple_pt`](https://huggingface.co/timm/aimv2_huge_patch14_336.apple_pt)
- [`timm/aimv2_huge_patch14_448.apple_pt`](https://huggingface.co/timm/aimv2_huge_patch14_448.apple_pt)
- [`timm/aimv2_large_patch14_224.apple_pt`](https://huggingface.co/timm/aimv2_large_patch14_224.apple_pt)
- [`timm/aimv2_large_patch14_224.apple_pt_dist`](https://huggingface.co/timm/aimv2_large_patch14_224.apple_pt_dist)
- [`timm/aimv2_large_patch14_336.apple_pt`](https://huggingface.co/timm/aimv2_large_patch14_336.apple_pt)
- [`timm/aimv2_large_patch14_336.apple_pt_dist`](https://huggingface.co/timm/aimv2_large_patch14_336.apple_pt_dist)
- [`timm/aimv2_large_patch14_448.apple_pt`](https://huggingface.co/timm/aimv2_large_patch14_448.apple_pt)

## AugReg
[Steiner et al. (2021)](https://arxiv.org/abs/2106.10270)

ViTs trained on ImageNet-21K/-1K with carefully tuned augmentation and regularization, matching models trained on an order of magnitude more data.

- [`timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
- [`timm/vit_base_patch16_224.augreg_in1k`](https://huggingface.co/timm/vit_base_patch16_224.augreg_in1k)
- [`timm/vit_base_patch16_224.augreg_in21k`](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k)
- [`timm/vit_base_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k)
- [`timm/vit_base_patch16_384.augreg_in1k`](https://huggingface.co/timm/vit_base_patch16_384.augreg_in1k)
- [`timm/vit_base_patch16_384.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_384.augreg_in21k_ft_in1k)
- [`timm/vit_base_patch32_224.augreg_in1k`](https://huggingface.co/timm/vit_base_patch32_224.augreg_in1k)
- [`timm/vit_base_patch32_224.augreg_in21k`](https://huggingface.co/timm/vit_base_patch32_224.augreg_in21k)
- [`timm/vit_base_patch32_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch32_224.augreg_in21k_ft_in1k)
- [`timm/vit_base_patch32_384.augreg_in1k`](https://huggingface.co/timm/vit_base_patch32_384.augreg_in1k)
- [`timm/vit_base_patch32_384.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch32_384.augreg_in21k_ft_in1k)
- [`timm/vit_base_patch8_224.augreg2_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch8_224.augreg2_in21k_ft_in1k)
- [`timm/vit_base_patch8_224.augreg_in21k`](https://huggingface.co/timm/vit_base_patch8_224.augreg_in21k)
- [`timm/vit_base_patch8_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch8_224.augreg_in21k_ft_in1k)
- [`timm/vit_large_patch16_224.augreg_in21k`](https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k)
- [`timm/vit_large_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k_ft_in1k)
- [`timm/vit_large_patch16_384.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_large_patch16_384.augreg_in21k_ft_in1k)
- [`timm/vit_small_patch16_224.augreg_in1k`](https://huggingface.co/timm/vit_small_patch16_224.augreg_in1k)
- [`timm/vit_small_patch16_224.augreg_in21k`](https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k)
- [`timm/vit_small_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k)
- [`timm/vit_small_patch16_384.augreg_in1k`](https://huggingface.co/timm/vit_small_patch16_384.augreg_in1k)
- [`timm/vit_small_patch16_384.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_small_patch16_384.augreg_in21k_ft_in1k)
- [`timm/vit_small_patch32_224.augreg_in21k`](https://huggingface.co/timm/vit_small_patch32_224.augreg_in21k)
- [`timm/vit_small_patch32_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_small_patch32_224.augreg_in21k_ft_in1k)
- [`timm/vit_small_patch32_384.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_small_patch32_384.augreg_in21k_ft_in1k)
- [`timm/vit_tiny_patch16_224.augreg_in21k`](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k)
- [`timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k)
- [`timm/vit_tiny_patch16_384.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_tiny_patch16_384.augreg_in21k_ft_in1k)

## BEiT
[Bao et al. (2022)](https://openreview.net/forum?id=p-BhZSz59o4)

ViTs pretrained by masked image modeling against discrete dVAE visual tokens on ImageNet-22K, optionally fine-tuned on ImageNet-1K.

- [`timm/beit_base_patch16_224.in22k_ft_in22k`](https://huggingface.co/timm/beit_base_patch16_224.in22k_ft_in22k)
- [`timm/beit_base_patch16_224.in22k_ft_in22k_in1k`](https://huggingface.co/timm/beit_base_patch16_224.in22k_ft_in22k_in1k)
- [`timm/beit_base_patch16_384.in22k_ft_in22k_in1k`](https://huggingface.co/timm/beit_base_patch16_384.in22k_ft_in22k_in1k)
- [`timm/beit_large_patch16_224.in22k_ft_in22k`](https://huggingface.co/timm/beit_large_patch16_224.in22k_ft_in22k)
- [`timm/beit_large_patch16_224.in22k_ft_in22k_in1k`](https://huggingface.co/timm/beit_large_patch16_224.in22k_ft_in22k_in1k)
- [`timm/beit_large_patch16_384.in22k_ft_in22k_in1k`](https://huggingface.co/timm/beit_large_patch16_384.in22k_ft_in22k_in1k)
- [`timm/beit_large_patch16_512.in22k_ft_in22k_in1k`](https://huggingface.co/timm/beit_large_patch16_512.in22k_ft_in22k_in1k)

## BEiT v2
[Peng et al. (2022)](https://arxiv.org/abs/2208.06366)

A BEiT variant that reconstructs CLIP visual features instead of dVAE tokens during masked image modeling.

- [`timm/beitv2_base_patch16_224.in1k_ft_in1k`](https://huggingface.co/timm/beitv2_base_patch16_224.in1k_ft_in1k)
- [`timm/beitv2_base_patch16_224.in1k_ft_in22k`](https://huggingface.co/timm/beitv2_base_patch16_224.in1k_ft_in22k)
- [`timm/beitv2_base_patch16_224.in1k_ft_in22k_in1k`](https://huggingface.co/timm/beitv2_base_patch16_224.in1k_ft_in22k_in1k)
- [`timm/beitv2_large_patch16_224.in1k_ft_in1k`](https://huggingface.co/timm/beitv2_large_patch16_224.in1k_ft_in1k)
- [`timm/beitv2_large_patch16_224.in1k_ft_in22k`](https://huggingface.co/timm/beitv2_large_patch16_224.in1k_ft_in22k)
- [`timm/beitv2_large_patch16_224.in1k_ft_in22k_in1k`](https://huggingface.co/timm/beitv2_large_patch16_224.in1k_ft_in22k_in1k)

## BEiT-3
[Wang et al. (2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Image_as_a_Foreign_Language_BEiT_Pretraining_for_Vision_and_CVPR_2023_paper.html)

A multimodal foundation model applying masked data modeling across text, images, and image–text pairs with a shared Multiway Transformer, fine-tuned on ImageNet for classification.

- [`timm/beit3_base_patch16_224.in22k_ft_in1k`](https://huggingface.co/timm/beit3_base_patch16_224.in22k_ft_in1k)
- [`timm/beit3_base_patch16_224.indomain_in22k_ft_in1k`](https://huggingface.co/timm/beit3_base_patch16_224.indomain_in22k_ft_in1k)
- [`timm/beit3_base_patch16_224.indomain_pt`](https://huggingface.co/timm/beit3_base_patch16_224.indomain_pt)
- [`timm/beit3_base_patch16_224.pt`](https://huggingface.co/timm/beit3_base_patch16_224.pt)
- [`timm/beit3_large_patch16_224.in22k_ft_in1k`](https://huggingface.co/timm/beit3_large_patch16_224.in22k_ft_in1k)
- [`timm/beit3_large_patch16_224.indomain_in22k_ft_in1k`](https://huggingface.co/timm/beit3_large_patch16_224.indomain_in22k_ft_in1k)
- [`timm/beit3_large_patch16_224.indomain_pt`](https://huggingface.co/timm/beit3_large_patch16_224.indomain_pt)
- [`timm/beit3_large_patch16_224.pt`](https://huggingface.co/timm/beit3_large_patch16_224.pt)

## CLIP (LAION-2B)
[Cherti et al. (2023)](https://arxiv.org/abs/2212.07143)

OpenCLIP reproduction of CLIP trained on the open [LAION-2B](https://laion.ai/blog/laion-5b/) dataset, enabling larger model sizes and public reproducibility.

- [`timm/vit_base_patch16_clip_224.laion2b`](https://huggingface.co/timm/vit_base_patch16_clip_224.laion2b)
- [`timm/vit_base_patch16_clip_224.laion2b_ft_in12k`](https://huggingface.co/timm/vit_base_patch16_clip_224.laion2b_ft_in12k)
- [`timm/vit_base_patch16_clip_224.laion2b_ft_in12k_in1k`](https://huggingface.co/timm/vit_base_patch16_clip_224.laion2b_ft_in12k_in1k)
- [`timm/vit_base_patch16_clip_224.laion2b_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_clip_224.laion2b_ft_in1k)
- [`timm/vit_base_patch16_clip_384.laion2b_ft_in12k_in1k`](https://huggingface.co/timm/vit_base_patch16_clip_384.laion2b_ft_in12k_in1k)
- [`timm/vit_base_patch16_clip_384.laion2b_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_clip_384.laion2b_ft_in1k)
- [`timm/vit_base_patch32_clip_224.laion2b`](https://huggingface.co/timm/vit_base_patch32_clip_224.laion2b)
- [`timm/vit_base_patch32_clip_224.laion2b_ft_in12k_in1k`](https://huggingface.co/timm/vit_base_patch32_clip_224.laion2b_ft_in12k_in1k)
- [`timm/vit_base_patch32_clip_224.laion2b_ft_in1k`](https://huggingface.co/timm/vit_base_patch32_clip_224.laion2b_ft_in1k)
- [`timm/vit_base_patch32_clip_384.laion2b_ft_in12k_in1k`](https://huggingface.co/timm/vit_base_patch32_clip_384.laion2b_ft_in12k_in1k)
- [`timm/vit_base_patch32_clip_448.laion2b_ft_in12k_in1k`](https://huggingface.co/timm/vit_base_patch32_clip_448.laion2b_ft_in12k_in1k)
- [`timm/vit_giant_patch14_clip_224.laion2b`](https://huggingface.co/timm/vit_giant_patch14_clip_224.laion2b)
- [`timm/vit_gigantic_patch14_clip_224.laion2b`](https://huggingface.co/timm/vit_gigantic_patch14_clip_224.laion2b)
- [`timm/vit_huge_patch14_clip_224.laion2b`](https://huggingface.co/timm/vit_huge_patch14_clip_224.laion2b)
- [`timm/vit_huge_patch14_clip_224.laion2b_ft_in12k`](https://huggingface.co/timm/vit_huge_patch14_clip_224.laion2b_ft_in12k)
- [`timm/vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k`](https://huggingface.co/timm/vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k)
- [`timm/vit_huge_patch14_clip_224.laion2b_ft_in1k`](https://huggingface.co/timm/vit_huge_patch14_clip_224.laion2b_ft_in1k)
- [`timm/vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k`](https://huggingface.co/timm/vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k)
- [`timm/vit_huge_patch14_clip_336.laion2b_ft_in1k`](https://huggingface.co/timm/vit_huge_patch14_clip_336.laion2b_ft_in1k)
- [`timm/vit_large_patch14_clip_224.laion2b`](https://huggingface.co/timm/vit_large_patch14_clip_224.laion2b)
- [`timm/vit_large_patch14_clip_224.laion2b_ft_in12k`](https://huggingface.co/timm/vit_large_patch14_clip_224.laion2b_ft_in12k)
- [`timm/vit_large_patch14_clip_224.laion2b_ft_in12k_in1k`](https://huggingface.co/timm/vit_large_patch14_clip_224.laion2b_ft_in12k_in1k)
- [`timm/vit_large_patch14_clip_224.laion2b_ft_in1k`](https://huggingface.co/timm/vit_large_patch14_clip_224.laion2b_ft_in1k)
- [`timm/vit_large_patch14_clip_336.laion2b_ft_in12k_in1k`](https://huggingface.co/timm/vit_large_patch14_clip_336.laion2b_ft_in12k_in1k)
- [`timm/vit_large_patch14_clip_336.laion2b_ft_in1k`](https://huggingface.co/timm/vit_large_patch14_clip_336.laion2b_ft_in1k)

## CLIP (OpenAI WIT-400M)
[Radford et al. (2021)](https://proceedings.mlr.press/v139/radford21a.html)

The original CLIP image encoders, aligned with text by a contrastive objective on OpenAI's private WIT-400M dataset, with optional ImageNet fine-tuning.

- [`timm/vit_base_patch16_clip_224.openai`](https://huggingface.co/timm/vit_base_patch16_clip_224.openai)
- [`timm/vit_base_patch16_clip_224.openai_ft_in12k`](https://huggingface.co/timm/vit_base_patch16_clip_224.openai_ft_in12k)
- [`timm/vit_base_patch16_clip_224.openai_ft_in12k_in1k`](https://huggingface.co/timm/vit_base_patch16_clip_224.openai_ft_in12k_in1k)
- [`timm/vit_base_patch16_clip_224.openai_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_clip_224.openai_ft_in1k)
- [`timm/vit_base_patch16_clip_384.openai_ft_in12k_in1k`](https://huggingface.co/timm/vit_base_patch16_clip_384.openai_ft_in12k_in1k)
- [`timm/vit_base_patch16_clip_384.openai_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_clip_384.openai_ft_in1k)
- [`timm/vit_base_patch32_clip_224.openai`](https://huggingface.co/timm/vit_base_patch32_clip_224.openai)
- [`timm/vit_base_patch32_clip_224.openai_ft_in12k_in1k`](https://huggingface.co/timm/vit_base_patch32_clip_224.openai_ft_in12k_in1k)
- [`timm/vit_base_patch32_clip_224.openai_ft_in1k`](https://huggingface.co/timm/vit_base_patch32_clip_224.openai_ft_in1k)
- [`timm/vit_base_patch32_clip_384.openai_ft_in12k_in1k`](https://huggingface.co/timm/vit_base_patch32_clip_384.openai_ft_in12k_in1k)
- [`timm/vit_large_patch14_clip_224.openai`](https://huggingface.co/timm/vit_large_patch14_clip_224.openai)
- [`timm/vit_large_patch14_clip_224.openai_ft_in12k`](https://huggingface.co/timm/vit_large_patch14_clip_224.openai_ft_in12k)
- [`timm/vit_large_patch14_clip_224.openai_ft_in12k_in1k`](https://huggingface.co/timm/vit_large_patch14_clip_224.openai_ft_in12k_in1k)
- [`timm/vit_large_patch14_clip_224.openai_ft_in1k`](https://huggingface.co/timm/vit_large_patch14_clip_224.openai_ft_in1k)
- [`timm/vit_large_patch14_clip_336.openai`](https://huggingface.co/timm/vit_large_patch14_clip_336.openai)
- [`timm/vit_large_patch14_clip_336.openai_ft_in12k_in1k`](https://huggingface.co/timm/vit_large_patch14_clip_336.openai_ft_in12k_in1k)

## DeiT
[Touvron et al. (2021)](https://proceedings.mlr.press/v139/touvron21a.html)

Data-efficient ViTs trained competitively on ImageNet-1K alone via knowledge distillation, with distilled variants adding a dedicated distillation token.

- [`timm/deit_base_distilled_patch16_224.fb_in1k`](https://huggingface.co/timm/deit_base_distilled_patch16_224.fb_in1k)
- [`timm/deit_base_distilled_patch16_384.fb_in1k`](https://huggingface.co/timm/deit_base_distilled_patch16_384.fb_in1k)
- [`timm/deit_base_patch16_224.fb_in1k`](https://huggingface.co/timm/deit_base_patch16_224.fb_in1k)
- [`timm/deit_base_patch16_384.fb_in1k`](https://huggingface.co/timm/deit_base_patch16_384.fb_in1k)
- [`timm/deit_small_distilled_patch16_224.fb_in1k`](https://huggingface.co/timm/deit_small_distilled_patch16_224.fb_in1k)
- [`timm/deit_small_patch16_224.fb_in1k`](https://huggingface.co/timm/deit_small_patch16_224.fb_in1k)
- [`timm/deit_tiny_distilled_patch16_224.fb_in1k`](https://huggingface.co/timm/deit_tiny_distilled_patch16_224.fb_in1k)
- [`timm/deit_tiny_patch16_224.fb_in1k`](https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k)

## DeiT III
[Touvron et al. (2022)](https://link.springer.com/chapter/10.1007/978-3-031-20083-0_26)

A revised DeiT recipe with stronger augmentation and supervised-training best practices, trained on ImageNet-1K or pretrained on ImageNet-21K and fine-tuned on ImageNet-1K.

- [`timm/deit3_base_patch16_224.fb_in1k`](https://huggingface.co/timm/deit3_base_patch16_224.fb_in1k)
- [`timm/deit3_base_patch16_224.fb_in22k_ft_in1k`](https://huggingface.co/timm/deit3_base_patch16_224.fb_in22k_ft_in1k)
- [`timm/deit3_base_patch16_384.fb_in1k`](https://huggingface.co/timm/deit3_base_patch16_384.fb_in1k)
- [`timm/deit3_base_patch16_384.fb_in22k_ft_in1k`](https://huggingface.co/timm/deit3_base_patch16_384.fb_in22k_ft_in1k)
- [`timm/deit3_huge_patch14_224.fb_in1k`](https://huggingface.co/timm/deit3_huge_patch14_224.fb_in1k)
- [`timm/deit3_huge_patch14_224.fb_in22k_ft_in1k`](https://huggingface.co/timm/deit3_huge_patch14_224.fb_in22k_ft_in1k)
- [`timm/deit3_large_patch16_224.fb_in1k`](https://huggingface.co/timm/deit3_large_patch16_224.fb_in1k)
- [`timm/deit3_large_patch16_224.fb_in22k_ft_in1k`](https://huggingface.co/timm/deit3_large_patch16_224.fb_in22k_ft_in1k)
- [`timm/deit3_large_patch16_384.fb_in1k`](https://huggingface.co/timm/deit3_large_patch16_384.fb_in1k)
- [`timm/deit3_large_patch16_384.fb_in22k_ft_in1k`](https://huggingface.co/timm/deit3_large_patch16_384.fb_in22k_ft_in1k)
- [`timm/deit3_medium_patch16_224.fb_in1k`](https://huggingface.co/timm/deit3_medium_patch16_224.fb_in1k)
- [`timm/deit3_medium_patch16_224.fb_in22k_ft_in1k`](https://huggingface.co/timm/deit3_medium_patch16_224.fb_in22k_ft_in1k)
- [`timm/deit3_small_patch16_224.fb_in1k`](https://huggingface.co/timm/deit3_small_patch16_224.fb_in1k)
- [`timm/deit3_small_patch16_224.fb_in22k_ft_in1k`](https://huggingface.co/timm/deit3_small_patch16_224.fb_in22k_ft_in1k)
- [`timm/deit3_small_patch16_384.fb_in1k`](https://huggingface.co/timm/deit3_small_patch16_384.fb_in1k)
- [`timm/deit3_small_patch16_384.fb_in22k_ft_in1k`](https://huggingface.co/timm/deit3_small_patch16_384.fb_in22k_ft_in1k)

## DINO
[Caron et al. (2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper)

Self-supervised ViTs trained by self-distillation between a student and a momentum teacher, exhibiting emergent object segmentation.

- [`timm/vit_base_patch16_224.dino`](https://huggingface.co/timm/vit_base_patch16_224.dino)
- [`timm/vit_base_patch8_224.dino`](https://huggingface.co/timm/vit_base_patch8_224.dino)
- [`timm/vit_small_patch16_224.dino`](https://huggingface.co/timm/vit_small_patch16_224.dino)
- [`timm/vit_small_patch8_224.dino`](https://huggingface.co/timm/vit_small_patch8_224.dino)

## DINOv2
[Oquab et al. (2024)](https://arxiv.org/abs/2304.07193), [Darcet et al. (2024)](https://arxiv.org/abs/2309.13568)

DINO scaled on the curated LVD-142M dataset for strong transferable dense features, with `reg4` variants adding register tokens that remove attention-map artifacts.

- [`timm/vit_base_patch14_dinov2.lvd142m`](https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m)
- [`timm/vit_base_patch14_reg4_dinov2.lvd142m`](https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m)
- [`timm/vit_giant_patch14_dinov2.lvd142m`](https://huggingface.co/timm/vit_giant_patch14_dinov2.lvd142m)
- [`timm/vit_giant_patch14_reg4_dinov2.lvd142m`](https://huggingface.co/timm/vit_giant_patch14_reg4_dinov2.lvd142m)
- [`timm/vit_large_patch14_dinov2.lvd142m`](https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m)
- [`timm/vit_large_patch14_reg4_dinov2.lvd142m`](https://huggingface.co/timm/vit_large_patch14_reg4_dinov2.lvd142m)
- [`timm/vit_small_patch14_dinov2.lvd142m`](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m)
- [`timm/vit_small_patch14_reg4_dinov2.lvd142m`](https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m)

## DINOv3
[Siméoni et al. (2025)](https://arxiv.org/abs/2508.10104)

A further-scaled DINO with an updated Transformer architecture and larger curated datasets (LVD-1689M and SAT-493M).

- [`timm/vit_7b_patch16_dinov3.lvd1689m`](https://huggingface.co/timm/vit_7b_patch16_dinov3.lvd1689m)
- [`timm/vit_7b_patch16_dinov3.sat493m`](https://huggingface.co/timm/vit_7b_patch16_dinov3.sat493m)
- [`timm/vit_base_patch16_dinov3.lvd1689m`](https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m)
- [`timm/vit_base_patch16_dinov3_qkvb.lvd1689m`](https://huggingface.co/timm/vit_base_patch16_dinov3_qkvb.lvd1689m)
- [`timm/vit_huge_plus_patch16_dinov3.lvd1689m`](https://huggingface.co/timm/vit_huge_plus_patch16_dinov3.lvd1689m)
- [`timm/vit_huge_plus_patch16_dinov3_qkvb.lvd1689m`](https://huggingface.co/timm/vit_huge_plus_patch16_dinov3_qkvb.lvd1689m)
- [`timm/vit_large_patch16_dinov3.lvd1689m`](https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m)
- [`timm/vit_large_patch16_dinov3.sat493m`](https://huggingface.co/timm/vit_large_patch16_dinov3.sat493m)
- [`timm/vit_large_patch16_dinov3_qkvb.lvd1689m`](https://huggingface.co/timm/vit_large_patch16_dinov3_qkvb.lvd1689m)
- [`timm/vit_large_patch16_dinov3_qkvb.sat493m`](https://huggingface.co/timm/vit_large_patch16_dinov3_qkvb.sat493m)
- [`timm/vit_small_patch16_dinov3.lvd1689m`](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m)
- [`timm/vit_small_patch16_dinov3_qkvb.lvd1689m`](https://huggingface.co/timm/vit_small_patch16_dinov3_qkvb.lvd1689m)
- [`timm/vit_small_plus_patch16_dinov3.lvd1689m`](https://huggingface.co/timm/vit_small_plus_patch16_dinov3.lvd1689m)
- [`timm/vit_small_plus_patch16_dinov3_qkvb.lvd1689m`](https://huggingface.co/timm/vit_small_plus_patch16_dinov3_qkvb.lvd1689m)

## EVA
[Fang et al. (2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_EVA_Exploring_the_Limits_of_Masked_Visual_Representation_Learning_at_CVPR_2023_paper.html)

ViTs pretrained by masked image modeling that reconstructs OpenAI CLIP-L/14 features, included here as ImageNet-fine-tuned checkpoints.

- [`timm/eva_giant_patch14_336.m30m_ft_in22k_in1k`](https://huggingface.co/timm/eva_giant_patch14_336.m30m_ft_in22k_in1k)
- [`timm/eva_giant_patch14_560.m30m_ft_in22k_in1k`](https://huggingface.co/timm/eva_giant_patch14_560.m30m_ft_in22k_in1k)
- [`timm/eva_large_patch14_196.in22k_ft_in1k`](https://huggingface.co/timm/eva_large_patch14_196.in22k_ft_in1k)
- [`timm/eva_large_patch14_196.in22k_ft_in22k_in1k`](https://huggingface.co/timm/eva_large_patch14_196.in22k_ft_in22k_in1k)
- [`timm/eva_large_patch14_336.in22k_ft_in1k`](https://huggingface.co/timm/eva_large_patch14_336.in22k_ft_in1k)
- [`timm/eva_large_patch14_336.in22k_ft_in22k_in1k`](https://huggingface.co/timm/eva_large_patch14_336.in22k_ft_in22k_in1k)

## EVA-02
[Fang et al. (2023)](https://www.sciencedirect.com/science/article/pii/S0262885624002762)

ViTs pretrained by masked image modeling with an EVA-CLIP teacher on ImageNet-22K or merged vision datasets, available as pretrained-only or ImageNet-fine-tuned checkpoints.

- [`timm/eva02_base_patch14_224.mim_in22k`](https://huggingface.co/timm/eva02_base_patch14_224.mim_in22k)
- [`timm/eva02_base_patch14_448.mim_in22k_ft_in1k`](https://huggingface.co/timm/eva02_base_patch14_448.mim_in22k_ft_in1k)
- [`timm/eva02_base_patch14_448.mim_in22k_ft_in22k`](https://huggingface.co/timm/eva02_base_patch14_448.mim_in22k_ft_in22k)
- [`timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k`](https://huggingface.co/timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k)
- [`timm/eva02_large_patch14_224.mim_in22k`](https://huggingface.co/timm/eva02_large_patch14_224.mim_in22k)
- [`timm/eva02_large_patch14_224.mim_m38m`](https://huggingface.co/timm/eva02_large_patch14_224.mim_m38m)
- [`timm/eva02_large_patch14_448.mim_in22k_ft_in1k`](https://huggingface.co/timm/eva02_large_patch14_448.mim_in22k_ft_in1k)
- [`timm/eva02_large_patch14_448.mim_in22k_ft_in22k`](https://huggingface.co/timm/eva02_large_patch14_448.mim_in22k_ft_in22k)
- [`timm/eva02_large_patch14_448.mim_in22k_ft_in22k_in1k`](https://huggingface.co/timm/eva02_large_patch14_448.mim_in22k_ft_in22k_in1k)
- [`timm/eva02_large_patch14_448.mim_m38m_ft_in1k`](https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in1k)
- [`timm/eva02_large_patch14_448.mim_m38m_ft_in22k`](https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k)
- [`timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`](https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k)
- [`timm/eva02_small_patch14_224.mim_in22k`](https://huggingface.co/timm/eva02_small_patch14_224.mim_in22k)
- [`timm/eva02_small_patch14_336.mim_in22k_ft_in1k`](https://huggingface.co/timm/eva02_small_patch14_336.mim_in22k_ft_in1k)
- [`timm/eva02_tiny_patch14_224.mim_in22k`](https://huggingface.co/timm/eva02_tiny_patch14_224.mim_in22k)
- [`timm/eva02_tiny_patch14_336.mim_in22k_ft_in1k`](https://huggingface.co/timm/eva02_tiny_patch14_336.mim_in22k_ft_in1k)

## EVA-CLIP
[Sun et al. (2023)](https://arxiv.org/abs/2303.15389)

CLIP scaled to multi-billion-parameter vision encoders via progressive scaling, trained on LAION-400M (EVA01) or a merged 2B dataset (EVA02) and optionally fine-tuned on ImageNet-1K.

- [`timm/eva02_base_patch16_clip_224.merged2b`](https://huggingface.co/timm/eva02_base_patch16_clip_224.merged2b)
- [`timm/eva02_enormous_patch14_clip_224.laion2b`](https://huggingface.co/timm/eva02_enormous_patch14_clip_224.laion2b)
- [`timm/eva02_enormous_patch14_clip_224.laion2b_plus`](https://huggingface.co/timm/eva02_enormous_patch14_clip_224.laion2b_plus)
- [`timm/eva02_enormous_patch14_clip_224.pretrain`](https://huggingface.co/timm/eva02_enormous_patch14_clip_224.pretrain)
- [`timm/eva02_large_patch14_clip_224.merged2b`](https://huggingface.co/timm/eva02_large_patch14_clip_224.merged2b)
- [`timm/eva02_large_patch14_clip_336.merged2b`](https://huggingface.co/timm/eva02_large_patch14_clip_336.merged2b)
- [`timm/eva_giant_patch14_224.clip_ft_in1k`](https://huggingface.co/timm/eva_giant_patch14_224.clip_ft_in1k)
- [`timm/eva_giant_patch14_336.clip_ft_in1k`](https://huggingface.co/timm/eva_giant_patch14_336.clip_ft_in1k)
- [`timm/eva_giant_patch14_clip_224.laion400m`](https://huggingface.co/timm/eva_giant_patch14_clip_224.laion400m)
- [`timm/eva_giant_patch14_clip_224.merged2b`](https://huggingface.co/timm/eva_giant_patch14_clip_224.merged2b)

## FlexiViT
[Beyer et al. (2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Beyer_FlexiViT_One_Model_for_All_Patch_Sizes_CVPR_2023_paper.html)

A ViT trained with randomly sampled patch sizes so a single checkpoint handles variable patch grids and the full compute–accuracy trade-off, trained on ImageNet-1K or ImageNet-21K.

- [`timm/flexivit_base.1000ep_in21k`](https://huggingface.co/timm/flexivit_base.1000ep_in21k)
- [`timm/flexivit_base.1200ep_in1k`](https://huggingface.co/timm/flexivit_base.1200ep_in1k)
- [`timm/flexivit_base.300ep_in1k`](https://huggingface.co/timm/flexivit_base.300ep_in1k)
- [`timm/flexivit_base.300ep_in21k`](https://huggingface.co/timm/flexivit_base.300ep_in21k)
- [`timm/flexivit_base.600ep_in1k`](https://huggingface.co/timm/flexivit_base.600ep_in1k)
- [`timm/flexivit_base.patch16_in21k`](https://huggingface.co/timm/flexivit_base.patch16_in21k)
- [`timm/flexivit_base.patch30_in21k`](https://huggingface.co/timm/flexivit_base.patch30_in21k)
- [`timm/flexivit_large.1200ep_in1k`](https://huggingface.co/timm/flexivit_large.1200ep_in1k)
- [`timm/flexivit_large.300ep_in1k`](https://huggingface.co/timm/flexivit_large.300ep_in1k)
- [`timm/flexivit_large.600ep_in1k`](https://huggingface.co/timm/flexivit_large.600ep_in1k)
- [`timm/flexivit_small.1200ep_in1k`](https://huggingface.co/timm/flexivit_small.1200ep_in1k)
- [`timm/flexivit_small.300ep_in1k`](https://huggingface.co/timm/flexivit_small.300ep_in1k)
- [`timm/flexivit_small.600ep_in1k`](https://huggingface.co/timm/flexivit_small.600ep_in1k)

## Franca
[Venkataramanan et al. (2026)](https://arxiv.org/abs/2507.14137)

An open-source DINOv2-based self-supervised framework with a Matryoshka multi-head clustering projector; `dinov2` names are baselines and `rasa` applies a position–semantic disentanglement post-training step.

- [`franca/vitb14_dinov2_In21k`](https://github.com/valeoai/Franca)
- [`franca/vitb14_dinov2_In21k_rasa`](https://github.com/valeoai/Franca)
- [`franca/vitb14_In21k`](https://github.com/valeoai/Franca)
- [`franca/vitb14_In21k_rasa`](https://github.com/valeoai/Franca)
- [`franca/vitg14_In21k`](https://github.com/valeoai/Franca)
- [`franca/vitg14_In21k_rasa`](https://github.com/valeoai/Franca)
- [`franca/vitg14_Laion600M`](https://github.com/valeoai/Franca)
- [`franca/vitg14_Laion600M_rasa`](https://github.com/valeoai/Franca)
- [`franca/vitl14_dinov2_In21k`](https://github.com/valeoai/Franca)
- [`franca/vitl14_dinov2_In21k_rasa`](https://github.com/valeoai/Franca)
- [`franca/vitl14_In21k`](https://github.com/valeoai/Franca)
- [`franca/vitl14_In21k_rasa`](https://github.com/valeoai/Franca)
- [`franca/vitl14_Laion600M`](https://github.com/valeoai/Franca)
- [`franca/vitl14_Laion600M_rasa`](https://github.com/valeoai/Franca)

## MAE
[He et al. (2022)](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper)

ViT encoder–decoders pretrained to reconstruct raw pixels from roughly 75% masked image patches, yielding strong transferable representations.

- [`timm/vit_base_patch16_224.mae`](https://huggingface.co/timm/vit_base_patch16_224.mae)
- [`timm/vit_huge_patch14_224.mae`](https://huggingface.co/timm/vit_huge_patch14_224.mae)
- [`timm/vit_large_patch16_224.mae`](https://huggingface.co/timm/vit_large_patch16_224.mae)

## MIIL
[Ridnik et al. (2021)](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/98f13708210194c475687be6106a3b84-Abstract-round1.html)

ViTs from an efficient large-scale ImageNet-21K pretraining pipeline with a semantic-tree class mapping that eases ImageNet-1K fine-tuning.

- [`timm/vit_base_patch16_224_miil.in21k`](https://huggingface.co/timm/vit_base_patch16_224_miil.in21k)
- [`timm/vit_base_patch16_224_miil.in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_224_miil.in21k_ft_in1k)

## Original ViT
[Dosovitskiy et al. (2021)](https://openreview.net/forum?id=YicbFdNTTy)

The original Vision Transformer checkpoints pretrained on ImageNet-21K and optionally fine-tuned on ImageNet-1K.

- [`timm/vit_base_patch16_224.orig_in21k`](https://huggingface.co/timm/vit_base_patch16_224.orig_in21k)
- [`timm/vit_base_patch16_224.orig_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_224.orig_in21k_ft_in1k)
- [`timm/vit_base_patch16_384.orig_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_384.orig_in21k_ft_in1k)
- [`timm/vit_base_patch32_224.orig_in21k`](https://huggingface.co/timm/vit_base_patch32_224.orig_in21k)
- [`timm/vit_huge_patch14_224.orig_in21k`](https://huggingface.co/timm/vit_huge_patch14_224.orig_in21k)
- [`timm/vit_large_patch16_224.orig_in21k`](https://huggingface.co/timm/vit_large_patch16_224.orig_in21k)
- [`timm/vit_large_patch32_224.orig_in21k`](https://huggingface.co/timm/vit_large_patch32_224.orig_in21k)
- [`timm/vit_large_patch32_384.orig_in21k_ft_in1k`](https://huggingface.co/timm/vit_large_patch32_384.orig_in21k_ft_in1k)

## Perception Encoder
[Bolya et al. (2025)](http://arxiv.org/abs/2504.13181)

Meta AI's PE-Core vision backbone trained on a large vision–language mix to produce spatially rich representations for dense perception tasks.

- [`timm/vit_pe_core_base_patch16_224.fb`](https://huggingface.co/timm/vit_pe_core_base_patch16_224.fb)
- [`timm/vit_pe_core_gigantic_patch14_448.fb`](https://huggingface.co/timm/vit_pe_core_gigantic_patch14_448.fb)
- [`timm/vit_pe_core_large_patch14_336.fb`](https://huggingface.co/timm/vit_pe_core_large_patch14_336.fb)
- [`timm/vit_pe_core_small_patch16_384.fb`](https://huggingface.co/timm/vit_pe_core_small_patch16_384.fb)
- [`timm/vit_pe_core_tiny_patch16_384.fb`](https://huggingface.co/timm/vit_pe_core_tiny_patch16_384.fb)

## RoPE ViT
[Heo et al. (2024)](https://link.springer.com/chapter/10.1007/978-3-031-72684-2_17)

ViTs that replace absolute position embeddings with Rotary Position Embeddings (RoPE), improving ImageNet classification and dense prediction.

- [`timm/vit_base_patch16_rope_224.naver_in1k`](https://huggingface.co/timm/vit_base_patch16_rope_224.naver_in1k)
- [`timm/vit_base_patch16_rope_ape_224.naver_in1k`](https://huggingface.co/timm/vit_base_patch16_rope_ape_224.naver_in1k)
- [`timm/vit_base_patch16_rope_mixed_224.naver_in1k`](https://huggingface.co/timm/vit_base_patch16_rope_mixed_224.naver_in1k)
- [`timm/vit_base_patch16_rope_mixed_ape_224.naver_in1k`](https://huggingface.co/timm/vit_base_patch16_rope_mixed_ape_224.naver_in1k)
- [`timm/vit_large_patch16_rope_224.naver_in1k`](https://huggingface.co/timm/vit_large_patch16_rope_224.naver_in1k)
- [`timm/vit_large_patch16_rope_ape_224.naver_in1k`](https://huggingface.co/timm/vit_large_patch16_rope_ape_224.naver_in1k)
- [`timm/vit_large_patch16_rope_mixed_224.naver_in1k`](https://huggingface.co/timm/vit_large_patch16_rope_mixed_224.naver_in1k)
- [`timm/vit_large_patch16_rope_mixed_ape_224.naver_in1k`](https://huggingface.co/timm/vit_large_patch16_rope_mixed_ape_224.naver_in1k)
- [`timm/vit_small_patch16_rope_224.naver_in1k`](https://huggingface.co/timm/vit_small_patch16_rope_224.naver_in1k)
- [`timm/vit_small_patch16_rope_ape_224.naver_in1k`](https://huggingface.co/timm/vit_small_patch16_rope_ape_224.naver_in1k)
- [`timm/vit_small_patch16_rope_mixed_224.naver_in1k`](https://huggingface.co/timm/vit_small_patch16_rope_mixed_224.naver_in1k)
- [`timm/vit_small_patch16_rope_mixed_ape_224.naver_in1k`](https://huggingface.co/timm/vit_small_patch16_rope_mixed_ape_224.naver_in1k)

## SAM
[Chen et al. (2022)](https://openreview.net/forum?id=LtKcMgGOeLt)

ImageNet-1K ViTs trained with Sharpness-Aware Minimization ([Foret et al. 2021](https://arxiv.org/abs/2010.01412)) for improved accuracy and generalization.

- [`timm/vit_base_patch16_224.sam_in1k`](https://huggingface.co/timm/vit_base_patch16_224.sam_in1k)
- [`timm/vit_base_patch32_224.sam_in1k`](https://huggingface.co/timm/vit_base_patch32_224.sam_in1k)

## SigLIP
[Zhai et al. (2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html)

CLIP-style image–text encoders trained with a pairwise sigmoid loss that removes the global normalization step for more efficient training.

- [`timm/vit_base_patch16_siglip_224.webli`](https://huggingface.co/timm/vit_base_patch16_siglip_224.webli)
- [`timm/vit_base_patch16_siglip_256.webli`](https://huggingface.co/timm/vit_base_patch16_siglip_256.webli)
- [`timm/vit_base_patch16_siglip_256.webli_i18n`](https://huggingface.co/timm/vit_base_patch16_siglip_256.webli_i18n)
- [`timm/vit_base_patch16_siglip_384.webli`](https://huggingface.co/timm/vit_base_patch16_siglip_384.webli)
- [`timm/vit_base_patch16_siglip_512.webli`](https://huggingface.co/timm/vit_base_patch16_siglip_512.webli)
- [`timm/vit_large_patch16_siglip_256.webli`](https://huggingface.co/timm/vit_large_patch16_siglip_256.webli)
- [`timm/vit_large_patch16_siglip_384.webli`](https://huggingface.co/timm/vit_large_patch16_siglip_384.webli)
- [`timm/vit_so400m_patch14_siglip_224.webli`](https://huggingface.co/timm/vit_so400m_patch14_siglip_224.webli)
- [`timm/vit_so400m_patch14_siglip_378.webli`](https://huggingface.co/timm/vit_so400m_patch14_siglip_378.webli)
- [`timm/vit_so400m_patch14_siglip_378.webli_ft_in1k`](https://huggingface.co/timm/vit_so400m_patch14_siglip_378.webli_ft_in1k)
- [`timm/vit_so400m_patch14_siglip_384.webli`](https://huggingface.co/timm/vit_so400m_patch14_siglip_384.webli)
- [`timm/vit_so400m_patch16_siglip_256.webli_i18n`](https://huggingface.co/timm/vit_so400m_patch16_siglip_256.webli_i18n)

## SigLIP 2
[Tschannen et al. (2025)](https://arxiv.org/abs/2502.14786)

SigLIP extended with multilingual captions, decoder-based pretraining tasks, and self-supervised objectives for improved spatial understanding and dense features.

- [`timm/vit_base_patch16_siglip_224.v2_webli`](https://huggingface.co/timm/vit_base_patch16_siglip_224.v2_webli)
- [`timm/vit_base_patch16_siglip_256.v2_webli`](https://huggingface.co/timm/vit_base_patch16_siglip_256.v2_webli)
- [`timm/vit_base_patch16_siglip_384.v2_webli`](https://huggingface.co/timm/vit_base_patch16_siglip_384.v2_webli)
- [`timm/vit_base_patch16_siglip_512.v2_webli`](https://huggingface.co/timm/vit_base_patch16_siglip_512.v2_webli)
- [`timm/vit_base_patch32_siglip_256.v2_webli`](https://huggingface.co/timm/vit_base_patch32_siglip_256.v2_webli)
- [`timm/vit_giantopt_patch16_siglip_256.v2_webli`](https://huggingface.co/timm/vit_giantopt_patch16_siglip_256.v2_webli)
- [`timm/vit_giantopt_patch16_siglip_384.v2_webli`](https://huggingface.co/timm/vit_giantopt_patch16_siglip_384.v2_webli)
- [`timm/vit_large_patch16_siglip_256.v2_webli`](https://huggingface.co/timm/vit_large_patch16_siglip_256.v2_webli)
- [`timm/vit_large_patch16_siglip_384.v2_webli`](https://huggingface.co/timm/vit_large_patch16_siglip_384.v2_webli)
- [`timm/vit_large_patch16_siglip_512.v2_webli`](https://huggingface.co/timm/vit_large_patch16_siglip_512.v2_webli)
- [`timm/vit_so400m_patch14_siglip_224.v2_webli`](https://huggingface.co/timm/vit_so400m_patch14_siglip_224.v2_webli)
- [`timm/vit_so400m_patch14_siglip_378.v2_webli`](https://huggingface.co/timm/vit_so400m_patch14_siglip_378.v2_webli)
- [`timm/vit_so400m_patch16_siglip_256.v2_webli`](https://huggingface.co/timm/vit_so400m_patch16_siglip_256.v2_webli)
- [`timm/vit_so400m_patch16_siglip_384.v2_webli`](https://huggingface.co/timm/vit_so400m_patch16_siglip_384.v2_webli)
- [`timm/vit_so400m_patch16_siglip_512.v2_webli`](https://huggingface.co/timm/vit_so400m_patch16_siglip_512.v2_webli)

## Web-SSL
[Fan et al. (2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Scaling_Language-Free_Visual_Representation_Learning_ICCV_2025_paper.html)

DINOv2- and MAE-style self-supervised models scaled to web-scale data (up to 8B images) and up to 7B parameters, showing language-free SSL can match language-supervised models.

- [`webssl/dino1b_full2b_224`](https://huggingface.co/facebook/webssl-dino1b-full2b-224)
- [`webssl/dino2b_full2b_224`](https://huggingface.co/facebook/webssl-dino2b-full2b-224)
- [`webssl/dino2b_heavy2b_224`](https://huggingface.co/facebook/webssl-dino2b-heavy2b-224)
- [`webssl/dino2b_light2b_224`](https://huggingface.co/facebook/webssl-dino2b-light2b-224)
- [`webssl/dino300m_full2b_224`](https://huggingface.co/facebook/webssl-dino300m-full2b-224)
- [`webssl/dino300m_light2b_224`](https://huggingface.co/facebook/webssl-dino300m-light2b-224)
- [`webssl/dino3b_full2b_224`](https://huggingface.co/facebook/webssl-dino3b-full2b-224)
- [`webssl/dino3b_heavy2b_224`](https://huggingface.co/facebook/webssl-dino3b-heavy2b-224)
- [`webssl/dino3b_light2b_224`](https://huggingface.co/facebook/webssl-dino3b-light2b-224)
- [`webssl/dino5b_full2b_224`](https://huggingface.co/facebook/webssl-dino5b-full2b-224)
- [`webssl/dino7b_full8b_224`](https://huggingface.co/facebook/webssl-dino7b-full8b-224)
- [`webssl/dino7b_full8b_378`](https://huggingface.co/facebook/webssl-dino7b-full8b-378)
- [`webssl/dino7b_full8b_518`](https://huggingface.co/facebook/webssl-dino7b-full8b-518)
- [`webssl/mae1b_full2b_224`](https://huggingface.co/facebook/webssl-mae1b-full2b-224)
- [`webssl/mae2b_full2b_224`](https://huggingface.co/facebook/webssl-mae2b-full2b-224)
- [`webssl/mae300m_full2b_224`](https://huggingface.co/facebook/webssl-mae300m-full2b-224)
- [`webssl/mae3b_full2b_224`](https://huggingface.co/facebook/webssl-mae3b-full2b-224)
- [`webssl/mae700m_full2b_224`](https://huggingface.co/facebook/webssl-mae700m-full2b-224)
