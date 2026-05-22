from functools import partial

import torch.nn as nn
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from simba_bf16 import BF16, FP32, SiMBA


@register_model
def simba_l_bf16(pretrained=False, **kwargs):
    kwargs = {
        **kwargs,
        "FFT_ACT_T": BF16,  # BF16 not supported for true FFT
        "USE_DFT": True,
        "EINFFT_ACT_T": BF16,
        "EINFFT_WEIGHT_T": FP32,  # Weights before casting
        "MAMBA_MAIN_T": FP32,  # Weights before casting, non-linear functions
        "MAMBA_ACT_T": BF16,  # Linear projections, state-update, etc
        "MAMBA_USE_HARDWARE_ACT": False,
        "PATCH_EMBED_T": FP32,
        "NORM_T": FP32,
        "AUTOCAST_T": BF16,
    }

    model = SiMBA(
        stem_hidden_dim=64,
        embed_dims=[96, 192, 384, 512],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 6, 18, 3],
        sr_ratios=[4, 2, 1, 1],
        cm_type="EinFFT",
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def simba_l_fp8(pretrained=False, **kwargs):
    kwargs = {
        **kwargs,
        "FFT_ACT_T": BF16,
        "USE_DFT": True,
        "FFT_QUANT": (5, 2),
        "EINFFT_ACT_T": BF16,
        "EINFFT_WEIGHT_T": FP32,  # Weights before casting
        "EINFFT_QUANT": (5, 2),
        "MAMBA_MAIN_T": FP32,  # Weights before casting, non-linear functions
        "MAMBA_ACT_T": BF16,  # Linear projections, state-update, etc
        "MAMBA_QUANT": (5, 2),
        "MAMBA_USE_HARDWARE_ACT": False,
        "PATCH_EMBED_T": FP32,
        "NORM_T": FP32,
        "AUTOCAST_T": BF16,
    }

    model = SiMBA(
        stem_hidden_dim=64,
        embed_dims=[96, 192, 384, 512],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 6, 18, 3],
        sr_ratios=[4, 2, 1, 1],
        cm_type="EinFFT",
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def simba_b_bf16(pretrained=False, **kwargs):
    kwargs = {
        **kwargs,
        "FFT_ACT_T": BF16,  # BF16 not supported for true FFT
        "USE_DFT": True,
        # "FFT_QUANT": (5, 2),
        "EINFFT_ACT_T": BF16,
        "EINFFT_WEIGHT_T": FP32,  # Weights before casting
        # "EINFFT_QUANT": (5, 2),
        "MAMBA_MAIN_T": FP32,  # Weights before casting, non-linear functions
        "MAMBA_ACT_T": BF16,  # Linear projections, state-update, etc
        # "MAMBA_QUANT": (5, 2),
        "MAMBA_USE_HARDWARE_ACT": False,
        "PATCH_EMBED_T": FP32,
        "NORM_T": FP32,
        "AUTOCAST_T": BF16,
    }

    # Simba-B (23M params)
    model = SiMBA(
        stem_hidden_dim=64,
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 12, 3],
        sr_ratios=[4, 2, 1, 1],
        cm_type="mlp",
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def simba_cityscapes(pretrained=False, **kwargs):
    kwargs = {
        **kwargs,
        "FFT_ACT_T": BF16,
        "USE_DFT": True,
        # "FFT_QUANT": (3, 2),
        "EINFFT_ACT_T": BF16,
        "EINFFT_WEIGHT_T": FP32,  # Weights before casting
        # "EINFFT_QUANT": (5, 2),
        "MAMBA_MAIN_T": FP32,  # Weights before casting, non-linear functions
        "MAMBA_ACT_T": BF16,  # Linear projections, state-update, etc
        # "MAMBA_QUANT": (5, 2),
        "MAMBA_USE_HARDWARE_ACT": False,
        "PATCH_EMBED_T": FP32,
        "NORM_T": FP32,
        "AUTOCAST_T": BF16,
    }

    model = SiMBA(
        stem_hidden_dim=64,
        embed_dims=[96, 192, 384, 512],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 6, 18, 3],
        sr_ratios=[4, 2, 1, 1],
        cm_type="EinFFT",
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model
