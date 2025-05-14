"""
Copy of simba.py but with explicit setting of bfloat16 dtype.
"""

import math
from functools import partial

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

FP16 = torch.float16
BF16 = torch.bfloat16
FP32 = torch.float32

ACT_T = BF16
FFT_ACT_T = FP16
FFT_WEIGHT_T = FP32
MAMBA_MAIN_T = FP32  # Put everything in FP32, to annoying to only put weights in FP32 for now
MAMBA_ACT_T = BF16
PATCH_EMBED_T = FP32


class EinFFT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim  # 768
        self.num_blocks = 4
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        self.complex_weight_1 = nn.Parameter(
            torch.randn(
                2,
                self.num_blocks,
                self.block_size,
                self.block_size,
                dtype=FFT_WEIGHT_T,
            )
            * self.scale
        )
        self.complex_weight_2 = nn.Parameter(
            torch.randn(
                2,
                self.num_blocks,
                self.block_size,
                self.block_size,
                dtype=FFT_WEIGHT_T,
            )
            * self.scale
        )
        self.complex_bias_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=FFT_WEIGHT_T) * self.scale
        )
        self.complex_bias_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=FFT_WEIGHT_T) * self.scale
        )

    def multiply(self, input: torch.Tensor, weights: torch.Tensor):
        weights = weights.to(FFT_ACT_T)
        return torch.einsum("...bd,bdk->...bk", input, weights)

    def get_pad_size(self, size):
        # Get next power of 2``
        return 2 ** math.ceil(math.log2(size))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Torch's fft is only implemented for FP16, not for BF16
        Moreover, FP16 FFTs only accept power-of-2 dimensions for some reason -> Pad the input"""
        B, N, C = x.shape
        x = x.to(FFT_ACT_T)
        x = x.view(B, N, self.num_blocks, self.block_size)

        # Pad the input
        pad_1 = self.get_pad_size(N) - N
        pad_2 = self.get_pad_size(self.num_blocks) - self.num_blocks
        x = F.pad(x, (0, 0, 0, pad_2, 0, pad_1))

        x = torch.fft.fft2(x, dim=(1, 2), norm="ortho")  # FFT on N dimension

        x_real_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[0])
            - self.multiply(x.imag, self.complex_weight_1[1])
            + self.complex_bias_1[0].to(FFT_ACT_T)
        )
        x_imag_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[1])
            + self.multiply(x.imag, self.complex_weight_1[0])
            + self.complex_bias_1[1].to(FFT_ACT_T)
        )
        x_real_2 = (
            self.multiply(x_real_1, self.complex_weight_2[0])
            - self.multiply(x_imag_1, self.complex_weight_2[1])
            + self.complex_bias_2[0].to(FFT_ACT_T)
        )
        x_imag_2 = (
            self.multiply(x_real_1, self.complex_weight_2[1])
            + self.multiply(x_imag_1, self.complex_weight_2[0])
            + self.complex_bias_2[1].to(FFT_ACT_T)
        )

        x = torch.stack([x_real_2, x_imag_2], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold) if self.sparsity_threshold else x
        x = x.to(FFT_ACT_T)

        x = torch.view_as_complex(x)
        x = torch.fft.ifft2(x, dim=(1, 2), norm="ortho")

        # Unpad the output
        x = x[:, :N, : self.num_blocks, :]

        x = x.reshape(B, N, C)
        x = x.to(ACT_T)
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            dtype=MAMBA_MAIN_T,  # NOTE should be enough to put most of Mamba's layer in the correct type
            dtype_act=MAMBA_MAIN_T,
        )

    def forward(self, x):
        # print('x',x.shape)
        B, L, C = x.shape
        x_norm = self.norm(x).to(MAMBA_MAIN_T)
        x_mamba = self.mamba(x_norm)
        return x_mamba


def rand_bbox(size, lam, scale=1):
    W = size[1] // scale
    H = size[2] // scale
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        raise NotImplementedError()
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        raise NotImplementedError()
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, norm_layer=nn.LayerNorm, cm_type="mlp"):
        assert cm_type == "EinFFT"
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MambaLayer(dim)  # MambaBlock(d_model=dim)
        if cm_type == "EinFFT":
            self.mlp = EinFFT(dim)
        else:
            self.mlp = FFN(dim, int(dim * mlp_ratio))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        cls_embed = x[:, :1]
        cls_embed = cls_embed + self.attn(x[:, :1])
        cls_embed = cls_embed + self.mlp(self.norm2(cls_embed), H, W)
        return torch.cat([cls_embed, x[:, 1:]], dim=1)


class Block_mamba(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        cm_type="mlp",
    ):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MambaLayer(dim)  # MambaBlock(d_model=dim)
        if cm_type == "EinFFT":
            self.mlp = EinFFT(dim)
        else:
            self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DownSamples(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            dtype=PATCH_EMBED_T,  # NOTE new
        )
        self.norm = nn.LayerNorm(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class Stem(nn.Module):
    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, hidden_dim, kernel_size=7, stride=2, padding=3, bias=False, dtype=PATCH_EMBED_T
            ),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False, dtype=PATCH_EMBED_T
            ),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False, dtype=PATCH_EMBED_T
            ),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, stride=2, padding=1, dtype=PATCH_EMBED_T)
        self.norm = nn.LayerNorm(out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class SiMBA(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        stem_hidden_dim=32,
        embed_dims=[64, 128, 320, 448],
        mlp_ratios=[8, 8, 4, 4],
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[4, 2, 1, 1],
        num_stages=4,
        token_label=True,
        cm_type="mlp",
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        alpha = 5  #
        for i in range(num_stages):
            if i == 0:
                patch_embed = Stem(in_chans, stem_hidden_dim, embed_dims[i])
            else:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i])

            block = nn.ModuleList(
                [
                    Block_mamba(
                        dim=embed_dims[i],
                        mlp_ratio=mlp_ratios[i],
                        drop_path=dpr[cur + j],
                        norm_layer=norm_layer,
                        sr_ratio=sr_ratios[i],
                        cm_type=cm_type,
                    )
                    for j in range(depths[i])
                ]
            )

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        post_layers = ["ca"]
        self.post_network = nn.ModuleList(
            [
                ClassBlock(
                    dim=embed_dims[-1],
                    mlp_ratio=mlp_ratios[-1],
                    norm_layer=norm_layer,
                    cm_type=cm_type,
                )
                for _ in range(len(post_layers))
            ]
        )

        # classification head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        ##################################### token_label #####################################
        self.return_dense = token_label
        self.mix_token = token_label
        self.beta = 1.0
        self.pooling_scale = 8
        if self.return_dense:
            self.aux_head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        ##################################### token_label #####################################

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_cls(self, x, H, W):
        B, N, C = x.shape
        cls_tokens = x.mean(dim=1, keepdim=True)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.post_network:
            x = block(x, H, W)
        return x

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)

            if i != self.num_stages - 1:
                norm = getattr(self, f"norm{i + 1}")
                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.forward_cls(x, H, W)[:, 0]
        norm = getattr(self, f"norm{self.num_stages}")
        x = norm(x)
        return x

    def forward(self, x):
        if not self.return_dense:
            x = self.forward_features(x)
            x = self.head(x)
            return x
        else:
            x, H, W = self.forward_embeddings(x)
            # mix token, see token labeling for details.
            if self.mix_token and self.training:
                lam = np.random.beta(self.beta, self.beta)
                patch_h, patch_w = (
                    x.shape[1] // self.pooling_scale,
                    x.shape[2] // self.pooling_scale,
                )
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam, scale=self.pooling_scale)
                temp_x = x.clone()
                sbbx1, sbby1, sbbx2, sbby2 = (
                    self.pooling_scale * bbx1,
                    self.pooling_scale * bby1,
                    self.pooling_scale * bbx2,
                    self.pooling_scale * bby2,
                )
                temp_x[:, sbbx1:sbbx2, sbby1:sbby2, :] = x.flip(0)[:, sbbx1:sbbx2, sbby1:sbby2, :]
                x = temp_x
            else:
                bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0
            x = self.forward_tokens(x, H, W)
            x_cls = self.head(x[:, 0])
            x_aux = self.aux_head(x[:, 1:])  # generate classes in all feature tokens, see token labeling

            if not self.training:
                return x_cls + 0.5 * x_aux.max(1)[0]

            if self.mix_token and self.training:  # reverse "mix token", see token labeling for details.
                x_aux = x_aux.reshape(x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])

                temp_x = x_aux.clone()
                temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
                x_aux = temp_x

                x_aux = x_aux.reshape(x_aux.shape[0], patch_h * patch_w, x_aux.shape[-1])

            return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)

    def forward_tokens(self, x, H, W):
        B = x.shape[0]
        x = x.view(B, -1, x.size(-1))

        for i in range(self.num_stages):
            if i != 0:
                patch_embed = getattr(self, f"patch_embed{i + 1}")
                x, H, W = patch_embed(x)
            block = getattr(self, f"block{i + 1}")
            for blk in block:
                x = blk(x, H, W)
            if i != self.num_stages - 1:
                norm = getattr(self, f"norm{i + 1}")
                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.forward_cls(x, H, W)
        norm = getattr(self, f"norm{self.num_stages}")
        x = norm(x)
        return x

    def forward_embeddings(self, x):
        patch_embed = getattr(self, f"patch_embed{0 + 1}")
        x, H, W = patch_embed(x)
        x = x.view(x.size(0), H, W, -1)
        return x, H, W


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim, dtype=ACT_T)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


@register_model
def simba_l_bf16(pretrained=False, **kwargs):
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
