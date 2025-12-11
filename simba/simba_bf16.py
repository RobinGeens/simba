"""
Copy of simba.py but with explicit setting of bfloat16 dtype.
"""

import math
from functools import partial
import os

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


from simba.quantizer import FloatQuantizer, QuantizerPassthrough

USE_OFFICIAL_MAMBA = os.getenv("USE_OFFICIAL_MAMBA", "false").lower() == "true"
if USE_OFFICIAL_MAMBA:
    try:
        from mamba_ssm import Mamba

        print("Using official mamba_ssm")
    except ImportError:
        print("Falling back to local implementation")
        from .mamba_simple import Mamba
else:
    from .mamba_simple import Mamba

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg


FP8 = torch.float8_e5m2
FP16 = torch.float16
BF16 = torch.bfloat16
FP32 = torch.float32


class EinFFT(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.hidden_size = dim  # 384
        self.num_blocks = 4
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        self.ACT_T = kwargs["EINFFT_ACT_T"]
        self.FFT_ACT_T = kwargs["FFT_ACT_T"]
        self.FFT_QUANT = kwargs.get("FFT_QUANT")
        self.EINFFT_QUANT = kwargs.get("EINFFT_QUANT")

        self.complex_weight_1 = nn.Parameter(
            torch.randn(
                2,
                self.num_blocks,
                self.block_size,
                self.block_size,
                dtype=kwargs["EINFFT_WEIGHT_T"],
            )
            * self.scale
        )
        self.complex_weight_2 = nn.Parameter(
            torch.randn(
                2,
                self.num_blocks,
                self.block_size,
                self.block_size,
                dtype=kwargs["EINFFT_WEIGHT_T"],
            )
            * self.scale
        )
        self.complex_bias_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=kwargs["EINFFT_WEIGHT_T"]) * self.scale
        )
        self.complex_bias_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=kwargs["EINFFT_WEIGHT_T"]) * self.scale
        )

        # Quantizer
        self.fft_quantizer = QuantizerPassthrough() if self.FFT_QUANT is None else FloatQuantizer(*self.FFT_QUANT)
        self.einfft_quantizer = (
            QuantizerPassthrough() if self.EINFFT_QUANT is None else FloatQuantizer(*self.EINFFT_QUANT)
        )
        print(f"Quantizing FFT: {self.fft_quantizer}")
        print(f"Quantizing EinFFT: {self.einfft_quantizer}")

    def multiply(self, input: torch.Tensor, weights: torch.Tensor):
        # [NOTE] Quantize here for all EinFFT multiplications
        return torch.einsum(
            "...bd,bdk->...bk",
            self.einfft_quantizer.quantize(input).to(self.ACT_T),
            self.einfft_quantizer.quantize(weights).to(self.ACT_T),
        )

    def get_pad_size(self, size):
        # Get next power of 2``
        return 2 ** math.ceil(math.log2(size))

    def get_dft_matrix_real(self, n: int, inverse: bool = False) -> torch.Tensor:
        """
        Construct the 2Nx2N real-valued DFT matrix for input [Re(x), Im(x)].

        The matrix structure is:
        [Re(W)  -Im(W)] [Re(x)]   [Re(X)]
        [Im(W)   Re(W)] [Im(x)] = [Im(X)]

        where W is the complex DFT matrix W[k,m] = exp(-j 2π k m / N).
        """
        device = next(self.parameters()).device
        k = torch.arange(n).reshape(n, 1)
        m = torch.arange(n).reshape(1, n)
        theta = 2 * torch.pi * k * m / n
        W_real = torch.cos(theta)
        # For forward DFT: Im(W) = -sin(theta); for inverse (conj W): +sin(theta)
        W_imag = (1.0 if inverse else -1.0) * torch.sin(theta)

        # Construct the 2Nx2N real matrix
        W_real_matrix = torch.cat([torch.cat([W_real, -W_imag], dim=1), torch.cat([W_imag, W_real], dim=1)], dim=0)

        return W_real_matrix.to(device=device, dtype=self.FFT_ACT_T)

    def dft(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Compute the DFT using real-valued matrix multiplication on N-D input."""
        n = x.shape[0]
        other_dims = x.shape[1:]  # All dimensions except the first

        W_real = self.get_dft_matrix_real(n, inverse)

        # Reshape to (n, ...) where ... represents all other dimensions flattened
        x_reshaped = x.reshape(n, -1)
        # Create input vector [Re(x), Im(x)] for all slices
        x_real_vec = torch.cat([torch.real(x_reshaped), torch.imag(x_reshaped)], dim=0)
        x_real_vec = x_real_vec.to(device=W_real.device, dtype=self.FFT_ACT_T)

        # [NOTE] quantize here
        X_real_vec = torch.matmul(self.fft_quantizer.quantize(W_real), self.fft_quantizer.quantize(x_real_vec)).to(
            self.FFT_ACT_T
        )

        # Split result back into real and imaginary parts
        X_real = X_real_vec[:n, :]
        X_imag = X_real_vec[n:, :]

        # Combine into complex array and reshape back to original shape
        X_complex = torch.view_as_complex(torch.stack([X_real, X_imag], dim=-1))
        result = X_complex.reshape(n, *other_dims)

        # Apply orthogonal normalization
        return result / torch.sqrt(torch.tensor(n, dtype=result.dtype, device=result.device))

    def dft_partitioned(self, x: torch.Tensor, L: int, inverse: bool = False) -> torch.Tensor:
        """
        Compute DFT using matrix partitioning strategy (based on the paper).
        Supports N = M * L factorization, reducing matrix multiply size.

        Args:
            x: complex tensor (1D or ND, transform along axis=0)
            inverse: compute inverse DFT if True
            L: partition length (divides N)

        Returns:
            complex tensor, same shape as input
        """
        n = x.shape[0]
        other_dims = x.shape[1:]
        assert n % L == 0, f"N must be divisible by L, got N={n}, L={L}"
        M = n // L

        # === Step 1: Reshape into MxL ===
        x = x.reshape(M, L, *other_dims)

        # === Step 2: Sub-DFTs across each column (size M) ===
        W_M_real = self.get_dft_matrix_real(M, inverse=inverse)
        x_real = torch.cat([torch.real(x), torch.imag(x)], dim=0)
        x_real = x_real.reshape(2 * M, L, -1)
        x_real = x_real.to(device=W_M_real.device, dtype=self.FFT_ACT_T)

        # [NOTE] quantize here
        x_real_reshaped = x_real.reshape(2 * M, -1)  # (2M, L * num_features)
        X1_real_flat = torch.matmul(
            self.fft_quantizer.quantize(W_M_real), self.fft_quantizer.quantize(x_real_reshaped)
        ).to(self.FFT_ACT_T)
        X1_real = X1_real_flat.reshape(2 * M, L, *other_dims)

        # === Step 3: Apply Hadamard phase correction ===
        device = x.device
        k = torch.arange(M, device=device).reshape(M, 1)
        l = torch.arange(L, device=device).reshape(1, L)
        # Twiddle factor: forward uses e^{-j2πkl/N}, inverse uses e^{+j2πkl/N}
        phase_angle = ((1.0) if inverse else (-1.0)) * 2 * torch.pi * k * l / (M * L)
        phase = torch.complex(torch.cos(phase_angle), torch.sin(phase_angle))
        X1_complex = torch.view_as_complex(torch.stack([X1_real[:M], X1_real[M:]], dim=-1))
        # Broadcast phase across all remaining dimensions (other_dims)
        phase = phase.reshape((M, L) + (1,) * (X1_complex.ndim - 2))
        X2 = X1_complex * phase  # Hadamard product

        # === Step 4: Row-wise DFT (size L) ===
        W_L_real = self.get_dft_matrix_real(L, inverse=inverse)
        # Make L the leading axis for the size-L DFT, then stack [Re; Im] along axis 0
        perm_dims = [1, 0] + list(range(2, X2.ndim))  # [1, 0, 2, 3, ...]
        X2_L_major = X2.permute(*perm_dims)  # (L, M, ...)
        X2r = torch.cat([torch.real(X2_L_major), torch.imag(X2_L_major)], dim=0)  # (2L, M, ...)
        X2r = X2r.reshape(2 * L, M, -1)
        X2r = X2r.to(device=W_L_real.device, dtype=self.FFT_ACT_T)

        # [NOTE] quantize here - handle tensor contraction like tensordot(W_L_real, X2r, axes=(1, 0))
        # This means multiply W_L_real[i,j] with X2r[j, k, ...] to get result[i, k, ...]
        X2r_reshaped = X2r.reshape(2 * L, -1)  # (2L, M * num_features)
        Xf_real_flat = torch.matmul(
            self.fft_quantizer.quantize(W_L_real), self.fft_quantizer.quantize(X2r_reshaped)
        ).to(self.FFT_ACT_T)
        Xf_real = Xf_real_flat.reshape(2 * L, M, *other_dims)

        # === Step 5: Recombine ===
        Xf = torch.view_as_complex(torch.stack([Xf_real[:L], Xf_real[L:]], dim=-1))  # shape (L, M, ...)
        # Flatten with k = k1 + M*k2 ordering by keeping (L, M, ...) then reshape
        Xf = Xf.reshape(n, *other_dims)

        # === Step 6: Normalization ===
        scale = torch.sqrt(torch.tensor(n, dtype=Xf.dtype, device=Xf.device))
        return Xf / scale

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Torch's fft is only implemented for FP16, not for BF16
        Moreover, FP16 FFTs only accept power-of-2 dimensions for some reason -> Pad the input"""
        DFT_PARTITIONS = {
            3136: 56,
            784: 28,
            196: 14,
            49: 7,
            4: 1,
        }
        B, N, C = x.shape
        x = x.to(self.FFT_ACT_T)
        x = x.view(B, N, self.num_blocks, self.block_size)

        # Pad the input # TODO omitting this results in a 0.6% accuracy drop because weights were trained at power-of-2 sizes
        # pad_1 = self.get_pad_size(N) - N
        # pad_2 = self.get_pad_size(self.num_blocks) - self.num_blocks
        # x = F.pad(x, (0, 0, 0, pad_2, 0, pad_1))
        # x = torch.fft.fft2(x, dim=(1, 2), norm="ortho")  # FFT on N dimension

        x = torch.view_as_complex(torch.stack([x, torch.zeros_like(x)], dim=-1))
        # Apply FFT to dimension 1 (N dimension)
        x = self.dft_partitioned(x.transpose(1, 0), L=DFT_PARTITIONS.get(N, 1)).transpose(1, 0)
        # Apply FFT to dimension 2 (num_blocks dimension)
        x = self.dft_partitioned(x.transpose(2, 0), L=DFT_PARTITIONS.get(self.num_blocks, 1)).transpose(2, 0)

        x_real_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[0])
            - self.multiply(x.imag, self.complex_weight_1[1])
            + self.complex_bias_1[0].to(self.ACT_T)
        )
        x_imag_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[1])
            + self.multiply(x.imag, self.complex_weight_1[0])
            + self.complex_bias_1[1].to(self.ACT_T)
        )
        x_real_2 = (
            self.multiply(x_real_1, self.complex_weight_2[0])
            - self.multiply(x_imag_1, self.complex_weight_2[1])
            + self.complex_bias_2[0].to(self.ACT_T)
        )
        x_imag_2 = (
            self.multiply(x_real_1, self.complex_weight_2[1])
            + self.multiply(x_imag_1, self.complex_weight_2[0])
            + self.complex_bias_2[1].to(self.ACT_T)
        )

        x = torch.stack([x_real_2, x_imag_2], dim=-1)
        x = F.softshrink(x.to(FP32), lambd=self.sparsity_threshold) if self.sparsity_threshold else x

        x = x.to(self.FFT_ACT_T)
        x = torch.view_as_complex(x)

        # x = torch.fft.ifft2(x, dim=(1, 2), norm="ortho")

        # Apply IDFT to dimension 2 (num_blocks dimension)
        x = self.dft_partitioned(x.transpose(2, 0), L=DFT_PARTITIONS.get(self.num_blocks, 1), inverse=True).transpose(
            2, 0
        )
        # Apply IDFT to dimension 1 (N dimension)
        x = self.dft_partitioned(x.transpose(1, 0), L=DFT_PARTITIONS.get(N, 1), inverse=True).transpose(1, 0)

        # Unpad the output
        # x = x[:, :N, : self.num_blocks, :]

        x = x.reshape(B, N, C)
        return x.to(self.ACT_T)


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2, **kwargs):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim, dtype=FP32)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dtype=kwargs["MAMBA_MAIN_T"],  # NOTE should be enough to put most of Mamba's layer in the correct type
            dtype_act=kwargs["MAMBA_ACT_T"],
            quant=kwargs["MAMBA_QUANT"],
        )

    def forward(self, x):
        # print('x',x.shape)
        B, L, C = x.shape
        x_norm = self.norm(x)
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


class ClassBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.norm2 = norm_layer(dim)
        self.attn = MambaLayer(dim, **kwargs)
        self.mlp = EinFFT(dim, **kwargs)

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
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    ):
        super().__init__()
        self.norm2 = norm_layer(dim, dtype=FP32)
        self.attn = MambaLayer(dim, **kwargs)
        self.mlp = EinFFT(dim, **kwargs)
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
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            dtype=kwargs["PATCH_EMBED_T"],
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
    def __init__(self, in_channels, stem_hidden_dim, out_channels, **kwargs):
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, hidden_dim, kernel_size=7, stride=2, padding=3, bias=False, dtype=kwargs["PATCH_EMBED_T"]
            ),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False, dtype=kwargs["PATCH_EMBED_T"]
            ),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False, dtype=kwargs["PATCH_EMBED_T"]
            ),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(
            hidden_dim, out_channels, kernel_size=3, stride=2, padding=1, dtype=kwargs["PATCH_EMBED_T"]
        )
        self.norm = nn.LayerNorm(out_channels, dtype=FP32)

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
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 6, 3],
        num_stages=4,
        token_label=True,
        cm_type="mlp",
        **kwargs: torch.dtype,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.AUTOCAST_T: torch.dtype = kwargs["AUTOCAST_T"]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(num_stages):
            if i == 0:
                patch_embed = Stem(in_chans, stem_hidden_dim, embed_dims[i], **kwargs)
            else:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i], **kwargs)

            block = nn.ModuleList(
                [
                    Block_mamba(
                        dim=embed_dims[i],
                        drop_path=dpr[cur + j],
                        norm_layer=norm_layer,
                        **kwargs,
                    )
                    for j in range(depths[i])
                ]
            )

            norm = norm_layer(embed_dims[i], dtype=FP32)
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        post_layers = ["ca"]
        self.post_network = nn.ModuleList(
            [
                ClassBlock(
                    dim=embed_dims[-1], mlp_ratio=mlp_ratios[-1], norm_layer=norm_layer, cm_type=cm_type, **kwargs
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


@register_model
def simba_l_bf16(pretrained=False, **kwargs):
    """Test with BF16. This is our main model."""
    kwargs = {
        **kwargs,
        # TODO currently, matmul inputs are provided as follows: `quant(in).to(dType)`
        # TODO What is the effect of omitting the `to(dType)` ?
        "FFT_ACT_T": FP16,  # BF16 not supported
        "FFT_QUANT": (3, 2),
        "EINFFT_ACT_T": BF16,
        "EINFFT_WEIGHT_T": FP32,  # Weights before casting
        "EINFFT_QUANT": (5, 2),
        "MAMBA_MAIN_T": FP32,  # Weights before casting, non-linear functions
        "MAMBA_ACT_T": BF16,  # Linear projections, state-update, etc
        "MAMBA_QUANT": (5, 2),
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
def simba_l_fp16(pretrained=False, **kwargs):
    """Test with FP16, only used as an experiment"""
    kwargs = {
        **kwargs,
        "FFT_ACT_T": FP16,
        "EINFFT_ACT_T": FP16,
        "EINFFT_WEIGHT_T": FP16,  # Weights before casting
        "MAMBA_MAIN_T": FP16,  # Weights before casting, non-linear functions
        "MAMBA_ACT_T": FP16,  # Linear projections, state-update, etc
        "PATCH_EMBED_T": FP16,
        "NORM_T": FP16,
        "AUTOCAST_T": FP16,
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
