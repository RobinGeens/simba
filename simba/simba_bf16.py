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

from quantizer import QuantizerPassthrough, FloatQuantizer
from mamba_simple import Mamba

from timm.models.layers import DropPath, trunc_normal_


FP8 = torch.float8_e5m2
FP16 = torch.float16
BF16 = torch.bfloat16
FP32 = torch.float32


def make_norm_layer(norm_type: str = "layernorm", eps: float = 1e-6):
    """Return a partial that constructs the requested norm layer.

    "layernorm" -> nn.LayerNorm (mean-centered, has bias)
    "rmsnorm"   -> nn.RMSNorm (no mean-centering, no bias)
    """
    norm_type = norm_type.lower()
    if norm_type == "layernorm":
        return partial(nn.LayerNorm, eps=eps)
    if norm_type == "rmsnorm":
        return partial(nn.RMSNorm, eps=eps)
    raise ValueError(f"Unknown norm_type: {norm_type!r} (expected 'layernorm' or 'rmsnorm')")


def _resolve_norm_layer(kwargs):
    """Pick the norm-layer factory from kwargs, defaulting to LayerNorm."""
    return kwargs.get("NORM_LAYER", partial(nn.LayerNorm, eps=1e-6))


# Cooley-Tukey partition length L (N = M * L) used by dft_partitioned, keyed by the
# (power-of-two) transform length. Missing keys fall back to L=1 (single matmul DFT).
_DFT_PARTITIONS = {
    # Unpadded
    3136: 56,
    784: 28,
    196: 14,
    49: 7,
    4: 1,
    # Padded
    4096: 64,
    1024: 32,
    256: 16,
    64: 8,
    # Cityscapes
    512: 32,
    2048: 64,
    8192: 128,
    16384: 128,
    32768: 256,
    65536: 256,
    131072: 512,
}


class EinFFT(nn.Module):
    _quantization_logged = False

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
        self.USE_DFT = kwargs.get("USE_DFT", False)
        # Video: number of frames flattened (spatial-first) into the token axis.
        # The FFT runs per frame, never across frames. Default 1 (single image).
        self.FFT_BATCH = kwargs.get("FFT_BATCH") or 1
        assert (
            isinstance(self.FFT_BATCH, int) and self.FFT_BATCH >= 1
        ), f"FFT_BATCH must be a positive int, got {self.FFT_BATCH!r}"
        # FFT_WINDOW_SIZE is a (win_seq, win_lines) tuple.
        self.FFT_WINDOW_SIZE = kwargs.get("FFT_WINDOW_SIZE")
        if self.FFT_WINDOW_SIZE is not None:
            if isinstance(self.FFT_WINDOW_SIZE, int):
                # Back-compat: a bare int is a 1D window over the token sequence.
                self.FFT_WINDOW_SIZE = (self.FFT_WINDOW_SIZE, 1)
            self.FFT_WINDOW_SIZE = tuple(self.FFT_WINDOW_SIZE)
            assert len(self.FFT_WINDOW_SIZE) == 2, (
                f"FFT_WINDOW_SIZE must be a (win_seq, win_lines) tuple, " f"got {self.FFT_WINDOW_SIZE}"
            )
            for w in self.FFT_WINDOW_SIZE:
                assert w >= 1 and w & (w - 1) == 0, (
                    f"FFT_WINDOW_SIZE entries must be powers of 2, " f"got {self.FFT_WINDOW_SIZE}"
                )
        self.FFT_WINDOW_THRESHOLD = kwargs.get("FFT_WINDOW_THRESHOLD") or (
            self.FFT_WINDOW_SIZE[0] * self.FFT_WINDOW_SIZE[1] if self.FFT_WINDOW_SIZE is not None else None
        )

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
        if not EinFFT._quantization_logged:
            print("[Init][EinFFT]")
            print(f"  - fft quantizer: {self.fft_quantizer}")
            print(f"  - einfft quantizer: {self.einfft_quantizer}")
            print(f"  - use dft: {self.USE_DFT}")
            print(f"  - fft window size: {self.FFT_WINDOW_SIZE}")
            print(f"  - fft window threshold: {self.FFT_WINDOW_THRESHOLD}")
            print(f"  - fft batch (frames): {self.FFT_BATCH}")
            EinFFT._quantization_logged = True

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

    def window_size_for(self, n: int) -> "tuple[int, int] | None":
        """Window shape (win_seq, win_lines) to use for a token-axis of length
        ``n``, or None for a full-sequence DFT"""
        win = self.FFT_WINDOW_SIZE
        threshold = self.FFT_WINDOW_THRESHOLD
        if win is not None and threshold is not None:
            win_len = win[0] * win[1]
            if n > threshold and n > win_len:
                return win
        return None

    def get_dft_matrix_real(self, n: int, inverse: bool = False) -> torch.Tensor:
        """
        Construct the 2Nx2N real-valued DFT matrix for input [Re(x), Im(x)].

        The matrix structure is:
        [Re(W)  -Im(W)] [Re(x)]   [Re(X)]
        [Im(W)   Re(W)] [Im(x)] = [Im(X)]

        where W is the complex DFT matrix W[k,m] = exp(-j 2π k m / N), pre-scaled
        by 1/sqrt(n) so the matmul output is already ortho-normalized. This keeps
        the accumulated dot-product magnitudes bounded for large n, preventing overflow.
        """
        device = next(self.parameters()).device
        k = torch.arange(n).reshape(n, 1)
        m = torch.arange(n).reshape(1, n)
        theta = 2 * torch.pi * k * m / n
        inv_sqrt_n = 1.0 / math.sqrt(n)
        W_real = torch.cos(theta) * inv_sqrt_n
        # For forward DFT: Im(W) = -sin(theta); for inverse (conj W): +sin(theta)
        W_imag = (1.0 if inverse else -1.0) * torch.sin(theta) * inv_sqrt_n

        # Construct the 2Nx2N real matrix
        W_real_matrix = torch.cat(
            [torch.cat([W_real, -W_imag], dim=1), torch.cat([W_imag, W_real], dim=1)],
            dim=0,
        )

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

        # Orthogonal normalization (1/sqrt(n)) is already baked into W_real.
        return result

    def dft_partitioned(
        self, x_re: torch.Tensor, x_im: torch.Tensor, L: int, inverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute DFT using matrix partitioning strategy (based on the paper).
        Supports N = M * L factorization, reducing matrix multiply size.

        Real and imaginary parts are passed/returned as separate real tensors so
        the path is BF16-compatible (torch complex types don't support bfloat16).

        Args:
            x_re, x_im: real and imaginary parts (transform along axis=0)
            L: partition length (divides N)
            inverse: compute inverse DFT if True

        Returns:
            (Xf_re, Xf_im) tuple with the same shape as the inputs.
        """
        N = x_re.shape[0]
        other_dims = x_re.shape[1:]
        assert N % L == 0, f"N must be divisible by L, got N={N}, L={L}"
        M = N // L

        # === Step 1: Reshape into MxL ===
        x_re = x_re.reshape(M, L, *other_dims)
        x_im = x_im.reshape(M, L, *other_dims)

        # === Step 2: Sub-DFTs across each column (size M) ===
        W_M_real = self.get_dft_matrix_real(M, inverse=inverse)
        x_real = torch.cat([x_re, x_im], dim=0)  # (2M, L, ...)
        x_real = x_real.to(device=W_M_real.device, dtype=self.FFT_ACT_T)

        # [NOTE] quantize here
        x_real_reshaped = x_real.reshape(2 * M, -1)  # (2M, L * num_features)
        X1_real_flat = torch.matmul(
            self.fft_quantizer.quantize(W_M_real),
            self.fft_quantizer.quantize(x_real_reshaped),
        ).to(self.FFT_ACT_T)
        X1_real = X1_real_flat.reshape(2 * M, L, *other_dims)
        X1_re = X1_real[:M]
        X1_im = X1_real[M:]

        # === Step 3: Apply Hadamard phase correction ===
        device = x_re.device
        k = torch.arange(M, device=device).reshape(M, 1)
        l = torch.arange(L, device=device).reshape(1, L)
        # Twiddle factor: forward uses e^{-j2πkl/N}, inverse uses e^{+j2πkl/N}
        phase_angle = ((1.0) if inverse else (-1.0)) * 2 * torch.pi * k * l / (M * L)
        extra_dims = (1,) * (X1_re.ndim - 2)
        phase_re = torch.cos(phase_angle).to(dtype=self.FFT_ACT_T).reshape((M, L) + extra_dims)
        phase_im = torch.sin(phase_angle).to(dtype=self.FFT_ACT_T).reshape((M, L) + extra_dims)
        # Complex multiply: (a+bj)*(c+dj) = (ac-bd) + (ad+bc)j
        X2_re = X1_re * phase_re - X1_im * phase_im
        X2_im = X1_re * phase_im + X1_im * phase_re

        # === Step 4: Row-wise DFT (size L) ===
        W_L_real = self.get_dft_matrix_real(L, inverse=inverse)
        # Make L the leading axis for the size-L DFT, then stack [Re; Im] along axis 0
        perm_dims = [1, 0] + list(range(2, X2_re.ndim))  # [1, 0, 2, 3, ...]
        X2_re = X2_re.permute(*perm_dims)  # (L, M, ...)
        X2_im = X2_im.permute(*perm_dims)
        X2r = torch.cat([X2_re, X2_im], dim=0)  # (2L, M, ...)
        X2r = X2r.to(device=W_L_real.device, dtype=self.FFT_ACT_T)

        # [NOTE] quantize here - handle tensor contraction like tensordot(W_L_real, X2r, axes=(1, 0))
        # This means multiply W_L_real[i,j] with X2r[j, k, ...] to get result[i, k, ...]
        X2r_reshaped = X2r.reshape(2 * L, -1)  # (2L, M * num_features)
        Xf_real_flat = torch.matmul(
            self.fft_quantizer.quantize(W_L_real),
            self.fft_quantizer.quantize(X2r_reshaped),
        ).to(self.FFT_ACT_T)
        Xf_real = Xf_real_flat.reshape(2 * L, M, *other_dims)

        # === Step 5: Recombine (k = k1 + M*k2 ordering preserved by (L, M, ...) layout) ===
        Xf_re = Xf_real[:L].reshape(N, *other_dims)
        Xf_im = Xf_real[L:].reshape(N, *other_dims)

        # Ortho normalization: W_M contributes 1/sqrt(M), W_L contributes 1/sqrt(L),
        # so the combined output already carries the 1/sqrt(M*L) = 1/sqrt(n) factor.
        return Xf_re, Xf_im

    def dft_router(
        self,
        x_re: torch.Tensor,
        x_im: torch.Tensor,
        axis: int,
        inverse: bool = False,
        window: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(Inverse) DFT of one ``axis``, dispatching to dft_partitioned (USE_DFT)
        or torch.fft, with optional windowing.

        When ``window`` is set and the axis is longer than FFT_WINDOW_THRESHOLD, the
        axis is split into contiguous windows of win_seq * win_lines tokens and each
        window is transformed independently. The windows are folded into the batch
        dim, the transform runs, then they are unfolded back here. The axis is
        assumed already prepared by forward() (asserted to be a whole number of
        windows, and reordered window-major for 2D windows so that each window's
        tokens are spatially near each other); the learnable filter has no
        N-frequency dependence, so the same weights apply at any window size.
        """
        B_orig = x_re.shape[0]
        win = self.window_size_for(x_re.shape[axis]) if window else None

        if win is not None:
            assert axis == 1, "windowing is only defined for the token (N) axis"
            win_len = win[0] * win[1]
            # Fold windows of `win_len` consecutive tokens into the batch dim:
            # (B, N, G, D) -> (B * N/win_len, win_len, G, D)
            x_re = x_re.reshape(B_orig * (x_re.shape[1] // win_len), win_len, *x_re.shape[2:])
            x_im = x_im.reshape(B_orig * (x_im.shape[1] // win_len), win_len, *x_im.shape[2:])

        if self.USE_DFT:
            n = x_re.shape[axis]
            x_re, x_im = self.dft_partitioned(
                x_re.transpose(axis, 0),
                x_im.transpose(axis, 0),
                L=_DFT_PARTITIONS.get(n, 1),
                inverse=inverse,
            )
            x_re = x_re.transpose(axis, 0)
            x_im = x_im.transpose(axis, 0)
        else:
            # torch.fft has no BF16/complex-BF16 support -> compute in float32.
            orig_dtype = x_re.dtype
            x_complex = torch.complex(x_re.to(FP32), x_im.to(FP32))
            fft_fn = torch.fft.ifft if inverse else torch.fft.fft
            x_complex = fft_fn(x_complex, dim=axis, norm="ortho")
            x_re = x_complex.real.to(orig_dtype)
            x_im = x_complex.imag.to(orig_dtype)

        if win is not None:
            # Unfold the windows back into the axis: (B*N/win, win, G, D) -> (B, N, G, D)
            x_re = x_re.reshape(B_orig, -1, *x_re.shape[2:])
            x_im = x_im.reshape(B_orig, -1, *x_im.shape[2:])
        return x_re, x_im

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Torch's fft is only implemented for FP16, not for BF16
        Moreover, FP16 FFTs only accept power-of-2 dimensions for some reason -> Pad the input
        """
        B_in, N_in, C = x.shape
        # Video: the token axis holds FFT_BATCH frames, flattened spatial-first.
        # Fold the frames into the batch dim so every transform below stays
        # within one frame (no FFT over time); unfolded again at the end.
        if self.FFT_BATCH > 1 and N_in > 1:
            assert N_in % self.FFT_BATCH == 0, (
                f"FFT_BATCH={self.FFT_BATCH} must divide the token axis " f"exactly, got N={N_in}"
            )
            x = x.reshape(B_in * self.FFT_BATCH, N_in // self.FFT_BATCH, C)
        B, N, C = x.shape
        x = x.to(self.FFT_ACT_T)
        x = x.view(B, N, self.num_blocks, self.block_size)

        # Windows must tile the raster exactly.
        # Only the unwindowed full-sequence DFT still pads the token axis to the next
        # power of two; that pad is cropped back at the end of the round trip.
        win = self.window_size_for(N)
        pad_2 = self.get_pad_size(self.num_blocks) - self.num_blocks
        G_pad = self.num_blocks + pad_2
        spatial = win is not None and win[1] > 1
        if spatial:
            # 2D window: view the tokens as the (H, W) raster and reorder
            # window-major so every win_lines x win_seq spatial block occupies
            # win_len consecutive tokens. Undone after the inverse transform below.
            win_seq, win_lines = win
            assert N == H * W, f"2D FFT windows need the raster token axis, got N={N}, HxW={H}x{W}"
            assert H % win_lines == 0 and W % win_seq == 0, (
                f"FFT_WINDOW_SIZE (win_seq, win_lines)={win} must tile the " f"raster exactly, got HxW={H}x{W}"
            )
            x = x.view(B, H, W, self.num_blocks, self.block_size)
            x = F.pad(x, (0, 0, 0, pad_2))
            x = x.view(B, H // win_lines, win_lines, W // win_seq, win_seq, G_pad, self.block_size)
            x = x.permute(0, 1, 3, 2, 4, 5, 6).reshape(B, N, G_pad, self.block_size)
        else:
            if win is not None:
                # 1D window (win_lines == 1): must divide the token axis exactly.
                win_len = win[0] * win[1]
                assert N % win_len == 0, (
                    f"FFT_WINDOW_SIZE {win} (= {win_len} tokens) must divide the " f"token axis exactly, got N={N}"
                )
                pad_1 = 0
            else:
                pad_1 = self.get_pad_size(N) - N
            x = F.pad(x, (0, 0, 0, pad_2, 0, pad_1))

        # Carry real/imag as separate real tensors because torch complex types don't support BF16.
        x_re = x
        x_im = torch.zeros_like(x)
        # Forward transform: token (N) axis, windowed, then num_blocks axis.
        x_re, x_im = self.dft_router(x_re, x_im, axis=1, window=True)
        x_re, x_im = self.dft_router(x_re, x_im, axis=2, window=False)

        assert torch.isfinite(x_re).all() and torch.isfinite(x_im).all()

        x_real_1 = F.relu(
            self.multiply(x_re, self.complex_weight_1[0])
            - self.multiply(x_im, self.complex_weight_1[1])
            + self.complex_bias_1[0].to(self.ACT_T)
        )
        x_imag_1 = F.relu(
            self.multiply(x_re, self.complex_weight_1[1])
            + self.multiply(x_im, self.complex_weight_1[0])
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

        if self.sparsity_threshold:
            x_stacked = torch.stack([x_real_2, x_imag_2], dim=-1)
            x_stacked = F.softshrink(x_stacked.to(FP32), lambd=self.sparsity_threshold)
            x_real_2 = x_stacked[..., 0]
            x_imag_2 = x_stacked[..., 1]

        x_re = x_real_2.to(self.FFT_ACT_T)
        x_im = x_imag_2.to(self.FFT_ACT_T)

        # Inverse transform: num_blocks axis then token (N) axis, windowed.
        x_re, x_im = self.dft_router(x_re, x_im, axis=2, window=False, inverse=True)
        x_re, x_im = self.dft_router(x_re, x_im, axis=1, window=True, inverse=True)

        assert torch.isfinite(x_re).all()

        # Input was real -> output of IFFT is (numerically) real; keep the real part only.
        # Drop the token-axis padding and the num_blocks padding, then restore (B, N, C).
        if spatial:
            # Undo the window-major reorder.
            x_re = x_re.view(B, H // win_lines, W // win_seq, win_lines, win_seq, G_pad, self.block_size)
            x_re = x_re.permute(0, 1, 3, 2, 4, 5, 6).reshape(B, N, G_pad, self.block_size)
            x_re = x_re[:, :, : self.num_blocks, :]
        else:
            x_re = x_re[:, :N, : self.num_blocks, :]
        # Unfold the frames back into the token axis (no-op when FFT_BATCH == 1).
        x_re = x_re.reshape(B_in, N_in, C)
        return x_re.to(self.ACT_T)


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2, **kwargs):
        super().__init__()
        self.dim = dim
        self.norm = _resolve_norm_layer(kwargs)(dim, dtype=FP32)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dtype=kwargs["MAMBA_MAIN_T"],  # NOTE should be enough to put most of Mamba's layer in the correct type
            dtype_act=kwargs["MAMBA_ACT_T"],
            quant=kwargs.get("MAMBA_QUANT"),
            use_hardware_act=kwargs["MAMBA_USE_HARDWARE_ACT"],
        )

    def forward(self, x):
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
        elif isinstance(m, nn.RMSNorm):
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
        elif isinstance(m, nn.RMSNorm):
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
        self.norm = _resolve_norm_layer(kwargs)(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.RMSNorm):
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
                in_channels,
                hidden_dim,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
                dtype=kwargs["PATCH_EMBED_T"],
            ),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                dtype=kwargs["PATCH_EMBED_T"],
            ),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                dtype=kwargs["PATCH_EMBED_T"],
            ),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(
            hidden_dim,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            dtype=kwargs["PATCH_EMBED_T"],
        )
        self.norm = _resolve_norm_layer(kwargs)(out_channels, dtype=FP32)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.RMSNorm):
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
        **kwargs: torch.dtype,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.AUTOCAST_T: torch.dtype = kwargs["AUTOCAST_T"]

        # Resolve norm_layer (caller may either pass it directly or via NORM_TYPE in kwargs)
        norm_type = kwargs.pop("NORM_TYPE", None)
        if norm_type is not None:
            norm_layer = make_norm_layer(norm_type, eps=1e-6)
        # Make the chosen factory visible to submodules that take **kwargs only.
        kwargs["NORM_LAYER"] = norm_layer

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
                    dim=embed_dims[-1],
                    mlp_ratio=mlp_ratios[-1],
                    norm_layer=norm_layer,
                    **kwargs,
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
        elif isinstance(m, nn.RMSNorm):
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
