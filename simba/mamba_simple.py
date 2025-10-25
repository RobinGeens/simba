"""
Copyright (c) 2023, Tri Dao, Albert Gu.

Local copy of mamba, where we can more easily change the types. Note that the CUDA kernels still come from the Mamba
Python package. To change, for example, the exponent implementation, run through the following steps:
1) Change the kernel source code in mamba sub-module, or ensure the submodule is in the proper commit
2) Install mamba as a package with `pip install ./mamba --no-build-isolation`
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .quantizer import FloatQuantizer, QuantizerPassthrough
import numpy as np
from pathlib import Path


class Mamba(nn.Module):

    _global_layer_counter = 0

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        dtype_act=None,
        quant: tuple[int, int] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        self.dtype_act = dtype_act if dtype_act is not None else dtype
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path

        # Use layer_idx if provided, otherwise use global counter
        if layer_idx is not None:
            self.layer_idx = layer_idx
        else:
            self.layer_idx = Mamba._global_layer_counter
            Mamba._global_layer_counter += 1

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # TensorBoard writer for profiling
        self.writer = SummaryWriter(log_dir="runs/mamba_profile")
        self.global_step = 0

        # Quantizer
        self.quantizer = QuantizerPassthrough() if quant is None else FloatQuantizer(*quant)
        print(f"Quantizing Mamba: {self.quantizer}")

        # Profiler
        self.enable_profile = False
        self.save_dir = Path("./simba_profile_data")

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        hidden_states = hidden_states.to(self.dtype_act)
        conv_state, ssm_state = None, None

        if self.enable_profile:
            self._save_tensor("hidden_states", hidden_states)
            self._save_tensor("weight_in_proj", self.in_proj.weight)

        # We do matmul and transpose BLH -> HBL at the same time
        # [NOTE] Chao: Quantize here
        xz = rearrange(
            self.quantizer.quantize(self.in_proj.weight.to(self.dtype_act))
            @ rearrange(self.quantizer.quantize(hidden_states), "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )

        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(self.dtype_act), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat

        x, z = xz.chunk(2, dim=1)
        if self.enable_profile:
            self._save_tensor("x", x)
            self._save_tensor("z", z)

        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        if causal_conv1d_fn is None:
            x: Tensor = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x: Tensor = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w").to(self.dtype_act),
                bias=self.conv1d.bias.to(self.dtype_act),
                activation=self.activation,
            )

            # [NOTE] Chao: Quantize here
            x_dbl = rearrange(
                self.quantizer.quantize(self.x_proj.weight).to(self.dtype_act)
                @ rearrange(self.quantizer.quantize(x), "b d l -> d (b l)"),
                "d (b l) -> (b l) d",
                b=x.shape[0],
                l=x.shape[2],
            )
            if self.x_proj.bias is not None:
                x_dbl = x_dbl + self.x_proj.bias.to(self.dtype_act)

            if self.enable_profile:
                self._save_tensor("x_proj_weight", self.x_proj.weight)
                self._save_tensor("x_dbl", x_dbl)

            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            # [NOTE] Chao: Quantize here
            dt = self.quantizer.quantize(self.dt_proj.weight) @ self.quantizer.quantize(dt).t()
            if self.enable_profile:
                self._save_tensor("dt_proj_weight", self.dt_proj.weight)
                self._save_tensor("dt", dt)

            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]

            y = selective_scan_fn(
                self.quantizer.quantize(x),
                dt,  # Not a big matrix
                A,  # External weight, fully tiled
                self.quantizer.quantize(B),
                self.quantizer.quantize(C),
                self.D.float(),  # TODO does this have to be FP32?
                z=z,  # TODO computed on-the-fly so memory capacity not an issue
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            # [NOTE] Chao: Quantize here
            out = rearrange(
                self.quantizer.quantize(self.out_proj.weight).to(self.dtype_act)
                @ rearrange(self.quantizer.quantize(y), "b l d -> d (b l)"),
                "d (b l) -> b l d",
                b=y.shape[0],
                l=y.shape[1],
            )
            if self.out_proj.bias is not None:
                out = out + self.out_proj.bias.to(self.dtype_act)

            if self.enable_profile:
                self._save_tensor("y", y)
                self._save_tensor("out", out)
                self._save_tensor("weight_out_proj", self.out_proj.weight)

        return out

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.dtype_act
        conv_state = torch.zeros(batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype)
        ssm_dtype = self.dtype_act
        ssm_state = torch.zeros(batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype)
        return conv_state, ssm_state

    def _save_tensor(self, name, tensor):
        """Save a single tensor to file"""
        if not self.enable_profile:
            return

        # Create save directory
        self.save_dir.mkdir(exist_ok=True)

        # Convert to numpy
        tensor_np = tensor.detach().cpu().float().numpy()

        # Filename format: layer_idx_name.npy
        filename = f"layer_{self.layer_idx:02d}_{name}.npy"
        filepath = self.save_dir / filename

        # Save
        np.save(filepath, tensor_np)
        print(f"Saved: {filepath} with shape {tensor_np.shape}")
