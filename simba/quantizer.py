import torch
import math
from torch import Tensor


class FloatQuantizer:
    """
    Quantize floating-point tensors to custom (e, m) format.
    e: exponent bits
    m: mantissa bits

    Implementation: precomputes a 65,536-entry BF16 lookup table at __init__,
    so runtime quantization is a single gather on GPU. Inputs are cast to BF16
    before lookup; outputs are returned in the input's dtype.
    """

    _logged_configs = set()

    def __init__(self, e_bits, m_bits):
        self.e_bits = e_bits
        self.m_bits = m_bits

        # Calculate maximum exponent value for this format
        self.max_exponent = (1 << (e_bits - 1)) - 1  # bias value
        self.bias = self.max_exponent

        # max_val = 2^max_exp * (2 - 2^(-m_bits))
        self.max_val = (2 ** self.max_exponent) * (2.0 - 2.0 ** (-m_bits))

        # Smallest normal value
        self.min_val = 2.0 ** (-self.bias)

        config_key = (self.e_bits, self.m_bits)
        if config_key not in FloatQuantizer._logged_configs:
            print(
                f"    * config E{e_bits}M{m_bits}: "
                f"max_exp={self.max_exponent}, max={self.max_val:.6g}, min={self.min_val:.6g}"
            )
            FloatQuantizer._logged_configs.add(config_key)

        # Build LUT on CPU once. Mirror to other devices lazily.
        self._lut_cpu = self._build_bf16_lut()
        self._lut_cache = {torch.device("cpu"): self._lut_cpu}

    def _quantize_reference(self, tensor: Tensor) -> Tensor:
        """
        Reference implementation using frexp/ldexp. Slow but numerically robust.
        Used only at __init__ to populate the LUT.
        """
        x = tensor.float()
        sign = torch.sign(x)
        x_abs = torch.abs(x)

        zero_mask = x_abs == 0
        nan_mask = torch.isnan(x_abs)
        inf_mask = torch.isinf(x_abs)
        x_abs = torch.where(inf_mask, torch.full_like(x_abs, self.max_val), x_abs)
        x_abs = torch.clamp(x_abs, 0, self.max_val)
        underflow_mask = (x_abs < self.min_val * 0.5) & ~zero_mask

        # frexp: x = mantissa_half * 2^exp, mantissa_half ∈ [0.5, 1)
        mantissa_half, exp_frexp = torch.frexp(x_abs)
        mantissa = mantissa_half * 2.0
        exponent = exp_frexp - 1

        mantissa_frac = mantissa - 1.0
        levels = 2 ** self.m_bits
        mantissa_q = torch.round(mantissa_frac * levels) / levels + 1.0

        exponent_q = torch.clamp(exponent, -self.bias, self.max_exponent)

        result = sign * torch.ldexp(mantissa_q, exponent_q.int())

        result = torch.where(zero_mask | underflow_mask, torch.zeros_like(result), result)
        result = torch.where(nan_mask, torch.full_like(result, float("nan")), result)
        return result

    def _build_bf16_lut(self) -> Tensor:
        """
        Enumerate all 65,536 BF16 bit patterns, quantize each, store as BF16.
        lut[i] is the quantized value of the BF16 with bit pattern i.
        """
        bits = torch.arange(65536, dtype=torch.int32)
        # Wrap to signed int16 (preserves bit pattern), then reinterpret as bfloat16.
        bf16_vals = bits.to(torch.int16).view(torch.bfloat16)

        with torch.no_grad():
            quant_fp32 = self._quantize_reference(bf16_vals)
        return quant_fp32.bfloat16().contiguous()

    def _get_lut(self, device: torch.device) -> Tensor:
        lut = self._lut_cache.get(device)
        if lut is None:
            lut = self._lut_cpu.to(device)
            self._lut_cache[device] = lut
        return lut

    def quantize(self, tensor: Tensor) -> Tensor:
        """
        Quantize input. BF16 inputs take the fast LUT path; other dtypes use the reference implementation to avoid a
        lossy BF16 round-trip that would double-round at quantization midpoints.

        Backward uses a straight-through estimator: forward returns the quantized value, backward passes the upstream 
        gradient through unchanged. Without STE the LUT gather / frexp ops sever autograd and any upstream parameters 
        never receive a gradient.
        """
        if tensor.dtype == torch.bfloat16:
            x = tensor if tensor.is_contiguous() else tensor.contiguous()
            lut = self._get_lut(x.device)
            # Reinterpret BF16 bits as int16, mask to unsigned 16-bit index, gather.
            idx = x.view(torch.int16).to(torch.int32) & 0xFFFF
            q = lut[idx]
        else:
            # FP16/FP32 fallback: identical math to the original quantizer.
            q = self._quantize_reference(tensor).to(tensor.dtype)

        return tensor + (q - tensor).detach()

    def get_stats(self, original: Tensor, quantized: Tensor):
        """Get quantization statistics"""
        valid_mask = torch.isfinite(original) & torch.isfinite(quantized)

        if valid_mask.sum() == 0:
            return {
                "mse": float("nan"),
                "max_error": float("nan"),
                "max_val": self.max_val,
                "min_val": self.min_val,
            }

        orig_valid = original[valid_mask]
        quant_valid = quantized[valid_mask]

        mse = torch.mean((orig_valid - quant_valid) ** 2).item()
        max_error = torch.max(torch.abs(orig_valid - quant_valid)).item()

        return {
            "mse": mse,
            "max_error": max_error,
            "max_val": self.max_val,
            "min_val": self.min_val,
        }

    def __str__(self):
        return f"E{self.e_bits}M{self.m_bits}"

    def __repr__(self):
        return f"FloatQuantizer(e_bits={self.e_bits}, m_bits={self.m_bits})"


class QuantizerPassthrough(FloatQuantizer):
    def __init__(self):
        pass

    def quantize(self, tensor: Tensor):
        return tensor

    def __str__(self):
        return "Passthrough"
