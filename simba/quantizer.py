import torch
import math
from torch import Tensor


class FloatQuantizer:
    """
    Quantize floating-point tensors to custom (e, m) format
    e: exponent bits
    m: mantissa bits
    Note: Subnormal numbers are not supported
    """
    
    _logged_configs = set()

    def __init__(self, e_bits, m_bits):
        self.e_bits = e_bits
        self.m_bits = m_bits
        
        # Calculate maximum exponent value for this format
        self.max_exponent = (1 << (e_bits - 1)) - 1  # bias value
        self.bias = self.max_exponent
        
        # Calculate maximum representable value
        # max_val = 2^max_exp * (1 + (2^m_bits - 1) / 2^m_bits)
        # Simplified to: 2^max_exp * (2 - 2^(-m_bits))
        self.max_val = (2 ** self.max_exponent) * (2.0 - 2.0 ** (-m_bits))
        
        # Calculate minimum representable value (smallest normal number)
        self.min_val = 2.0 ** (-self.bias)
        
        # Print configuration once per format
        config_key = (self.e_bits, self.m_bits)
        if config_key not in FloatQuantizer._logged_configs:
            print(
                f"    * config E{e_bits}M{m_bits}: "
                f"max_exp={self.max_exponent}, max={self.max_val:.6g}, min={self.min_val:.6g}"
            )
            FloatQuantizer._logged_configs.add(config_key)
    
    def quantize(self, tensor : Tensor):
        # Convert to FP32 for processing
        x = tensor.float()

        # Save sign
        sign = torch.sign(x)
        x_abs = torch.abs(x)
        
        # Handle zero values
        zero_mask = (x_abs == 0)
        
        # Handle NaN values
        nan_mask = torch.isnan(x_abs)
        
        # Handle infinity values - clamp to max value
        inf_mask = torch.isinf(x_abs)
        x_abs[inf_mask] = self.max_val
        
        # Clamp values exceeding the range to max value
        x_abs = torch.clamp(x_abs, 0, self.max_val)
        
        # Handle underflow - values too small to represent
        underflow_mask = (x_abs < self.min_val * 0.5) & ~zero_mask
        
        # Extract exponent and mantissa using frexp for numerical stability
        # frexp returns: x = mantissa × 2^exponent, mantissa ∈ [0.5, 1)
        mantissa_half, exponent_frexp = torch.frexp(x_abs)
        
        # Convert to standard form: mantissa ∈ [1, 2), exponent -= 1
        mantissa = mantissa_half * 2.0
        exponent = exponent_frexp - 1

        
        # Quantize mantissa to m_bits
        # Mantissa range is [1, 2), we need to quantize the fractional part [0, 1)
        mantissa_frac = mantissa - 1.0
        mantissa_levels = 2 ** self.m_bits
        mantissa_quantized = torch.round(mantissa_frac * mantissa_levels) / mantissa_levels
        mantissa_quantized = mantissa_quantized + 1.0

        # Quantize exponent to e_bits
        exponent_clamped = torch.clamp(exponent, -self.bias, self.max_exponent)
        
        # Reconstruct quantized value using ldexp for numerical stability
        result = sign * torch.ldexp(mantissa_quantized, exponent_clamped.int())

        # Restore special values
        result[zero_mask | underflow_mask] = 0
        result[nan_mask] = float('nan')
        
        # Convert back to original data type
        if tensor.dtype == torch.bfloat16:
            result = result.bfloat16()
        elif tensor.dtype == torch.float16:
            result = result.half()
              
        return result
    
    def get_stats(self, original : Tensor, quantized : Tensor):
        """Get quantization statistics"""
        # Filter out inf and nan for statistics
        valid_mask = torch.isfinite(original) & torch.isfinite(quantized)
        
        if valid_mask.sum() == 0:
            return {
                'mse': float('nan'),
                'max_error': float('nan'),
                'max_val': self.max_val,
                'min_val': self.min_val
            }
        
        orig_valid = original[valid_mask]
        quant_valid = quantized[valid_mask]
        
        mse = torch.mean((orig_valid - quant_valid) ** 2).item()
        max_error = torch.max(torch.abs(orig_valid - quant_valid)).item()
        
        return {
            'mse': mse,
            'max_error': max_error,
            'max_val': self.max_val,
            'min_val': self.min_val
        }

    def __str__(self):
        return f"E{self.e_bits}M{self.m_bits}"

    def __repr__(self):
        return f"FloatQuantizer(e_bits={self.e_bits}, m_bits={self.m_bits})"


class QuantizerPassthrough(FloatQuantizer):
    def __init__(self):
        pass

    def quantize(self, tensor : Tensor):
        return tensor

    def __str__(self):
        return "Passthrough"

