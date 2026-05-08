import torch
import numpy as np


class FloatQuantizer:
    """
    Quantize floating-point tensors to custom (e, m) format
    e: exponent bits
    m: mantissa bits
    Note: Subnormal numbers are not supported
    Args:
        dummy: Iff True, this module passes inputs through without quantizing
    """

    def __init__(self, e_bits, m_bits, verbose=False):

        self.e_bits = e_bits
        self.m_bits = m_bits

        # Calculate maximum exponent value for this format
        self.max_exponent = (1 << (e_bits - 1)) - 1  # bias value
        self.bias = self.max_exponent

        # Calculate maximum representable value
        # max_val = 2^max_exp * (1 + (2^m_bits - 1) / 2^m_bits)
        # Simplified to: 2^max_exp * (2 - 2^(-m_bits))
        self.max_val = (2**self.max_exponent) * (2.0 - 2.0 ** (-m_bits))

        if verbose:
            print(f"Quantization format: E{e_bits}M{m_bits}")
            print(f"Max exponent: {self.max_exponent}")
            print(f"Max representable value: {self.max_val}")

    def quantize(self, tensor):
        """
        Quantize input tensor

        Args:
            tensor: Input BF16/FP16/FP32 tensor

        Returns:
            Quantized tensor
        """
        # Convert to FP32 for processing
        x = tensor.float()

        # Save sign
        sign = torch.sign(x)
        x_abs = torch.abs(x)

        # Handle zero values
        zero_mask = x_abs == 0

        # Clamp values exceeding the range to max value
        overflow_mask = x_abs > self.max_val
        x_abs = torch.clamp(x_abs, 0, self.max_val)

        # Extract exponent and mantissa
        # log2(x) = exponent + log2(mantissa)
        exponent = torch.floor(torch.log2(x_abs + 1e-45))  # avoid log(0)

        # Calculate mantissa: mantissa = x / 2^exponent
        mantissa = x_abs / (2.0**exponent)

        # Quantize mantissa to m_bits
        # Mantissa range is [1, 2), we need to quantize the fractional part [0, 1)
        mantissa_frac = mantissa - 1.0
        mantissa_levels = 2**self.m_bits
        mantissa_quantized = torch.round(mantissa_frac * mantissa_levels) / mantissa_levels
        mantissa_quantized = mantissa_quantized + 1.0

        # Quantize exponent to e_bits
        exponent_clamped = torch.clamp(exponent, -self.bias, self.max_exponent)

        # Reconstruct quantized value
        result = sign * (2.0**exponent_clamped) * mantissa_quantized

        # Restore zero values
        result[zero_mask] = 0

        # Convert back to original data type
        if tensor.dtype == torch.bfloat16:
            result = result.bfloat16()
        elif tensor.dtype == torch.float16:
            result = result.half()

        return result

    def __str__(self):
        return f"E{self.e_bits}M{self.m_bits}"

    def get_stats(self, original, quantized):
        """Get quantization statistics"""
        mse = torch.mean((original - quantized) ** 2).item()
        max_error = torch.max(torch.abs(original - quantized)).item()

        return {"mse": mse, "max_error": max_error, "max_val": self.max_val}


class QuantizerPassthrough(FloatQuantizer):
    def __init__(self):
        pass

    def quantize(self, tensor):
        return tensor

    def __str__(self):
        return "Passthrough"


# Usage example
if __name__ == "__main__":
    # Create test data
    test_tensor = torch.randn(100) * 10  # FP32

    # Create quantizer: E4M3 format (e.g., FP8)
    quantizer = FloatQuantizer(e_bits=4, m_bits=3)

    # Quantize
    quantized = quantizer.quantize(test_tensor)

    # Statistics
    stats = quantizer.get_stats(test_tensor, quantized)
    print(f"\nMSE: {stats['mse']:.6f}")
    print(f"Max error: {stats['max_error']:.6f}")

    # Show some samples
    print("\nSample comparison:")
    for i in range(5):
        print(f"Original: {test_tensor[i]:10.6f} -> Quantized: {quantized[i]:10.6f}")
