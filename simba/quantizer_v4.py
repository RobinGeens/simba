import torch
import numpy as np
import math

class FloatQuantizer:
    """
    Quantize floating-point tensors to custom (e, m) format
    e: exponent bits
    m: mantissa bits
    Note: Subnormal numbers are not supported
    """
    
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
        
        print(f"Quantization format: E{e_bits}M{m_bits}")
        print(f"Max exponent: {self.max_exponent}")
        print(f"Max representable value: {self.max_val}")
        print(f"Min representable value: {self.min_val}")
    
    def quantize(self, tensor, debug=False):
        """
        Quantize input tensor
        
        Args:
            tensor: Input BF16/FP16/FP32 tensor
            debug: Whether to print debug information
            
        Returns:
            Quantized tensor
        """
        # Convert to FP32 for processing
        x = tensor.float()
        
        if debug:
            print("X:", x)
        
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
        overflow_mask = (x_abs > self.max_val)
        x_abs = torch.clamp(x_abs, 0, self.max_val)
        
        # Handle underflow - values too small to represent
        underflow_mask = (x_abs < self.min_val * 0.5) & ~zero_mask
        
        if debug:
            print("X abs:", x_abs)
            print("Underflow Mask:", underflow_mask)
        
        # Extract exponent and mantissa using frexp for numerical stability
        # frexp returns: x = mantissa × 2^exponent, mantissa ∈ [0.5, 1)
        mantissa_half, exponent_frexp = torch.frexp(x_abs)
        
        # Convert to standard form: mantissa ∈ [1, 2), exponent -= 1
        mantissa = mantissa_half * 2.0
        exponent = exponent_frexp - 1
        
        if debug:
            print("Exponent:", exponent)
            print("Mantissa:", mantissa)
        
        # Quantize mantissa to m_bits
        # Mantissa range is [1, 2), we need to quantize the fractional part [0, 1)
        mantissa_frac = mantissa - 1.0
        mantissa_levels = 2 ** self.m_bits
        mantissa_quantized = torch.round(mantissa_frac * mantissa_levels) / mantissa_levels
        mantissa_quantized = mantissa_quantized + 1.0
        
        if debug:
            print("Mantissa Quantized:", mantissa_quantized)
        
        # Quantize exponent to e_bits
        exponent_clamped = torch.clamp(exponent, -self.bias, self.max_exponent)
        
        if debug:
            print("Exponent Clamped:", exponent_clamped)
        
        # Reconstruct quantized value using ldexp for numerical stability
        result = sign * torch.ldexp(mantissa_quantized, exponent_clamped.int())
        if debug:
            print("")
            print("Reconstructed Result:", result)
        
        # Restore special values
        result[zero_mask | underflow_mask] = 0
        result[nan_mask] = float('nan')
        
        # Convert back to original data type
        if tensor.dtype == torch.bfloat16:
            result = result.bfloat16()
        elif tensor.dtype == torch.float16:
            result = result.half()
            
        if debug:
            print("Final Quantized Result:", result)
            print("")
        
        return result
    
    def get_stats(self, original, quantized):
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


# Usage example
if __name__ == "__main__":
    # Create test data
    test_tensor = torch.randn(100) * 10  # FP32
    
    # Create quantizer: E4M3 format (e.g., FP8)
    # quantizer = FloatQuantizer(e_bits=4, m_bits=3)
    quantizer = FloatQuantizer(e_bits=5, m_bits=2)
    
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
    
    # Test edge cases with debug mode
    print("\n" + "="*60)
    print("Testing edge cases with debug output:")
    print("="*60)
    
    edge_cases = torch.tensor([
        2.7551e-40,            # subnormal (underflow)
        0.0078125,             # exactly min_val (2^-7)
        1.0,                   # normal value
    ])
    
    for i, val in enumerate(edge_cases):
        print(f"\n--- Test case {i+1}: {val.item():.6e} ---")
        quantized_val = quantizer.quantize(val.unsqueeze(0), debug=True)
        print(f"Result: {quantized_val[0].item():.6e}")
        
    # ========== New section: Enumerate all BF16 values ==========
    print("\n" + "="*60)
    print("Enumerating all BF16 normal and subnormal values")
    print("="*60)
    
    bf16_to_quantized = {}
    
    # Generate all possible BF16 bit patterns
    for bits in range(0, 65536):
        # Create BF16 value from bit pattern
        bf16_bytes = bits.to_bytes(2, byteorder='little')
        bf16_value = torch.frombuffer(
            bytearray(bf16_bytes), 
            dtype=torch.bfloat16
        )[0]
        
        # Quantize this BF16 value
        quantized_value = quantizer.quantize(bf16_value.unsqueeze(0), debug=True)[0]
        
        # Store in dictionary: key is the quantized value (as float32)
        key = quantized_value.item()
        
        # Skip NaN and Inf values
        if math.isfinite(key):
            if key not in bf16_to_quantized:
                bf16_to_quantized[key] = []
            bf16_to_quantized[key].append(bf16_value.item())
    
    # Print statistics
    print(f"\nTotal unique quantized values: {len(bf16_to_quantized)}")
    print(f"Total BF16 values mapped: {sum(len(v) for v in bf16_to_quantized.values())}")
    
    # Show some examples of the mapping
    print("\nSample mappings (first 10 unique quantized values):")
    sorted_keys = sorted(bf16_to_quantized.keys())
    for i, quant_val in enumerate(sorted_keys[:10]):
        bf16_vals = bf16_to_quantized[quant_val]
        valid_vals = [v for v in bf16_vals if math.isfinite(v)]
        if valid_vals:
            print(f"Quantized: {quant_val:12.6f} <- {len(valid_vals):5d} BF16 values, "
                  f"range: [{min(valid_vals):12.6e}, {max(valid_vals):12.6e}]")
    
    # Show positive values near 1.0
    print("\nQuantized values near 1.0:")
    near_one = {k: v for k, v in bf16_to_quantized.items() if 0.5 < k < 2.0}
    for quant_val in sorted(near_one.keys())[:15]:
        bf16_vals = near_one[quant_val]
        print(f"Quantized: {quant_val:12.6f} <- {len(bf16_vals):5d} BF16 values")
    
    # Show smallest positive quantized values
    print("\nSmallest positive quantized values:")
    positive_keys = [k for k in sorted_keys if k > 0]
    for quant_val in positive_keys[:10]:
        bf16_vals = bf16_to_quantized[quant_val]
        print(f"Quantized: {quant_val:12.6e} <- {len(bf16_vals):5d} BF16 values")