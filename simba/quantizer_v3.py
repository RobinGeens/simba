import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class FloatType(Enum):
    """Floating-point number types according to IEEE 754"""
    ZERO = "Zero"
    SUBNORMAL = "Subnormal"
    NORMAL = "Normal"
    INFINITY = "Infinity"
    NAN = "NaN"

@dataclass
class FloatEntry:
    """Entry in the floating-point lookup table"""
    value: Optional[float]  # None for NaN/Inf
    float_type: FloatType
    sign: int  # 0 for positive, 1 for negative
    exponent_bits: int  # Raw exponent bits
    mantissa_bits: int  # Raw mantissa bits

class FloatQuantizer:
    """
    Quantize floating-point tensors to custom (e, m) format following IEEE 754
    
    Parameters:
        e: exponent bits (including sign bit in exponent encoding)
        m: mantissa bits (fractional part only, implicit leading 1 for normal numbers)
    
    Supports:
        - Normal numbers: standard floating-point representation
        - Subnormal numbers: for gradual underflow
        - Zero: positive and negative zero
        - Infinity: positive and negative infinity
        - NaN: Not-a-Number
    
    Rounding mode: Round toward zero (truncation)
    """
    
    def __init__(self, e_bits: int, m_bits: int):
        """
        Initialize the quantizer with specified bit widths
        
        Args:
            e_bits: Number of exponent bits (typically 4-8)
            m_bits: Number of mantissa bits (typically 2-10)
        """
        self.e_bits = e_bits
        self.m_bits = m_bits
        
        # IEEE 754 parameters
        self.bias = (1 << (e_bits - 1)) - 1  # Exponent bias: 2^(e-1) - 1
        self.emax = self.bias  # Maximum exponent value
        self.emin = 1 - self.bias  # Minimum normal exponent value
        
        # Calculate boundary values
        self.max_normal = (2 ** self.emax) * (2.0 - 2.0 ** (-m_bits))
        self.min_normal = 2.0 ** self.emin
        self.min_subnormal = (2.0 ** self.emin) * (2.0 ** (-m_bits))
        
        # Mantissa quantization levels
        self.mantissa_levels = 2 ** m_bits
        
        # Build lookup table (for inspection only, not used in fast quantization)
        self.lookup_table = self._build_lookup_table()
        
        # Print format information
        print(f"Quantization format: E{e_bits}M{m_bits}")
        print(f"Bias: {self.bias}")
        print(f"Exponent range: [{self.emin}, {self.emax}]")
        print(f"Number of representable values: {len(self.lookup_table)}")
        print(f"Max normal value: {self.max_normal}")
        print(f"Min normal value: {self.min_normal}")
        print(f"Min subnormal value: {self.min_subnormal}")
    
    def _build_lookup_table(self) -> List[FloatEntry]:
        """
        Build complete lookup table for all representable values
        (Used for inspection and validation, not for quantization)
        
        Returns:
            List of FloatEntry objects representing all possible values
        """
        table = []
        
        max_exp_bits = (1 << self.e_bits) - 1
        
        # Iterate through all possible bit patterns
        for sign in [0, 1]:
            for exp_bits in range(1 << self.e_bits):
                for mant_bits in range(1 << self.m_bits):
                    entry = self._decode_bits(sign, exp_bits, mant_bits)
                    table.append(entry)
        
        return table
    
    def _decode_bits(self, sign: int, exp_bits: int, mant_bits: int) -> FloatEntry:
        """
        Decode bit pattern to floating-point value according to IEEE 754
        
        Args:
            sign: Sign bit (0=positive, 1=negative)
            exp_bits: Exponent bit pattern
            mant_bits: Mantissa bit pattern
            
        Returns:
            FloatEntry object with decoded value and type
        """
        max_exp_bits = (1 << self.e_bits) - 1
        
        if exp_bits == 0:
            if mant_bits == 0:
                value = 0.0 if sign == 0 else -0.0
                return FloatEntry(value, FloatType.ZERO, sign, exp_bits, mant_bits)
            else:
                mantissa_frac = mant_bits / (1 << self.m_bits)
                value = (2.0 ** self.emin) * mantissa_frac
                if sign == 1:
                    value = -value
                return FloatEntry(value, FloatType.SUBNORMAL, sign, exp_bits, mant_bits)
        elif exp_bits == max_exp_bits:
            if mant_bits == 0:
                return FloatEntry(None, FloatType.INFINITY, sign, exp_bits, mant_bits)
            else:
                return FloatEntry(None, FloatType.NAN, sign, exp_bits, mant_bits)
        else:
            exponent = exp_bits - self.bias
            mantissa_frac = mant_bits / (1 << self.m_bits)
            mantissa = 1.0 + mantissa_frac
            value = (2.0 ** exponent) * mantissa
            if sign == 1:
                value = -value
            return FloatEntry(value, FloatType.NORMAL, sign, exp_bits, mant_bits)
    
    def print_lookup_table(self, max_entries: int = 50):
        """Print the lookup table for inspection"""
        print("\n" + "="*80)
        print(f"Lookup Table (showing first {max_entries} entries)")
        print("="*80)
        print(f"{'Sign':<6} {'Exp':<8} {'Mant':<10} {'Type':<12} {'Value':<20}")
        print("-"*80)
        
        for i, entry in enumerate(self.lookup_table[:max_entries]):
            sign_str = '-' if entry.sign == 1 else '+'
            exp_str = f"{entry.exponent_bits:0{self.e_bits}b}"
            mant_str = f"{entry.mantissa_bits:0{self.m_bits}b}"
            
            if entry.value is None:
                if entry.float_type == FloatType.INFINITY:
                    value_str = f"{sign_str}Inf"
                else:
                    value_str = "NaN"
            else:
                value_str = f"{entry.value:.10e}"
            
            print(f"{sign_str:<6} {exp_str:<8} {mant_str:<10} {entry.float_type.value:<12} {value_str:<20}")
        
        if len(self.lookup_table) > max_entries:
            print(f"... ({len(self.lookup_table) - max_entries} more entries)")
        print("="*80 + "\n")
    
    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Fast quantization using vectorized mathematical operations
        
        Quantization rules (IEEE 754 compliant):
        1. NaN → NaN
        2. ±Inf → ±max_val (overflow clamping)
        3. |x| > max_val → ±max_val (overflow clamping)
        4. min_normal ≤ |x| ≤ max_val → quantize to normal numbers
        5. min_subnormal ≤ |x| < min_normal → quantize to subnormal numbers
        6. |x| < min_subnormal → 0 (underflow)
        
        Rounding mode: Round toward zero (truncation)
        
        Args:
            tensor: Input BF16/FP16/FP32 tensor
            
        Returns:
            Quantized tensor in original dtype
        """
        original_dtype = tensor.dtype
        x = tensor.float()
        
        # Save sign and work with absolute values
        sign = torch.sign(x)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)  # Handle zero
        x_abs = torch.abs(x)
        
        # Handle special values
        nan_mask = torch.isnan(x)
        inf_mask = torch.isinf(x)
        zero_mask = (x_abs == 0)
        
        # Initialize result
        result = torch.zeros_like(x)
        
        # Clamp overflow to max value
        x_abs = torch.clamp(x_abs, 0, self.max_normal)
        
        # Separate normal and subnormal ranges
        normal_mask = (x_abs >= self.min_normal) & (x_abs <= self.max_normal)
        subnormal_mask = (x_abs >= self.min_subnormal) & (x_abs < self.min_normal)
        underflow_mask = (x_abs < self.min_subnormal) & (~zero_mask)
        
        # Quantize normal numbers
        if normal_mask.any():
            x_normal = x_abs[normal_mask]
            
            # Extract exponent: floor(log2(x))
            exponent = torch.floor(torch.log2(x_normal))
            
            # Calculate mantissa: x / 2^exponent, range [1, 2)
            mantissa = x_normal / (2.0 ** exponent)
            
            # Quantize mantissa fractional part [0, 1) to m_bits
            # Round toward zero: use floor instead of round
            mantissa_frac = mantissa - 1.0
            mantissa_quantized_frac = torch.floor(mantissa_frac * self.mantissa_levels) / self.mantissa_levels
            mantissa_quantized = mantissa_quantized_frac + 1.0
            
            # Clamp exponent to valid range
            exponent_clamped = torch.clamp(exponent, self.emin, self.emax)
            
            # Reconstruct quantized value
            result[normal_mask] = (2.0 ** exponent_clamped) * mantissa_quantized
        
        # Quantize subnormal numbers
        if subnormal_mask.any():
            x_subnormal = x_abs[subnormal_mask]
            
            # Subnormal: x = 2^emin * (0.mantissa)
            # mantissa_frac = x / 2^emin, range [0, 1)
            mantissa_frac = x_subnormal / self.min_normal
            
            # Quantize mantissa to m_bits (round toward zero)
            mantissa_quantized_frac = torch.floor(mantissa_frac * self.mantissa_levels) / self.mantissa_levels
            
            # Reconstruct subnormal value
            result[subnormal_mask] = self.min_normal * mantissa_quantized_frac
        
        # Handle underflow: values too small become zero
        result[underflow_mask] = 0.0
        
        # Handle zeros
        result[zero_mask] = 0.0
        
        # Restore sign
        result = sign * result
        
        # Handle special values
        result[nan_mask] = float('nan')
        result[inf_mask] = sign[inf_mask] * self.max_normal  # Clamp infinities
        
        # Convert back to original dtype
        result = result.to(original_dtype)
        
        return result
    
    def get_stats(self, original: torch.Tensor, quantized: torch.Tensor) -> Dict:
        """Get quantization statistics"""
        orig = original.float()
        quant = quantized.float()
        
        # Only compute error on finite values
        finite_mask = torch.isfinite(orig) & torch.isfinite(quant)
        
        if finite_mask.sum() > 0:
            error = orig[finite_mask] - quant[finite_mask]
            mse = torch.mean(error ** 2).item()
            max_error = torch.max(torch.abs(error)).item()
            mean_error = torch.mean(torch.abs(error)).item()
        else:
            mse = 0.0
            max_error = 0.0
            mean_error = 0.0
        
        return {
            'mse': mse,
            'max_error': max_error,
            'mean_error': mean_error,
            'max_val': self.max_normal,
            'min_normal': self.min_normal,
            'min_subnormal': self.min_subnormal
        }


# Performance comparison
if __name__ == "__main__":
    import time
    
    # Create quantizer
    quantizer = FloatQuantizer(e_bits=4, m_bits=3)
    
    # Print lookup table
    quantizer.print_lookup_table(max_entries=256)
    
    # Test correctness with small examples
    print("\n" + "="*80)
    print("Correctness Tests")
    print("="*80)
    
    test_cases = [
        torch.tensor([0.0, -0.0, 1.0, -1.0]),
        torch.tensor([0.001, 0.01, 0.1, 1.0, 10.0, 100.0]),
        torch.tensor([float('nan'), float('inf'), float('-inf')]),
    ]
    
    for i, test_tensor in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        quantized = quantizer.quantize(test_tensor)
        
        print(f"{'Original':<20} {'Quantized':<20}")
        print("-"*40)
        for j in range(len(test_tensor)):
            orig = test_tensor[j].item()
            quant = quantized[j].item()
            print(f"{orig:<20.10f} {quant:<20.10f}")
    
    # Performance benchmark
    print("\n" + "="*80)
    print("Performance Benchmark")
    print("="*80)
    
    sizes = [1000, 10000, 100000, 1000000]
    
    for size in sizes:
        test_tensor = torch.randn(size) * 10
        
        # Warm up
        _ = quantizer.quantize(test_tensor)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(10):
            _ = quantizer.quantize(test_tensor)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        
        avg_time = (end - start) / 10 * 1000  # ms
        throughput = size / (avg_time / 1000) / 1e6  # M elements/sec
        
        print(f"Size: {size:>7} | Time: {avg_time:>6.2f} ms | Throughput: {throughput:>6.2f} M elem/s")
    
    # GPU test if available
    if torch.cuda.is_available():
        print("\n" + "="*80)
        print("GPU Performance Benchmark")
        print("="*80)
        
        for size in sizes:
            test_tensor = torch.randn(size, device='cuda') * 10
            
            # Warm up
            _ = quantizer.quantize(test_tensor)
            
            # Benchmark
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                _ = quantizer.quantize(test_tensor)
            torch.cuda.synchronize()
            end = time.time()
            
            avg_time = (end - start) / 10 * 1000  # ms
            throughput = size / (avg_time / 1000) / 1e6  # M elements/sec
            
            print(f"Size: {size:>7} | Time: {avg_time:>6.2f} ms | Throughput: {throughput:>6.2f} M elem/s")