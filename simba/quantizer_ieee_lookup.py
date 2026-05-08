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
        
        # Build lookup table containing all representable values
        self.lookup_table = self._build_lookup_table()
        
        # Extract positive finite values for efficient quantization
        # These will be used for binary search during quantization
        self.positive_values = []
        self.positive_entries = []
        for entry in self.lookup_table:
            if entry.sign == 0 and entry.value is not None and entry.value > 0:
                self.positive_values.append(entry.value)
                self.positive_entries.append(entry)
        
        # Convert to PyTorch tensor for GPU-accelerated quantization
        self.positive_values_tensor = torch.tensor(self.positive_values, dtype=torch.float32)
        
        # Print format information
        print(f"Quantization format: E{e_bits}M{m_bits}")
        print(f"Bias: {self.bias}")
        print(f"Exponent range: [{self.emin}, {self.emax}]")
        print(f"Number of representable values: {len(self.lookup_table)}")
        print(f"Max normal value: {self._get_max_normal()}")
        print(f"Min normal value: {self._get_min_normal()}")
        print(f"Min subnormal value: {self._get_min_subnormal()}")
    
    def _build_lookup_table(self) -> List[FloatEntry]:
        """
        Build complete lookup table for all representable values
        
        Iterates through all possible bit patterns (sign, exponent, mantissa)
        and decodes them according to IEEE 754 standard
        
        Returns:
            List of FloatEntry objects representing all possible values
        """
        table = []
        
        max_exp_bits = (1 << self.e_bits) - 1  # All 1s in exponent
        max_mant_bits = (1 << self.m_bits) - 1  # All 1s in mantissa
        
        # Iterate through all possible bit patterns
        for sign in [0, 1]:  # 0: positive, 1: negative
            for exp_bits in range(1 << self.e_bits):  # All exponent patterns
                for mant_bits in range(1 << self.m_bits):  # All mantissa patterns
                    entry = self._decode_bits(sign, exp_bits, mant_bits)
                    table.append(entry)
        
        return table
    
    def _decode_bits(self, sign: int, exp_bits: int, mant_bits: int) -> FloatEntry:
        """
        Decode bit pattern to floating-point value according to IEEE 754
        
        IEEE 754 encoding rules:
        1. exp=0, mant=0: Zero (±0)
        2. exp=0, mant≠0: Subnormal (±2^emin × 0.mantissa)
        3. exp=max, mant=0: Infinity (±Inf)
        4. exp=max, mant≠0: NaN
        5. Otherwise: Normal (±2^(exp-bias) × 1.mantissa)
        
        Args:
            sign: Sign bit (0=positive, 1=negative)
            exp_bits: Exponent bit pattern
            mant_bits: Mantissa bit pattern
            
        Returns:
            FloatEntry object with decoded value and type
        """
        max_exp_bits = (1 << self.e_bits) - 1
        
        # Case 1: Exponent = 0 (Zero or Subnormal)
        if exp_bits == 0:
            if mant_bits == 0:
                # Zero: ±0
                value = 0.0 if sign == 0 else -0.0
                return FloatEntry(value, FloatType.ZERO, sign, exp_bits, mant_bits)
            else:
                # Subnormal number: ±2^emin × (0.mantissa)
                # No implicit leading 1, represents values smaller than min normal
                mantissa_frac = mant_bits / (1 << self.m_bits)
                value = (2.0 ** self.emin) * mantissa_frac
                if sign == 1:
                    value = -value
                return FloatEntry(value, FloatType.SUBNORMAL, sign, exp_bits, mant_bits)
        
        # Case 2: Exponent = all 1s (Infinity or NaN)
        elif exp_bits == max_exp_bits:
            if mant_bits == 0:
                # Infinity: ±Inf
                return FloatEntry(None, FloatType.INFINITY, sign, exp_bits, mant_bits)
            else:
                # NaN: Not a Number
                return FloatEntry(None, FloatType.NAN, sign, exp_bits, mant_bits)
        
        # Case 3: Normal number
        else:
            # Normal number: ±2^(exp-bias) × (1.mantissa)
            # Has implicit leading 1 in mantissa
            exponent = exp_bits - self.bias
            mantissa_frac = mant_bits / (1 << self.m_bits)
            mantissa = 1.0 + mantissa_frac  # Implicit leading 1
            value = (2.0 ** exponent) * mantissa
            if sign == 1:
                value = -value
            return FloatEntry(value, FloatType.NORMAL, sign, exp_bits, mant_bits)
    
    def _get_max_normal(self) -> float:
        """
        Get maximum representable normal value
        
        Returns:
            Maximum finite positive value in this format
        """
        max_normal = 0.0
        for entry in self.lookup_table:
            if entry.float_type == FloatType.NORMAL and entry.value is not None:
                max_normal = max(max_normal, abs(entry.value))
        return max_normal
    
    def _get_min_normal(self) -> float:
        """
        Get minimum positive normal value
        
        Returns:
            Smallest positive normal number: 2^emin
        """
        return 2.0 ** self.emin
    
    def _get_min_subnormal(self) -> float:
        """
        Get minimum positive subnormal value
        
        Returns:
            Smallest positive subnormal number: 2^emin × 2^(-m)
        """
        return (2.0 ** self.emin) * (1.0 / (1 << self.m_bits))
    
    def print_lookup_table(self, max_entries: int = 50):
        """
        Print the lookup table for inspection
        
        Displays the bit patterns, types, and values of representable numbers
        Useful for understanding the quantization behavior
        
        Args:
            max_entries: Maximum number of entries to display
        """
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
        Quantize input tensor using lookup table (pure PyTorch implementation)
        
        Quantization rules:
        1. NaN → NaN
        2. ±Inf → ±max_val (overflow clamping)
        3. |x| > max_val → ±max_val (overflow clamping)
        4. min_normal ≤ |x| ≤ max_val → quantize to normal numbers
        5. 0 < |x| < min_normal → quantize to subnormal numbers
        6. |x| < min_subnormal → 0 (underflow)
        
        Rounding mode: Round toward zero (truncation)
        - Positive values: round down (choose smaller absolute value)
        - Negative values: round up (choose smaller absolute value)
        - Both directions move toward zero
        
        Args:
            tensor: Input BF16/FP16/FP32 tensor
            
        Returns:
            Quantized tensor in original dtype
        """
        original_dtype = tensor.dtype
        original_device = tensor.device
        x = tensor.float()
        
        # Move lookup table to same device as input tensor
        lookup_values = self.positive_values_tensor.to(original_device)
        
        # Initialize result tensor
        result = torch.zeros_like(x)
        
        # Identify special cases
        nan_mask = torch.isnan(x)
        pos_inf_mask = torch.isposinf(x)
        neg_inf_mask = torch.isneginf(x)
        
        # Get maximum representable value
        max_val = self._get_max_normal()
        
        # Process finite values
        finite_mask = torch.isfinite(x)
        x_finite = x[finite_mask]
        
        # Separate positive, negative, and zero values
        pos_mask = x_finite > 0
        neg_mask = x_finite < 0
        zero_mask = x_finite == 0
        
        quantized_finite = torch.zeros_like(x_finite)
        
        # Handle positive values: round toward zero (round down)
        # For positive numbers, rounding toward zero means choosing the largest
        # representable value that is still ≤ the input value
        if pos_mask.sum() > 0:
            x_pos = x_finite[pos_mask]
            
            # Clamp overflow values
            overflow_mask = x_pos > max_val
            quantized_pos = torch.where(overflow_mask, 
                                       torch.tensor(max_val, device=original_device, dtype=torch.float32),
                                       x_pos)
            
            # For non-overflow values, find largest value ≤ input using searchsorted
            non_overflow_mask = ~overflow_mask
            if non_overflow_mask.sum() > 0:
                x_search = quantized_pos[non_overflow_mask]
                
                # searchsorted finds insertion indices
                # We subtract 1 to get the largest value ≤ input
                indices = torch.searchsorted(lookup_values, x_search, right=True) - 1
                
                # Handle underflow (indices < 0)
                underflow_mask = indices < 0
                indices = torch.clamp(indices, min=0)
                
                # Gather quantized values from lookup table
                quantized_search = lookup_values[indices]
                quantized_search[underflow_mask] = 0.0
                
                quantized_pos[non_overflow_mask] = quantized_search
            
            quantized_finite[pos_mask] = quantized_pos
        
        # Handle negative values: round toward zero (round up in absolute value)
        # For negative numbers, rounding toward zero means choosing the smallest
        # absolute value that is still ≥ |input value|
        # We work with absolute values and then restore the sign
        if neg_mask.sum() > 0:
            x_neg_abs = -x_finite[neg_mask]  # Work with absolute values
            
            # Clamp overflow values
            overflow_mask = x_neg_abs > max_val
            quantized_neg_abs = torch.where(overflow_mask,
                                           torch.tensor(max_val, device=original_device, dtype=torch.float32),
                                           x_neg_abs)
            
            # For non-overflow values, find largest value ≤ |input| using searchsorted
            non_overflow_mask = ~overflow_mask
            if non_overflow_mask.sum() > 0:
                x_search = quantized_neg_abs[non_overflow_mask]
                
                # searchsorted finds insertion indices
                # We subtract 1 to get the largest value ≤ |input|
                indices = torch.searchsorted(lookup_values, x_search, right=True) - 1
                
                # Handle underflow (indices < 0)
                underflow_mask = indices < 0
                indices = torch.clamp(indices, min=0)
                
                # Gather quantized values from lookup table
                quantized_search = lookup_values[indices]
                quantized_search[underflow_mask] = 0.0
                
                quantized_neg_abs[non_overflow_mask] = quantized_search
            
            # Restore negative sign
            quantized_finite[neg_mask] = -quantized_neg_abs
        
        # Handle zeros (both +0 and -0 map to +0)
        quantized_finite[zero_mask] = 0.0
        
        # Assign quantized finite values
        result[finite_mask] = quantized_finite
        
        # Handle special values
        result[nan_mask] = float('nan')  # NaN stays NaN
        result[pos_inf_mask] = max_val  # +Inf clamps to max
        result[neg_inf_mask] = -max_val  # -Inf clamps to -max
        
        # Convert back to original dtype
        result = result.to(original_dtype)
        
        return result
    
    def get_stats(self, original: torch.Tensor, quantized: torch.Tensor) -> Dict:
        """
        Get quantization statistics
        
        Computes error metrics between original and quantized tensors
        Only considers finite values for error computation
        
        Args:
            original: Original tensor before quantization
            quantized: Quantized tensor
            
        Returns:
            Dictionary containing:
                - mse: Mean squared error
                - max_error: Maximum absolute error
                - mean_error: Mean absolute error
                - max_val: Maximum representable value
                - min_normal: Minimum normal value
                - min_subnormal: Minimum subnormal value
        """
        orig = original.float()
        quant = quantized.float()
        
        # Only compute error on finite values (exclude NaN and Inf)
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
            'max_val': self._get_max_normal(),
            'min_normal': self._get_min_normal(),
            'min_subnormal': self._get_min_subnormal()
        }


# Usage example
if __name__ == "__main__":
    # Create quantizer: E4M3 format (similar to FP8 E4M3)
    quantizer = FloatQuantizer(e_bits=4, m_bits=3)
    
    # Print lookup table for inspection
    quantizer.print_lookup_table(max_entries=256)
    
    # Create test data with various ranges
    test_cases = [
        torch.tensor([0.0, -0.0, 1.0, -1.0]),  # Simple values
        torch.tensor([0.001, 0.01, 0.1, 1.0, 10.0, 100.0]),  # Different scales
        torch.randn(100) * 10,  # Random normal distribution
        torch.tensor([float('nan'), float('inf'), float('-inf')]),  # Special values
    ]
    
    # Test on different devices
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    for device in devices:
        print(f"\n{'#'*80}")
        print(f"Testing on device: {device}")
        print(f"{'#'*80}")
        
        for i, test_tensor in enumerate(test_cases):
            test_tensor = test_tensor.to(device)
            
            print(f"\n{'='*80}")
            print(f"Test Case {i+1}")
            print(f"{'='*80}")
            
            # Quantize
            quantized = quantizer.quantize(test_tensor)
            
            # Statistics
            stats = quantizer.get_stats(test_tensor, quantized)
            print(f"\nStatistics:")
            print(f"  MSE: {stats['mse']:.6e}")
            print(f"  Max error: {stats['max_error']:.6e}")
            print(f"  Mean error: {stats['mean_error']:.6e}")
            
            # Show samples
            print(f"\nSample comparison:")
            print(f"{'Original':<20} {'Quantized':<20} {'Error':<20}")
            print("-"*60)
            for j in range(min(10, len(test_tensor))):
                orig = test_tensor[j].item()
                quant = quantized[j].item()
                error = orig - quant if np.isfinite(orig) and np.isfinite(quant) else 0
                print(f"{orig:<20.10f} {quant:<20.10f} {error:<20.10e}")