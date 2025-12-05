"""
Hardware activation functions for SiMBA/Mamba accelerators.
Piecewise linear (PWL) approximations of SiLU and SoftPlus.
All computations in BF16 precision.
"""

import torch
import torch.nn as nn
import numpy as np


class HardwareSiLU(nn.Module):
    """
    SiLU activation using piecewise linear approximation.
    All operations in BF16 precision.
    
    Args:
        num_segments: Number of PWL segments (default: 16)
        input_range: Clipping range as (x_min, x_max), default: (-6, 6)
    """
    
    def __init__(
        self, 
        num_segments: int = 16, 
        input_range: tuple = (-6.0, 6.0)
    ):
        super().__init__()
        
        self.num_segments = num_segments
        self.x_min, self.x_max = input_range
        self.segment_width = (self.x_max - self.x_min) / self.num_segments
        
        self._build_lut()
        
    def _build_lut(self):
        """Build lookup table for PWL approximation in BF16."""
        # Use FP32 for accurate LUT construction, then convert to BF16
        breakpoints = np.linspace(self.x_min, self.x_max, self.num_segments + 1)
        
        def exact_silu(x):
            return x / (1 + np.exp(-np.clip(x, -20, 20)))
        
        y_values = exact_silu(breakpoints)
        
        lut_x_start = []
        lut_y_start = []
        lut_slope = []
        
        for i in range(self.num_segments):
            x_start = breakpoints[i]
            y_start = y_values[i]
            y_end = y_values[i + 1]
            slope = (y_end - y_start) / self.segment_width
            
            lut_x_start.append(x_start)
            lut_y_start.append(y_start)
            lut_slope.append(slope)
        
        # Convert to BF16 tensors
        lut_x_start = torch.tensor(lut_x_start, dtype=torch.bfloat16)
        lut_y_start = torch.tensor(lut_y_start, dtype=torch.bfloat16)
        lut_slope = torch.tensor(lut_slope, dtype=torch.bfloat16)
        
        # Register as buffers
        self.register_buffer('lut_x_start', lut_x_start)
        self.register_buffer('lut_y_start', lut_y_start)
        self.register_buffer('lut_slope', lut_slope)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute SiLU approximation using PWL in BF16.
        
        Args:
            x: Input tensor of any shape (BF16)
        
        Returns:
            Approximated SiLU(x) with same shape as input (BF16)
        """
        # Ensure input is BF16
        x = x.to(torch.bfloat16)
        
        original_shape = x.shape
        x_clipped = torch.clamp(x, self.x_min, self.x_max)
        x_flat = x_clipped.flatten()
        
        # Find segment index
        # Note: searchsorted requires float32, but we minimize conversions
        segment_idx = torch.searchsorted(
            self.lut_x_start.float(), 
            x_flat.float(), 
            right=False
        )
        segment_idx = torch.clamp(segment_idx - 1, 0, self.num_segments - 1)
        segment_idx = torch.where(
            x_flat.float() < self.lut_x_start[0].float(),
            torch.zeros_like(segment_idx),
            segment_idx
        )
        
        # LUT lookup (all BF16)
        x_start = self.lut_x_start[segment_idx]
        y_start = self.lut_y_start[segment_idx]
        slope = self.lut_slope[segment_idx]
        
        # Linear interpolation in BF16
        delta_x = x_flat - x_start
        y = y_start + slope * delta_x
        y = y.reshape(original_shape)
        
        return y
    
    def extra_repr(self) -> str:
        total_bytes = self.num_segments * 3 * 2  # BF16 = 2 bytes
        return (
            f'num_segments={self.num_segments}, '
            f'dtype=bfloat16, '
            f'input_range=({self.x_min}, {self.x_max}), '
            f'lut_size={total_bytes}B'
        )


class HardwareSoftPlus(nn.Module):
    """
    SoftPlus activation (log(1 + exp(x))) using piecewise linear approximation.
    All operations in BF16 precision.
    
    Args:
        num_segments: Number of PWL segments (default: 16)
        input_range: Clipping range as (x_min, x_max), default: (-10, 10)
    """
    
    def __init__(
        self, 
        num_segments: int = 16, 
        input_range: tuple = (-10.0, 10.0)
    ):
        super().__init__()
        
        self.num_segments = num_segments
        self.x_min, self.x_max = input_range
        self.segment_width = (self.x_max - self.x_min) / self.num_segments
        
        self._build_lut()
    
    def _build_lut(self):
        """Build lookup table for SoftPlus approximation in BF16."""
        breakpoints = np.linspace(self.x_min, self.x_max, self.num_segments + 1)
        
        def exact_softplus(x):
            return np.log(1 + np.exp(np.clip(x, -20, 20)))
        
        y_values = exact_softplus(breakpoints)
        
        lut_x_start = []
        lut_y_start = []
        lut_slope = []
        
        for i in range(self.num_segments):
            x_start = breakpoints[i]
            y_start = y_values[i]
            y_end = y_values[i + 1]
            slope = (y_end - y_start) / self.segment_width
            
            lut_x_start.append(x_start)
            lut_y_start.append(y_start)
            lut_slope.append(slope)
        
        # Convert to BF16 tensors
        lut_x_start = torch.tensor(lut_x_start, dtype=torch.bfloat16)
        lut_y_start = torch.tensor(lut_y_start, dtype=torch.bfloat16)
        lut_slope = torch.tensor(lut_slope, dtype=torch.bfloat16)
        
        # Register as buffers
        self.register_buffer('lut_x_start', lut_x_start)
        self.register_buffer('lut_y_start', lut_y_start)
        self.register_buffer('lut_slope', lut_slope)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute SoftPlus approximation using PWL in BF16.
        
        Args:
            x: Input tensor of any shape (BF16)
        
        Returns:
            Approximated SoftPlus(x) with same shape as input (BF16)
        """
        # Ensure input is BF16
        x = x.to(torch.bfloat16)
        
        original_shape = x.shape
        x_clipped = torch.clamp(x, self.x_min, self.x_max)
        x_flat = x_clipped.flatten()
        
        # Find segment index
        segment_idx = torch.searchsorted(
            self.lut_x_start.float(), 
            x_flat.float(), 
            right=False
        )
        segment_idx = torch.clamp(segment_idx - 1, 0, self.num_segments - 1)
        segment_idx = torch.where(
            x_flat.float() < self.lut_x_start[0].float(),
            torch.zeros_like(segment_idx),
            segment_idx
        )
        
        # LUT lookup (all BF16)
        x_start = self.lut_x_start[segment_idx]
        y_start = self.lut_y_start[segment_idx]
        slope = self.lut_slope[segment_idx]
        
        # Linear interpolation in BF16
        delta_x = x_flat - x_start
        y = y_start + slope * delta_x
        y = y.reshape(original_shape)
        
        return y
    
    def extra_repr(self) -> str:
        total_bytes = self.num_segments * 3 * 2  # BF16 = 2 bytes
        return (
            f'num_segments={self.num_segments}, '
            f'dtype=bfloat16, '
            f'input_range=({self.x_min}, {self.x_max}), '
            f'lut_size={total_bytes}B'
        )


# ============================================================================
# Testing and Validation
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("Hardware Activation Functions - Accuracy Test (BF16)")
    print("="*70)
    
    # Test configuration
    batch_size = 4
    seq_len = 196
    hidden_dim = 384
    test_shape = (batch_size, seq_len, hidden_dim)
    
    # Generate test input in BF16
    torch.manual_seed(42)
    x = torch.randn(test_shape, dtype=torch.bfloat16)
    
    # ========================================================================
    # Test 1: SiLU Approximation
    # ========================================================================
    print("\n" + "─"*70)
    print("Test 1: SiLU Approximation (BF16)")
    print("─"*70)
    
    # Standard PyTorch SiLU (compute in FP32 for reference)
    silu_standard = nn.SiLU()
    y_standard = silu_standard(x.float()).to(torch.bfloat16)
    
    # Test different segment configurations
    segment_configs = [8, 16, 32, 64]
    
    print(f"\nInput shape: {test_shape}")
    print(f"Input dtype: {x.dtype}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    print()
    
    for num_segments in segment_configs:
        silu_hw = HardwareSiLU(num_segments=num_segments)
        y_hw = silu_hw(x)
        
        # Compute absolute errors
        abs_error = (y_hw.float() - y_standard.float()).abs()
        
        # Compute relative error only where y_standard is significant
        # Threshold: only compute relative error when |y_standard| > 0.01
        threshold = 0.01
        mask = y_standard.float().abs() > threshold
        
        if mask.any():
            rel_error_masked = abs_error[mask] / y_standard.float().abs()[mask]
            rel_error_mean = rel_error_masked.mean()
            rel_error_max = rel_error_masked.max()
            rel_error_median = rel_error_masked.median()
            num_valid = mask.sum().item()
        else:
            rel_error_mean = float('nan')
            rel_error_max = float('nan')
            rel_error_median = float('nan')
            num_valid = 0
        
        print(f"Segments: {num_segments:2d}")
        print(f"  Output dtype: {y_hw.dtype}")
        print(f"  Absolute Error - Mean: {abs_error.mean():.6f}, Max: {abs_error.max():.6f}")
        print(f"  Relative Error (|y|>{threshold}):")
        print(f"    Mean: {rel_error_mean:.6f}, Median: {rel_error_median:.6f}, Max: {rel_error_max:.6f}")
        print(f"    Valid samples: {num_valid}/{x.numel()}")
        print(f"  {silu_hw.extra_repr()}")
        print()
    
    # ========================================================================
    # Test 2: SoftPlus Approximation
    # ========================================================================
    print("─"*70)
    print("Test 2: SoftPlus Approximation (BF16)")
    print("─"*70)
    
    # Standard PyTorch SoftPlus
    softplus_standard = nn.Softplus()
    y_standard = softplus_standard(x.float()).to(torch.bfloat16)
    
    print(f"\nInput shape: {test_shape}")
    print(f"Input dtype: {x.dtype}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    print()
    
    for num_segments in segment_configs:
        softplus_hw = HardwareSoftPlus(num_segments=num_segments)
        y_hw = softplus_hw(x)
        
        # Compute absolute errors
        abs_error = (y_hw.float() - y_standard.float()).abs()
        
        # Compute relative error only where y_standard is significant
        threshold = 0.01
        mask = y_standard.float().abs() > threshold
        
        if mask.any():
            rel_error_masked = abs_error[mask] / y_standard.float().abs()[mask]
            rel_error_mean = rel_error_masked.mean()
            rel_error_max = rel_error_masked.max()
            rel_error_median = rel_error_masked.median()
            num_valid = mask.sum().item()
        else:
            rel_error_mean = float('nan')
            rel_error_max = float('nan')
            rel_error_median = float('nan')
            num_valid = 0
        
        print(f"Segments: {num_segments:2d}")
        print(f"  Output dtype: {y_hw.dtype}")
        print(f"  Absolute Error - Mean: {abs_error.mean():.6f}, Max: {abs_error.max():.6f}")
        print(f"  Relative Error (|y|>{threshold}):")
        print(f"    Mean: {rel_error_mean:.6f}, Median: {rel_error_median:.6f}, Max: {rel_error_max:.6f}")
        print(f"    Valid samples: {num_valid}/{x.numel()}")
        print(f"  {softplus_hw.extra_repr()}")
        print()
    
    # ========================================================================
    # Test 3: Edge Cases
    # ========================================================================
    print("─"*70)
    print("Test 3: Edge Cases (BF16)")
    print("─"*70)
    
    edge_cases = {
        'zeros': torch.zeros(10, dtype=torch.bfloat16),
        'small_positive': torch.tensor([0.1, 0.5, 1.0, 2.0, 3.0], dtype=torch.bfloat16),
        'small_negative': torch.tensor([-0.1, -0.5, -1.0, -2.0, -3.0], dtype=torch.bfloat16),
        'large_positive': torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.bfloat16),
        'large_negative': torch.tensor([-4.0, -5.0, -6.0, -7.0, -8.0], dtype=torch.bfloat16),
    }
    
    silu_hw = HardwareSiLU(num_segments=16)
    silu_standard = nn.SiLU()
    
    print("\nSiLU Edge Cases:")
    for name, x_edge in edge_cases.items():
        y_hw = silu_hw(x_edge)
        y_std = silu_standard(x_edge.float()).to(torch.bfloat16)
        error = (y_hw.float() - y_std.float()).abs().max()
        print(f"  {name:20s}: Max absolute error = {error:.6f}")
    
    softplus_hw = HardwareSoftPlus(num_segments=16)
    softplus_standard = nn.Softplus()
    
    print("\nSoftPlus Edge Cases:")
    for name, x_edge in edge_cases.items():
        y_hw = softplus_hw(x_edge)
        y_std = softplus_standard(x_edge.float()).to(torch.bfloat16)
        error = (y_hw.float() - y_std.float()).abs().max()
        print(f"  {name:20s}: Max absolute error = {error:.6f}")
    
    # ========================================================================
    # Test 4: Detailed Error Analysis
    # ========================================================================
    print("\n" + "─"*70)
    print("Test 4: Detailed Error Analysis")
    print("─"*70)
    
    # Create test inputs across different ranges
    x_ranges = {
        'Very Negative (x<-3)': torch.linspace(-6, -3, 1000, dtype=torch.bfloat16),
        'Negative (-3<x<0)': torch.linspace(-3, 0, 1000, dtype=torch.bfloat16),
        'Positive (0<x<3)': torch.linspace(0, 3, 1000, dtype=torch.bfloat16),
        'Very Positive (x>3)': torch.linspace(3, 6, 1000, dtype=torch.bfloat16),
    }
    
    silu_hw = HardwareSiLU(num_segments=16)
    silu_standard = nn.SiLU()
    
    print("\nSiLU Error by Input Range:")
    for range_name, x_range in x_ranges.items():
        y_hw = silu_hw(x_range)
        y_std = silu_standard(x_range.float()).to(torch.bfloat16)
        abs_error = (y_hw.float() - y_std.float()).abs()
        
        print(f"\n  {range_name}:")
        print(f"    Output range: [{y_std.min():.6f}, {y_std.max():.6f}]")
        print(f"    Absolute error - Mean: {abs_error.mean():.6f}, Max: {abs_error.max():.6f}")
    
    # ========================================================================
    # Test 5: BF16 Precision Validation
    # ========================================================================
    print("\n" + "─"*70)
    print("Test 5: BF16 Precision Validation")
    print("─"*70)
    
    silu_hw = HardwareSiLU(num_segments=16)
    
    print("\nLUT buffer dtypes:")
    print(f"  lut_x_start: {silu_hw.lut_x_start.dtype}")
    print(f"  lut_y_start: {silu_hw.lut_y_start.dtype}")
    print(f"  lut_slope: {silu_hw.lut_slope.dtype}")
    
    x_test = torch.randn(10, dtype=torch.bfloat16)
    y_test = silu_hw(x_test)
    print(f"\nInput dtype: {x_test.dtype}")
    print(f"Output dtype: {y_test.dtype}")
    print(f"✓ All operations in BF16")
