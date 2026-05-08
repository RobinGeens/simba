import torch
import torch.nn as nn

def compare_matmul_precision(use_bias=False):
    torch.manual_seed(42)
    
    batch_size, in_features, out_features = 32, 4096, 1024
    
    # Create test data
    x_fp32 = torch.randn(batch_size, in_features, device='cuda', dtype=torch.float32)
    weight_fp32 = torch.randn(out_features, in_features, device='cuda', dtype=torch.float32)
    bias_fp32 = torch.randn(out_features, device='cuda', dtype=torch.float32) if use_bias else None
    
    # Convert to BF16
    x_bf16 = x_fp32.to(torch.bfloat16)
    weight_bf16 = weight_fp32.to(torch.bfloat16)
    bias_bf16 = bias_fp32.to(torch.bfloat16) if use_bias else None
    
    # 1. FP32 baseline
    if use_bias:
        output_fp32 = torch.matmul(x_fp32, weight_fp32.t()) + bias_fp32
    else:
        output_fp32 = torch.matmul(x_fp32, weight_fp32.t())
    
    # 2. Method 1: nn.functional.linear
    output_method1 = torch.nn.functional.linear(x_bf16, weight_bf16, bias_bf16)
    
    # 3. Method 2: torch.matmul
    if use_bias:
        output_method2 = torch.matmul(x_bf16, weight_bf16.t()) + bias_bf16
    else:
        output_method2 = torch.matmul(x_bf16, weight_bf16.t())
    
    # 4. Method 3: @ operator
    if use_bias:
        output_method3 = (x_bf16 @ weight_bf16.t()) + bias_bf16
    else:
        output_method3 = x_bf16 @ weight_bf16.t()
    
    # 5. Manual BF16 accumulation (element-wise, truly BF16)
    output_manual_bf16 = torch.zeros(batch_size, out_features, device='cuda', dtype=torch.bfloat16)
    for i in range(in_features):
        contribution = (x_bf16[:, i:i+1] * weight_bf16[:, i:i+1].t()).to(torch.bfloat16)
        output_manual_bf16 = (output_manual_bf16 + contribution).to(torch.bfloat16)
    if use_bias:
        output_manual_bf16 = (output_manual_bf16 + bias_bf16).to(torch.bfloat16)
    
    # Convert all to FP32 for comparison
    output_fp32 = output_fp32.float()
    output_method1 = output_method1.float()
    output_method2 = output_method2.float()
    output_method3 = output_method3.float()
    output_manual_bf16 = output_manual_bf16.float()
    
    # Calculate errors relative to FP32 baseline
    error_method1 = torch.abs(output_fp32 - output_method1)
    error_method2 = torch.abs(output_fp32 - output_method2)
    error_method3 = torch.abs(output_fp32 - output_method3)
    error_manual = torch.abs(output_fp32 - output_manual_bf16)
    
    # Print results
    bias_str = "WITH bias" if use_bias else "WITHOUT bias"
    print("=" * 70)
    print(f"Matrix Multiplication Precision Comparison ({bias_str})")
    print("=" * 70)
    print(f"Shape: [{batch_size}, {in_features}] @ [{in_features}, {out_features}]")
    print()
    
    print("Errors relative to FP32 baseline:")
    print()
    
    methods = [
        ("Method 1: F.linear", error_method1),
        ("Method 2: torch.matmul", error_method2),
        ("Method 3: @ operator", error_method3),
        ("Manual BF16 accumulation", error_manual)
    ]
    
    for method_name, error in methods:
        print(f"{method_name}:")
        print(f"  Max error:      {error.max().item():.6e}")
        print(f"  Mean error:     {error.mean().item():.6e}")
        print(f"  Relative error: {(error / (torch.abs(output_fp32) + 1e-8)).mean().item():.6e}")
        print()
    
    print("=" * 70)
    print()
    
    return {
        'output_fp32': output_fp32,
        'output_method1': output_method1,
        'output_method2': output_method2,
        'output_method3': output_method3,
        'output_manual': output_manual_bf16,
        'error_method1': error_method1,
        'error_method2': error_method2,
        'error_method3': error_method3,
        'error_manual': error_manual
    }

if __name__ == "__main__":
    print("\n")
    print("EXPERIMENT 1: Matrix multiplication WITHOUT bias")
    print("\n")
    results_no_bias = compare_matmul_precision(use_bias=False)
    
    print("\n")
    print("EXPERIMENT 2: Matrix multiplication WITH bias")
    print("\n")
    results_with_bias = compare_matmul_precision(use_bias=True)