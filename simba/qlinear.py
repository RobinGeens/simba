import torch
import torch.nn as nn
from typing import Optional

def replace_linear_with_qlinear(model, verbose=True):
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            modules_to_replace.append(name)
    
    for name in modules_to_replace:
        *parent_path, attr_name = name.split('.')
        parent = model
        for part in parent_path:
            parent = getattr(parent, part)
        
        linear_module = getattr(parent, attr_name)
        qlinear = QLinear.from_linear(linear_module)
        setattr(parent, attr_name, qlinear)
        
        if verbose:
            print(f"Replaced {name}")
    
    return model

class QLinear(nn.Module):
    """
    A quantized linear layer that performs true BF16 accumulation.
    
    This module can be used as a drop-in replacement for torch.nn.Linear,
    but performs matrix multiplication with bfloat16 precision throughout,
    including intermediate accumulations.
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If True, adds a learnable bias to the output
        device: Device to place the parameters on
        dtype: Data type for the parameters (default: torch.bfloat16)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight and bias parameters in BF16
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with true BF16 accumulation.
        
        Args:
            x: Input tensor of shape (..., in_features)
        
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Convert input to BF16 if needed
        x_bf16 = x.to(torch.bfloat16)
        
        # Get batch dimensions
        original_shape = x_bf16.shape
        batch_dims = original_shape[:-1]
        
        # Flatten batch dimensions for processing
        x_flat = x_bf16.view(-1, self.in_features)
        batch_size = x_flat.shape[0]
        
        # Initialize output accumulator in BF16
        output = torch.zeros(
            batch_size, self.out_features,
            device=x_bf16.device,
            dtype=torch.bfloat16
        )
        
        # Manual BF16 accumulation (element-wise)
        for i in range(self.in_features):
            contribution = (x_flat[:, i:i+1] * self.weight[:, i:i+1].t()).to(torch.bfloat16)
            output = (output + contribution).to(torch.bfloat16)
        
        # Add bias if present
        if self.bias is not None:
            output = (output + self.bias).to(torch.bfloat16)
        
        # Reshape output to match input batch dimensions
        output = output.view(*batch_dims, self.out_features)
        
        return output
    
    def extra_repr(self) -> str:
        """String representation for print()."""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
    
    @classmethod
    def from_linear(cls, linear_module: nn.Linear, device: Optional[torch.device] = None) -> 'QLinear':
        """
        Convert a standard nn.Linear module to QLinear.
        
        Args:
            linear_module: The nn.Linear module to convert
            device: Optional device to move the module to
        
        Returns:
            A QLinear module with the same parameters
        """
        if device is None:
            device = linear_module.weight.device
        
        qlinear = cls(
            in_features=linear_module.in_features,
            out_features=linear_module.out_features,
            bias=linear_module.bias is not None,
            device=device,
            dtype=torch.bfloat16
        )
        
        # Copy weights
        qlinear.weight.data = linear_module.weight.data.to(torch.bfloat16)
        if linear_module.bias is not None:
            qlinear.bias.data = linear_module.bias.data.to(torch.bfloat16)
        
        return qlinear


# Example usage and comparison
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create a standard linear layer
    linear = nn.Linear(4096, 4096, bias=True).cuda()
    
    # Convert to QLinear
    qlinear = QLinear.from_linear(linear)
    
    # Test input
    x = torch.randn(32, 4096, device='cuda', dtype=torch.float32)
    
    # Forward pass
    output_linear = linear(x)
    output_qlinear = qlinear(x)
    
    # Compare
    print(f"Standard Linear output dtype: {output_linear.dtype}")
    print(f"QLinear output dtype: {output_qlinear.dtype}")
    print(f"Max difference: {torch.abs(output_linear.float() - output_qlinear.float()).max().item():.6e}")
    print(f"Mean difference: {torch.abs(output_linear.float() - output_qlinear.float()).mean().item():.6e}")