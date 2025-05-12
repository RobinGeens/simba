"""Sanity check to ensure that the model is using bfloat16 for weights."""

import torch
from timm.models import create_model

from eval import get_most_recent_checkpoint, load_checkpoint

CHECKPOINT_DIR = "checkpoints/simba_l_bf16"

seen_modules = set()


def log_dtype(module, input, output):
    if module.__class__.__name__ not in seen_modules:
        seen_modules.add(module.__class__.__name__)

        # Get all parameter dtypes
        param_dtypes = {}
        for name, param in module.named_parameters():
            param_dtypes[name] = str(param.dtype)

        # Get all buffer dtypes
        buffer_dtypes = {}
        for name, buffer in module.named_buffers():
            buffer_dtypes[name] = str(buffer.dtype)

        print(f"\n{module.__class__.__name__}:")
        print(f"  input dtype={dtype_str(input[0])}")
        print(f"  output dtype={dtype_str(output)}")
        if param_dtypes:
            print("  parameters:")
            for name, dtype in param_dtypes.items():
                print(f"    {name}: {dtype}")
        if buffer_dtypes:
            print("  buffers:")
            for name, dtype in buffer_dtypes.items():
                print(f"    {name}: {dtype}")


def dtype_str(x):
    if hasattr(x, "dtype"):
        return str(x.dtype)
    elif isinstance(x, (tuple, list)):
        return str([dtype_str(xx) for xx in x])
    else:
        return str(type(x))


model: torch.nn.Module = create_model(
    "simba_l_bf16",
    pretrained=False,
    num_classes=1000,
    drop_rate=0.0,
    drop_path_rate=0.0,
    drop_block_rate=None,
)

checkpoint_path = get_most_recent_checkpoint(CHECKPOINT_DIR)
load_checkpoint(model, checkpoint_path)


# model = model.to(dtype=torch.bfloat16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Register hooks to check dtypes
for name, module in model.named_modules():
    module.register_forward_hook(log_dtype)

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224, device=device)

# Run forward pass with autocast
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    output = model(dummy_input)
