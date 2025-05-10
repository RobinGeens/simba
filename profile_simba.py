from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from simba.simba import simba_l


def count_parameters_by_type(model):
    param_counts = defaultdict(int)
    total = 0

    for module_name, module in model.named_modules():
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        if n_params == 0:
            continue

        # Special case: if part of Mamba, include submodule name
        if "mamba" in module_name:
            parts = module_name.split(".")
            for i in range(len(parts)):
                if parts[i] == "mamba" and i + 1 < len(parts):
                    label = f"Mamba.{parts[i+1]}"
                    break
            else:
                label = "Mamba"
        else:
            label = type(module).__name__

        param_counts[label] += n_params
        total += n_params

    for label, count in sorted(param_counts.items(), key=lambda x: -x[1]):
        print(f"{label:30s} : {count:,} parameters")

    print(f"\nTotal parameters: {total:,}")


def count_operator_flops(model, input_tensor):
    model.eval().to(input_tensor.device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True,
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        with torch.no_grad():
            with record_function("model_inference"):
                model(input_tensor)

    flops_by_op = defaultdict(int)
    for evt in prof.key_averages():
        if evt.flops > 0:
            flops_by_op[evt.key] += evt.flops

    print("\n[Operator FLOP totals]")
    for op, flops in sorted(flops_by_op.items(), key=lambda x: -x[1]):
        print(f"{op:30s} : {flops / 1e9:10.2f} GFLOPs")


# Example usage
if __name__ == "__main__":
    model = simba_l().cuda()
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    count_parameters_by_type(model)
    count_operator_flops(model, dummy_input)
