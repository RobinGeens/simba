import argparse
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.amp
from timm.models import create_model

from simba.simba import SiMBA, simba_l, MambaLayer  # noqa: F401
from simba.simba_bf16 import BF16, FP32, simba_l_fp16  # ,  simba_l_bf16     # noqa: F401

if __name__ == "__main__":
    from simba.datasets import build_dataset
    from simba.engine import evaluate

################# CONFIG #################

MODEL_NAME = "simba_l_bf16" # "simba_l_fp16"
RUN_NAME = "simba_l_bf16" # "simba_l_bf16_B"
BEST_CHECKPOINT = 316  # Should be 83.0% Top-1 acc

############### CONFIG END ###############


# Global parameters for datatypes
EVAL_WEIGHT_DTYPE = None  # Not used

# Configuration constants
GPU_NODE = 0 #　either 0 or 1 for which GPU to use on a node
DATA_PATH = "/users/micas/rgeens/Public/dataset/ILSVRC2012" # "/volume1/users/rgeens/simba/dataset/ILSVRC2012"
BATCH_SIZE = 256
NUM_WORKERS = 12
PIN_MEMORY = True


def get_checkpoint():
    checkpoint_dir = os.path.join("checkpoints", RUN_NAME)

    if BEST_CHECKPOINT:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{BEST_CHECKPOINT}.pth.tar")
        if os.path.isfile(checkpoint_path):
            return checkpoint_path
        else:
            logging.error(f"Best checkpoint {BEST_CHECKPOINT} not found in {checkpoint_dir}")

    try:
        return max(
            [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if ".pth" in f],
            key=os.path.getctime,
        )
    except Exception:
        logging.error(f"No checkpoint found in directory {checkpoint_dir}")
        return


def load_checkpoint(model, checkpoint_path):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        torch.serialization.add_safe_globals([argparse.Namespace])  # Allow loading Namespace
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]["state_dict"]
        else:
            state_dict = checkpoint["state_dict"]

        # Remove module prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module") else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logging.error(f"No checkpoint found at {checkpoint_path}")
        # raise FileNotFoundError()


def main():
    logging.basicConfig(level=logging.INFO)
    checkpoint_path = get_checkpoint()
    logging.info(f"Using checkpoint: {checkpoint_path}")

    model: SiMBA = create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=None,
    )

    load_checkpoint(model, checkpoint_path)
    
    print(model)

    # Set data type # TODO this doesn't work because sensitive weights (e.g. batch norm) must stay in FP32
    # model = model.to(dtype=EVAL_WEIGHT_DTYPE)

    device = torch.device(f"cuda:{GPU_NODE}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Observe the model structure and layer names
    # def list_all_modules(model):
    #     """List all modules in the model."""
    #     print("="*80)
    #     for name, module in model.named_modules():
    #         # if 'block' in name.lower() or 'proj' in name.lower():
    #         # if 'in_proj' in name.lower():
    #         if name.endswith('in_proj') and 'block' in name:
    #             print(f"{name:60s} {type(module).__name__}")
    #     print("="*80)

    # list_all_modules(model)
    
    
    # Hooks to log input/output shapes and weight shapes of in_proj layers
    hook_results = {}
    hooks = []

    def create_hook(name):
        def hook_fn(module, input, output):
            print(f"!!! HOOK TRIGGERED: {name} !!!")
            hook_results[name] = {
                'input_shape': input[0].shape if isinstance(input, tuple) else input.shape,
                'output_shape': output[0].shape if isinstance(output, tuple) else output.shape,
                # 'weight_shape': module.weight.shape,
            }
        return hook_fn

    # register hooks
    for name, module in model.named_modules():
        if True:
        # Note that the implementation of Mamba layers cannot be hooked because they directly use functional calls instead of nn.Module layers
        # if name.endswith('in_proj') and 'block' in name:
            print(name, type(module))
        # if isinstance(module, MambaLayer) and 'block' in name:
            handle = module.register_forward_hook(create_hook(name))
            hooks.append(handle)
    print(f"{len(hooks)} hooks registered.\n")

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            _ = model(torch.randn(1, 3, 224, 224).to(device))
        
    # Print hook results
    print("\n" + "="*80)
    print("Hook Results:")
    print("="*80)
    for name, info in hook_results.items():
        # print(f"{name}: {info['input_shape']} → {info['output_shape']} | W: {info['weight_shape']}")
        print(f"{name}: {info['input_shape']} → {info['output_shape']}")
    print("="*80)
    
    # Clean hooks
    for h in hooks:
        h.remove()

    # Create dataset and dataloader
    # Args = type(
    #     "Args",
    #     (),
    #     {
    #         "data_path": DATA_PATH,
    #         "data_set": "IMNET",
    #         "use_mcloader": False,
    #         "input_size": 224,
    #         "color_jitter": None,
    #         "aa": None,
    #         "train_interpolation": "bicubic",
    #         "reprob": 0.0,
    #         "remode": None,
    #         "recount": 1,
    #     },
    # )
    # dataset_val, _ = build_dataset(is_train=False, args=Args)

    # data_loader = torch.utils.data.DataLoader(
    #     dataset_val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False
    # )

    # # Evaluate
    # test_stats = evaluate(data_loader, model, device)

    # logging.info(
    #     f"Accuracy of the network on the {len(dataset_val)} test images ({EVAL_WEIGHT_DTYPE} W, {model.AUTOCAST_T} A):"
    # )
    # logging.info(f"Top-1 accuracy: {test_stats['acc1']:.1f}%")
    # logging.info(f"Top-5 accuracy: {test_stats['acc5']:.1f}%")


if __name__ == "__main__":
    main()
