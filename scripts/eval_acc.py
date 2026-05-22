import argparse
import logging
import os
import sys

# Mirror main.py's import convention: put simba/ on sys.path so its modules can be
# imported by bare name (they use bare imports internally, e.g. `from quantizer import ...`).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "simba")))
import torch
from timm.models import create_model

from simba_bf16 import SiMBA  # noqa: F401
import simba_configs  # noqa: F401  - registers simba_l_bf16 etc. with timm
from qlinear import replace_linear_with_qlinear  # noqa: F401

if __name__ == "__main__":
    from datasets import build_dataset
    from engine import evaluate

################# CONFIG #################b

MODEL_NAME = "simba_s_bf16"
RUN_NAME = "simba_s_bf16"
BEST_CHECKPOINT = 0  
GPU_NODE = 0  # 　either 0 or 1 for which GPU to use on a node

############### CONFIG END ###############

# Configuration constants
DATA_PATH = "/scratch/leuven/379/vsc37999/imagenet"
BATCH_SIZE = 256
NUM_WORKERS = 12
PIN_MEMORY = True


def get_checkpoint():
    checkpoint_dir = os.path.join("checkpoints", RUN_NAME)

    if BEST_CHECKPOINT is not None:
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
    # model = replace_linear_with_qlinear(model, verbose=True)


    device = torch.device(f"cuda:{GPU_NODE}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create dataset and dataloader
    Args = type(
        "Args",
        (),
        {
            "data_path": DATA_PATH,
            "data_set": "IMNET",
            "use_mcloader": False,
            "input_size": 224,
            "color_jitter": None,
            "aa": None,
            "train_interpolation": "bicubic",
            "reprob": 0.0,
            "remode": None,
            "recount": 1,
        },
    )
    dataset_val, _ = build_dataset(is_train=False, args=Args)

    data_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False
    )

    # Evaluate
    test_stats = evaluate(data_loader, model, device, eval_one_sample=False)

    logging.info(
        f"Accuracy of the network on the {len(dataset_val)} test images ({model.AUTOCAST_T}):"
    )
    logging.info(f"Top-1 accuracy: {test_stats['acc1']:.1f}%")
    logging.info(f"Top-5 accuracy: {test_stats['acc5']:.1f}%")


if __name__ == "__main__":
    main()
