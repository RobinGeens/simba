import argparse  # Add this import
import logging
import os

import torch
from timm.models import create_model

from simba.datasets import build_dataset
from simba.engine import evaluate

# Global parameters for datatypes
EVAL_ACTIVATION_DTYPE = "bfloat16"
EVAL_WEIGHT_DTYPE = "bfloat16"

CHECKPOINT_DIR = "checkpoints/simba_l"

# Configuration constants
DATA_PATH = "dataset/ILSVRC2012"
BATCH_SIZE = 256
NUM_WORKERS = 12
PIN_MEMORY = True

dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def get_most_recent_checkpoint(checkpoint_dir):
    try:
        return max(
            [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if ".pth" in f],
            key=os.path.getctime,
        )
    except Exception:
        logging.error(f"No checkpoint fond in directory {checkpoint_dir}")
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
    checkpoint_path = get_most_recent_checkpoint(CHECKPOINT_DIR)
    logging.info(f"Checkpoint found: {checkpoint_path}")

    model: torch.nn.Module = create_model(
        "simba_l",
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=None,
    )

    load_checkpoint(model, checkpoint_path)

    # Set data type
    model = model.to(dtype=dtype_map[EVAL_WEIGHT_DTYPE])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    with torch.amp.autocast("cuda", dtype=dtype_map[EVAL_ACTIVATION_DTYPE]):
        test_stats = evaluate(data_loader, model, device)

    logging.info(
        f"Accuracy of the network on the {len(dataset_val)} test images ({EVAL_WEIGHT_DTYPE} W, {EVAL_ACTIVATION_DTYPE} A):"
    )
    logging.info(f"Top-1 accuracy: {test_stats['acc1']:.1f}%")
    logging.info(f"Top-5 accuracy: {test_stats['acc5']:.1f}%")


if __name__ == "__main__":
    main()
