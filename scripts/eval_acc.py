import argparse
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import torch
import torch.amp
from timm.models import create_model

from simba.simba import simba_l  # noqa: F401
from simba.simba_bf16 import BF16, FP32, simba_l_bf16  # noqa: F401

if __name__ == "__main__":
    from simba.datasets import build_dataset
    from simba.engine import evaluate

# Global parameters for datatypes
EVAL_AUTOCAST_TYPE = FP32
EVAL_WEIGHT_DTYPE = BF16

MODEL_NAME = "simba_l_bf16"
CHECKPOINT_DIR = "checkpoints/simba_l_bf16_B"
BEST_CHECKPOINT = "checkpoints/simba_l_bf16_B/checkpoint-316.pth.tar"  # Should be 83.0%

# Configuration constants
DATA_PATH = "dataset/ILSVRC2012"
BATCH_SIZE = 256
NUM_WORKERS = 12
PIN_MEMORY = True


def get_checkpoint(checkpoint_dir):
    if BEST_CHECKPOINT:
        return BEST_CHECKPOINT
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


def main(checkpoint_path=None):
    logging.basicConfig(level=logging.INFO)
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint(CHECKPOINT_DIR)
    logging.info(f"Using checkpoint: {checkpoint_path}")

    model: torch.nn.Module = create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=None,
    )

    load_checkpoint(model, checkpoint_path)

    # Set data type # TODO this doesn't work because sensitive weights (e.g. batch norm) must stay in FP32
    # model = model.to(dtype=EVAL_WEIGHT_DTYPE)

    # Explicitly use GPU 1
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    with torch.amp.autocast("cuda", dtype=EVAL_AUTOCAST_TYPE):
        test_stats = evaluate(data_loader, model, device)

    logging.info(
        f"Accuracy of the network on the {len(dataset_val)} test images ({EVAL_WEIGHT_DTYPE} W, {EVAL_AUTOCAST_TYPE} A):"
    )
    logging.info(f"Top-1 accuracy: {test_stats['acc1']:.1f}%")
    logging.info(f"Top-5 accuracy: {test_stats['acc5']:.1f}%")


def get_checkpoints(checkpoint_dir, min_id=300):
    checkpoint_pattern = re.compile(r"^checkpoint-(\d+)\.pth\.tar$")
    checkpoint_files = []
    for fname in os.listdir(checkpoint_dir):
        match = checkpoint_pattern.match(fname)
        if match:
            checkpoint_id = int(match.group(1))
            if checkpoint_id > min_id:
                checkpoint_files.append(os.path.join(checkpoint_dir, fname))
    return sorted(checkpoint_files)


if __name__ == "__main__":
    # checkpoints = get_checkpoints(CHECKPOINT_DIR, min_id=300)
    # for checkpoint in checkpoints:
    #     main(checkpoint)
    main()
