#!/bin/bash
set -euo pipefail
shopt -s nullglob

# Resolve repo root (the dir containing the tars = parent of this scripts/ dir).
# Using absolute paths avoids the compounding-`cd` bugs that broke earlier runs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Log file
LOGFILE="$ROOT/prepare_imagenet.log"
exec > >(tee -a "$LOGFILE") 2>&1

# === CONFIGURATION ===
# Tar files downloaded from https://image-net.org/download-images
TRAIN_TAR="$ROOT/ILSVRC2012_img_train.tar"
VAL_TAR="$ROOT/ILSVRC2012_img_val.tar"

# Output dirs
OUTPUT_DIR="$ROOT/dataset/ILSVRC2012"
TRAIN_DIR="$OUTPUT_DIR/train"
VAL_DIR="$OUTPUT_DIR/val"

echo "=== Extracting ILSVRC2012 ==="

# === [1] Extract Training Set ===
echo "-> Step 1: Preparing training set"
existing_train=0
[ -d "$TRAIN_DIR" ] && existing_train=$(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [ "$existing_train" -ge 1000 ]; then
    echo "✅ Training set already prepared ($existing_train classes) — skipping."
else
    mkdir -p "$TRAIN_DIR"

    # Extract the outer tar IN PLACE (no cp of the 147 GB file) -> 1000 class tarballs.
    echo "Untarring outer $TRAIN_TAR into $TRAIN_DIR ..."
    tar -xf "$TRAIN_TAR" -C "$TRAIN_DIR"

    # Extract each per-class tarball into its own folder, removing the tar as we go
    # so peak disk usage stays ~one dataset-size instead of doubling.
    echo "Extracting per-class tarballs..."
    for archive in "$TRAIN_DIR"/*.tar; do
        class_name="$(basename "${archive%.tar}")"
        mkdir -p "$TRAIN_DIR/$class_name"
        tar --no-same-owner -xf "$archive" -C "$TRAIN_DIR/$class_name"
        rm -f "$archive"
    done
    echo "✅ Training set prepared."
fi

# === [2] Extract Validation Set ===
echo "-> Step 2: Preparing validation set"
existing_val=0
[ -d "$VAL_DIR" ] && existing_val=$(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [ "$existing_val" -ge 1000 ]; then
    echo "✅ Validation set already prepared ($existing_val classes) — skipping."
else
    mkdir -p "$VAL_DIR"

    echo "Untarring $VAL_TAR ..."
    tar -xf "$VAL_TAR" -C "$VAL_DIR"

    echo "Downloading valprep script..."
    wget -q https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh -O "$VAL_DIR/valprep.sh"

    echo "Organizing validation images into class folders..."
    ( cd "$VAL_DIR" && bash valprep.sh && rm -f valprep.sh )
    echo "✅ Validation set prepared."
fi

# === DONE ===
echo "🎉 ImageNet extraction complete!"
echo "Train classes: $(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "Val classes:   $(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
