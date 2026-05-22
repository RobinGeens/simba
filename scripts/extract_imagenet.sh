#!/bin/bash
set -e
shopt -s nullglob

# Activate project venv so `python` has scipy for reading devkit meta.mat
SIMBA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$SIMBA_ROOT/env/bin/activate"

ROOT="/scratch/leuven/379/vsc37999/imagenet"
TARS="$ROOT/tars"
TRAIN_DIR="$ROOT/train"
VAL_DIR="$ROOT/val"
DEVKIT_DIR="$ROOT/devkit"
LOGFILE="$ROOT/extract.log"

exec > >(tee -a "$LOGFILE") 2>&1

echo "=== ImageNet extraction started at $(date) ==="

# --- [0] Devkit (needed for val ground truth + synset mapping) ---
if [ ! -d "$DEVKIT_DIR/ILSVRC2012_devkit_t12" ]; then
    echo "-> Extracting devkit"
    mkdir -p "$DEVKIT_DIR"
    tar -xzf "$TARS/ILSVRC2012_devkit_t12.tar.gz" -C "$DEVKIT_DIR"
fi

META="$DEVKIT_DIR/ILSVRC2012_devkit_t12/data/meta.mat"
GT="$DEVKIT_DIR/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
[ -f "$GT" ] || { echo "Missing $GT"; exit 1; }

# --- [1] Train set: tar of 1000 inner tars, one per synset ---
if [ ! -f "$ROOT/.train_done" ]; then
    echo "-> Extracting train set into per-synset folders"
    mkdir -p "$TRAIN_DIR"
    cd "$TRAIN_DIR"

    # Unpack the outer tar (creates 1000 nXXXXXXXX.tar files in cwd).
    # Use a marker file rather than glob-checking — nullglob + bare `ls` is fragile.
    if [ ! -f "$ROOT/.train_outer_done" ]; then
        echo "Unpacking outer train tar..."
        tar -xf "$TARS/ILSVRC2012_img_train.tar"
        touch "$ROOT/.train_outer_done"
    fi

    echo "Unpacking 1000 per-synset tars in parallel..."
    # Use find + xargs -P for parallel extraction. find is robust under nullglob.
    find . -maxdepth 1 -name 'n*.tar' -print0 | xargs -0 -P 8 -I {} bash -c '
        archive="{}"
        class="${archive%.tar}"
        mkdir -p "$class"
        tar --no-same-owner -xf "$archive" -C "$class" && rm -f "$archive"
    '

    n_classes=$(find . -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "Train classes extracted: $n_classes"
    [ "$n_classes" -eq 1000 ] || { echo "Expected 1000, got $n_classes"; exit 1; }
    touch "$ROOT/.train_done"
fi

# --- [2] Val set: 50k flat JPEGs -> reorganize into synset folders ---
if [ ! -f "$ROOT/.val_done" ]; then
    echo "-> Extracting val set"
    mkdir -p "$VAL_DIR"
    cd "$VAL_DIR"
    tar -xf "$TARS/ILSVRC2012_img_val.tar"

    echo "Reorganizing val into per-synset folders using devkit ground truth..."
    # Build label-id -> synset mapping by parsing meta.mat via Python.
    python3 - <<'PY'
import os, scipy.io as sio, shutil
root = "/scratch/leuven/379/vsc37999/imagenet"
val_dir = os.path.join(root, "val")
meta = sio.loadmat(os.path.join(root, "devkit/ILSVRC2012_devkit_t12/data/meta.mat"),
                   squeeze_me=True)
# synsets array: fields include ILSVRC2012_ID (1..1000) and WNID (nXXXXXXXX)
id_to_wnid = {int(s["ILSVRC2012_ID"]): str(s["WNID"]) for s in meta["synsets"]
              if int(s["ILSVRC2012_ID"]) <= 1000}
with open(os.path.join(root, "devkit/ILSVRC2012_devkit_t12/data/"
                              "ILSVRC2012_validation_ground_truth.txt")) as f:
    gt = [int(x) for x in f.read().split()]
assert len(gt) == 50000, len(gt)
for wnid in set(id_to_wnid.values()):
    os.makedirs(os.path.join(val_dir, wnid), exist_ok=True)
for i, label_id in enumerate(gt, start=1):
    wnid = id_to_wnid[label_id]
    src = os.path.join(val_dir, f"ILSVRC2012_val_{i:08d}.JPEG")
    dst = os.path.join(val_dir, wnid, f"ILSVRC2012_val_{i:08d}.JPEG")
    if os.path.exists(src):
        os.rename(src, dst)
print("val reorganization done")
PY

    n_val_classes=$(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
    n_val_imgs=$(find "$VAL_DIR" -mindepth 2 -name '*.JPEG' | wc -l)
    echo "Val classes: $n_val_classes (expect 1000)"
    echo "Val images:  $n_val_imgs (expect 50000)"
    [ "$n_val_classes" -eq 1000 ] && [ "$n_val_imgs" -eq 50000 ] \
        || { echo "Val sanity check failed"; exit 1; }
    touch "$ROOT/.val_done"
fi

echo ""
echo "=== Extraction complete at $(date) ==="
echo "Train: $(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l) classes, $(find "$TRAIN_DIR" -mindepth 2 -name '*.JPEG' | wc -l) images"
echo "Val:   $(find "$VAL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l) classes, $(find "$VAL_DIR" -mindepth 2 -name '*.JPEG' | wc -l) images"
