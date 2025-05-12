#!/bin/bash
set -e
shopt -s nullglob

# Log file
LOGFILE="prepare_imagenet.log"
exec > >(tee -a "$LOGFILE") 2>&1

# === CONFIGURATION ===
# Set to where the tar files are. Should be downloaded from https://image-net.org/download-images
TRAIN_TAR="ILSVRC2012_img_train.tar"
VAL_TAR="ILSVRC2012_img_val.tar"

# Set desired output dirs
OUTPUT_DIR="dataset/ILSVRC2012/"
TRAIN_DIR="$OUTPUT_DIR/train"
VAL_DIR="$OUTPUT_DIR/val"

echo "=== Extracting ILSVRC2012 ==="

# === [1] Extract Training Set ===
echo "-> Step 1: Preparing training set"

# mkdir -p "$TRAIN_DIR"
# cp "$TRAIN_TAR" "$TRAIN_DIR/"
# cd "$TRAIN_DIR"

# echo "Untarring $TRAIN_TAR..."
# tar -xf "$TRAIN_TAR"
# rm -f "$TRAIN_TAR"

# # Handle possible top-level folder (e.g. ILSVRC2012_img_train)
# if [ -d ILSVRC2012_img_train ]; then
#     echo "Found ILSVRC2012_img_train/, moving contents up..."
#     mv ILSVRC2012_img_train/* .
#     rmdir ILSVRC2012_img_train
# fi

# echo "Extracting class tarballs..."
# for archive in *.tar; do
#     class_name="${archive%.tar}"
#     mkdir -p "$class_name"
#     tar --no-same-owner -xf "$archive" -C "$class_name"
#     rm -f "$archive"
# done

cd "$OUTPUT_DIR"
echo "✅ Training set prepared."

# === [2] Extract Validation Set ===
echo "-> Step 2: Preparing validation set"
cd "../.."

mkdir -p "$VAL_DIR"
cp "$VAL_TAR" "$VAL_DIR/"
cd "$VAL_DIR"

echo "Untarring $VAL_TAR..."
tar -xf "$VAL_TAR"
rm -f "$VAL_TAR"

echo "Downloading valprep script..."
wget -q https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh -O valprep.sh
chmod +x valprep.sh

echo "Organizing validation images into class folders..."
./valprep.sh
rm -f valprep.sh


cd "$VAL_DIR"

cd ".."
echo "✅ Validation set prepared."

# === DONE ===
echo "🎉 ImageNet extraction complete!"
echo "Train classes: $(find train -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "Val classes:   $(find val -mindepth 1 -maxdepth 1 -type d | wc -l)"
