#!/bin/bash
set -e
shopt -s nullglob

DEST="/scratch/leuven/379/vsc37999/imagenet/tars"
LOGFILE="$DEST/download.log"
mkdir -p "$DEST"
cd "$DEST"

exec > >(tee -a "$LOGFILE") 2>&1

echo "=== ImageNet ILSVRC2012 download started at $(date) ==="
echo "Destination: $DEST"

BASE="https://image-net.org/data/ILSVRC/2012"

# filename  md5  url
FILES=(
    "ILSVRC2012_devkit_t12.tar.gz|fa75699e90414af021442c21a62c3abf|$BASE/ILSVRC2012_devkit_t12.tar.gz"
    "ILSVRC2012_img_val.tar|29b22e2961454d5413ddabcf34fc5622|$BASE/ILSVRC2012_img_val.tar"
    "ILSVRC2012_img_train.tar|1d675b47d978889d74fa0da5fadfb00e|$BASE/ILSVRC2012_img_train.tar"
)

for entry in "${FILES[@]}"; do
    IFS='|' read -r fname md5 url <<< "$entry"
    echo ""
    echo "--- $fname ---"

    if [ -f "$fname" ] && [ -f "$fname.ok" ]; then
        echo "Already downloaded and verified, skipping."
        continue
    fi

    echo "Downloading from $url"
    wget -c --tries=20 --retry-connrefused --timeout=60 -O "$fname" "$url"

    echo "Verifying MD5 (expect $md5)..."
    actual=$(md5sum "$fname" | awk '{print $1}')
    if [ "$actual" = "$md5" ]; then
        echo "MD5 OK"
        touch "$fname.ok"
    else
        echo "MD5 MISMATCH! expected=$md5 actual=$actual"
        exit 1
    fi
done

echo ""
echo "=== All downloads complete at $(date) ==="
ls -lh "$DEST"
