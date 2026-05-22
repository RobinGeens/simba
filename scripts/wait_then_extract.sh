#!/bin/bash
# Watches the in-flight ImageNet download (already running) and, once all
# three .ok marker files exist, launches extraction. Safe to leave running.
set -e

TARS="/scratch/leuven/379/vsc37999/imagenet/tars"
EXTRACT="/data/leuven/379/vsc37999/workspace/simba/scripts/extract_imagenet.sh"
LOG="/scratch/leuven/379/vsc37999/imagenet/watcher.log"

exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] watcher started, waiting for downloads to finish"

while true; do
    if [ -f "$TARS/ILSVRC2012_devkit_t12.tar.gz.ok" ] \
       && [ -f "$TARS/ILSVRC2012_img_val.tar.ok" ] \
       && [ -f "$TARS/ILSVRC2012_img_train.tar.ok" ]; then
        echo "[$(date)] all .ok markers present, launching extract"
        break
    fi
    sleep 60
done

bash "$EXTRACT"
echo "[$(date)] watcher done"
