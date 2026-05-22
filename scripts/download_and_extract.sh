#!/bin/bash
# Wrapper: run download, then extract on success.
# Logs go to the same /scratch/leuven/379/vsc37999/imagenet/ tree.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[$(date)] === pipeline start ==="
bash "$SCRIPT_DIR/download_imagenet.sh"
echo "[$(date)] download finished, starting extract"
bash "$SCRIPT_DIR/extract_imagenet.sh"
echo "[$(date)] === pipeline complete ==="
