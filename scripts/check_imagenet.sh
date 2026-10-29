#!/bin/bash
# Check the unpacked ImageNet on scratch by file count (scratch purges files unread for 30 days).
# The durable copy is the per-class tars on staging; --restore unpacks them over the scratch copy first.
#   bash scripts/check_imagenet.sh [--restore]
set -uo pipefail
SCR=/scratch/leuven/391/vsc39144/simba/dataset
TARS=/lustre1/project/stg_00218/simba/dataset/tars
TREES="ILSVRC2012/train:train:1281167 ILSVRC2012/val:val:50000 label_top5_train_nfnet:label_top5_train_nfnet:1281167"

if [ "${1:-}" = --restore ]; then
    for t in $TREES; do
        IFS=: read -r d s _ <<< "$t"
        echo "restoring $SCR/$d from $TARS/$s"
        mkdir -p "$SCR/$d"
        ls "$TARS/$s"/*.tar | xargs -P 16 -n 1 tar -C "$SCR/$d" -m --exclude=@eaDir --exclude='.*' -xf \
            || { echo "ERROR: restoring $SCR/$d failed" >&2; exit 1; }
    done
fi

t0=$(date +%s); bad=0; msg=""
for t in $TREES; do
    IFS=: read -r d _ n <<< "$t"
    got=$(ls -d "$SCR/$d"/*/ 2>/dev/null | xargs -r -P 16 -n 50 sh -c 'find "$@" -type f -printf .' _ | wc -c)
    msg+=" $d $got/$n,"
    [ "$got" -eq "$n" ] || bad=1
done
if [ "$bad" = 1 ]; then
    echo "ERROR: ImageNet on scratch is incomplete:${msg%,}. Restore it: bash scripts/check_imagenet.sh --restore" >&2
    exit 1
fi
echo "ImageNet on scratch OK:${msg%,} ($(( $(date +%s) - t0 )) s)"
