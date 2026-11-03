#!/bin/bash
#SBATCH --cluster=wice
#SBATCH --partition=gpu_h100,gpu_a100
#SBATCH --account=lp_marianslab
#SBATCH --job-name=simba_s_ref
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=320G
#SBATCH --time=72:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=robin.geens@kuleuven.be
#SBATCH --output=slurm_logs/slurm-simba-%j.out
#SBATCH --error=slurm_logs/slurm-simba-%j.err
#
# The job name picks the run (case block below):
#   mkdir -p slurm_logs && sbatch main.sh                      (simba_s_ref)
#   sbatch --job-name=simba_b_ref main.sh
# Local:   RUN_NAME=simba_s_ref NGPUS=1 ./main.sh

set -e

RUN_NAME=${SLURM_JOB_NAME:-${RUN_NAME:?set RUN_NAME for a local run}}
FINETUNE_CKPT=""  # set in a run's case entry to start it from a checkpoint
case "$RUN_NAME" in
    # Original SiMBA as reference, from scratch: total batch 1024, 310 cosine epochs + 20 at min-lr
    simba_s_ref | simba_b_ref | simba_l_ref)
        MODEL="$RUN_NAME"
        TOTAL_BATCH=1024; NUM_WORKERS=15
        TRAIN_ARGS=(--epochs 310 --extra-epochs 20 --lr 1e-3 --warmup-lr 1e-6 --warmup-epochs 5 --min-lr 1e-5 --no-grad-stats)
        ;;
    *)
        echo "ERROR: unknown run '$RUN_NAME' (see the case block in main.sh)" >&2
        exit 1
        ;;
esac

# Multi-GPU config. Total batch is held constant
NGPUS=${NGPUS:-4}  # keep in sync with #SBATCH --gres=gpu: above
PER_GPU_BATCH=$(( TOTAL_BATCH / NGPUS ))
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(( NGPUS - 1 )))
export CUDA_VISIBLE_DEVICES

# Paths: server-agnostic
pick() { for p in "$@"; do [ -e "$p" ] && { echo "$p"; return; }; done; }
DATA_PATH=$(pick /volume1/users/rgeens/simba/dataset/ILSVRC2012 \
                 /scratch/leuven/391/vsc39144/simba/dataset/ILSVRC2012)
TOKEN_LABEL_PATH=$(pick /volume1/users/rgeens/simba/dataset/label_top5_train_nfnet/ \
                     /scratch/leuven/391/vsc39144/simba/dataset/label_top5_train_nfnet/)

# On VSC the unpacked copy is on scratch, which purges unread files. Check if still valid first
if [ ! -d /volume1/users/rgeens/simba/dataset ]; then
    bash scripts/check_imagenet.sh || exit 1
fi

# pick() only checks that a path exists. Fail here, before the chain queues the
# next job, if the data is missing or a partial copy (empty class dirs).
for p in "$DATA_PATH/train" "$DATA_PATH/val" "$TOKEN_LABEL_PATH"; do
    if [ -z "$(find -L "$p" -type f -print -quit 2>/dev/null)" ]; then
        echo "ERROR: no files under '$p' (DATA_PATH='$DATA_PATH', TOKEN_LABEL_PATH='$TOKEN_LABEL_PATH')" >&2
        exit 1
    fi
done

# Self-resubmit chain; the next job may start in this job's last 24 h (1440 min) and then takes over
OUTPUT_DIR="checkpoints/$RUN_NAME"
if [ -f ../slurm_chain.sh ]; then
    export SBATCH_JOB_NAME="$RUN_NAME"
    source ../slurm_chain.sh
    self_resubmit main.sh "$OUTPUT_DIR" 1440
fi

# Random port so concurrent runs on the same host don't collide on the rendezvous endpoint.
MASTER_PORT=${MASTER_PORT:-$(( 29500 + RANDOM % 1000 ))}

echo "Run $RUN_NAME: GPU(s) $CUDA_VISIBLE_DEVICES (NGPUS=$NGPUS, per-GPU batch=$PER_GPU_BATCH, total batch=$TOTAL_BATCH)"
nvidia-smi
source env/bin/activate
    
# Resume from the run's newest checkpoint (last.pth.tar holds the newest epoch when that
# epoch is not in the top 10); otherwise finetune from FINETUNE_CKPT, or start from scratch.
CHECKPOINT=$(ls -t "$OUTPUT_DIR"/last.pth.tar "$OUTPUT_DIR"/checkpoint-*.pth.tar 2>/dev/null | head -n1 || true)
if [ -n "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $CHECKPOINT"
    CKPT_ARGS=(--resume "$CHECKPOINT")
elif [ -n "$FINETUNE_CKPT" ]; then
    echo "Finetuning from checkpoint: $FINETUNE_CKPT"
    CKPT_ARGS=(--finetune "$FINETUNE_CKPT")
else
    echo "Training from scratch"
    CKPT_ARGS=()
fi

torchrun  \
   --nproc_per_node=$NGPUS \
   --master_port=$MASTER_PORT \
   simba/main.py \
   --config config/$MODEL.py \
   --run-name $RUN_NAME \
   --output_dir $OUTPUT_DIR \
   --data-path $DATA_PATH \
   --batch-size $PER_GPU_BATCH \
   --weight-decay 0.05 \
   --num_workers $NUM_WORKERS \
   --pin-mem \
   --token-label \
   --token-label-size 7 \
   --token-label-data $TOKEN_LABEL_PATH \
   "${TRAIN_ARGS[@]}" \
   "${CKPT_ARGS[@]}"

touch "$OUTPUT_DIR/DONE"
