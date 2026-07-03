#!/bin/bash
#SBATCH --cluster=wice
#SBATCH --partition=gpu_h100
#SBATCH --account=lp_marianslab
#SBATCH --job-name=simba_xs_bf16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=robin.geens@kuleuven.be
#SBATCH --output=slurm_logs/slurm-simba.out
#SBATCH --error=slurm_logs/slurm-simba.err
# 
# Submit:  mkdir -p slurm_logs && sbatch main.sh
# Local:   NGPUS=1 ./main.sh

set -e

MODEL="simba_xs_bf16"
RUN_NAME="simba_xs_bf16"

# Multi-GPU config. Total batch is held constant
NGPUS=${NGPUS:-${SLURM_GPUS_ON_NODE:-1}}
TOTAL_BATCH=128
PER_GPU_BATCH=$(( TOTAL_BATCH / NGPUS ))
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(( NGPUS - 1 )))
export CUDA_VISIBLE_DEVICES

# Paths: server-agnostic
pick() { for p in "$@"; do [ -e "$p" ] && { echo "$p"; return; }; done; }
DATA_PATH=$(pick /volume1/users/rgeens/simba/dataset/ILSVRC2012 \
                 /scratch/leuven/391/vsc39144/simba/dataset/ILSVRC2012)
TOKEN_LABEL_PATH=$(pick /volume1/users/rgeens/simba/dataset/label_top5_train_nfnet/ \
                     /scratch/leuven/391/vsc39144/simba/dataset/label_top5_train_nfnet/)

# Self-resubmit chain: queue the next job to start after this one ends, so training
# can survive multiple walltime hits unattended. Skipped when running locally
# (no SLURM_JOB_ID) or when training has already completed (DONE exists).
DONE_FLAG="../out/$RUN_NAME/DONE"
if [ -f "$DONE_FLAG" ]; then
    echo "Found $DONE_FLAG — training already complete, exiting."
    exit 0
fi
if [ -n "$SLURM_JOB_ID" ]; then
    NEXT_JID=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID main.sh)
    echo "Queued next job $NEXT_JID with dependency afterany:$SLURM_JOB_ID"
fi

# Random port so concurrent runs on the same host don't collide on the rendezvous endpoint.
MASTER_PORT=${MASTER_PORT:-$(( 29500 + RANDOM % 1000 ))}

echo "Running on GPU(s) $CUDA_VISIBLE_DEVICES (NGPUS=$NGPUS, per-GPU batch=$PER_GPU_BATCH, total batch=$TOTAL_BATCH)"
nvidia-smi
source env/bin/activate

CHECKPOINT=$(ls -v checkpoints/$RUN_NAME/checkpoint-*.pth.tar | tail -n1)
# CHECKPOINT=checkpoints/simba_l_replace_rms/checkpoint-50.pth.tar
echo "Resuming from checkpoint: $CHECKPOINT"

torchrun  \
   --nproc_per_node=$NGPUS \
   --master_port=$MASTER_PORT \
   simba/main.py \
   --config config/$MODEL.py \
   --run-name $RUN_NAME \
   --output_dir checkpoints/$RUN_NAME \
   --data-path $DATA_PATH \
   --epochs 310  \
   --batch-size $PER_GPU_BATCH \
   --weight-decay 0.05 \
   --lr 1e-3 \
   --warmup-lr 1e-6 \
   --warmup-epochs 5 \
   --min-lr 1e-5 \
   --num_workers 32\
   --pin-mem \
   --token-label \
   --token-label-size 7 \
   --token-label-data $TOKEN_LABEL_PATH \
   --resume "$CHECKPOINT"
   # --drop-path 0.05 \
   # --finetune $CHECKPOINT \
