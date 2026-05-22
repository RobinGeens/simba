#!/bin/bash
#SBATCH --cluster=wice
#SBATCH --partition=gpu_h100
#SBATCH --account=lp_marianslab
#SBATCH --job-name=simba_b_bf16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=chao.fang@kuleuven.be
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err

# Submit:  mkdir -p slurm_logs && sbatch main.sh
# Local:   NGPUS=2 ./main.sh
# When submitted via sbatch, NGPUS defaults to 2 to match --gres=gpu:2 below.

MODEL="simba_b_bf16"

# Extract RUN_NAME from the config file
# RUN_NAME=$(python3 -c "
# import os
# import sys
# sys.path.insert(0, 'config')
# from $MODEL import cfg
# print(os.path.basename(cfg['output_dir']))
# ")
RUN_NAME="simba_b_bf16_rms"

# Multi-GPU config. Total batch is held constant at TOTAL_BATCH so the LR auto-scaling (lr * batch_size * world_size / 512) is unchanged.
# Under sbatch, default NGPUS to whatever --gres=gpu:N gave us; otherwise 1.
NGPUS=${NGPUS:-${SLURM_GPUS_ON_NODE:-1}}
TOTAL_BATCH=128
PER_GPU_BATCH=$(( TOTAL_BATCH / NGPUS ))
# Use first N GPUs
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(( NGPUS - 1 )))
export CUDA_VISIBLE_DEVICES

# Random port so concurrent runs on the same host don't collide on the rendezvous endpoint.
MASTER_PORT=${MASTER_PORT:-$(( 29500 + RANDOM % 1000 ))}

echo "Running on GPU(s) $CUDA_VISIBLE_DEVICES (NGPUS=$NGPUS, per-GPU batch=$PER_GPU_BATCH, total batch=$TOTAL_BATCH)"
nvidia-smi
source env/bin/activate

# Auto-resume from the latest checkpoint if one exists (first run starts from scratch).
CHECKPOINT=$(ls -v checkpoints/$RUN_NAME/checkpoint-*.pth.tar 2>/dev/null | tail -n1)
RESUME_ARG=""
if [ -n "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $CHECKPOINT"
    RESUME_ARG="--resume $CHECKPOINT"
else
    echo "No checkpoint found in checkpoints/$RUN_NAME — training from scratch."
fi

DATA_PATH="/scratch/leuven/379/vsc37999/imagenet"
TOKEN_LABEL_PATH="/scratch/leuven/379/vsc37999/label_top5_train_nfnet/"

torchrun  \
   --nproc_per_node=$NGPUS \
   --master_port=$MASTER_PORT \
   simba/main.py \
   --config config/$MODEL.py \
   --run-name $RUN_NAME \
   --output_dir checkpoints/$RUN_NAME \
   --data-path $DATA_PATH \
   --epochs 347  \
   --batch-size $PER_GPU_BATCH \
   --drop-path 0.05 \
   --weight-decay 0.05 \
   --lr 1e-3 \
   --num_workers 32\
   --pin-mem \
   --token-label \
   --token-label-size 7 \
   --token-label-data $TOKEN_LABEL_PATH \
   $RESUME_ARG
