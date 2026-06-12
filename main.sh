#!/bin/bash

MODEL="simba_s_bf16"
RUN_NAME="simba_s_bf16"

# Multi-GPU config. Total batch is held constant at TOTAL_BATCH so the LR auto-scaling (lr * batch_size * world_size / 512) is unchanged.
NGPUS=${NGPUS:-1}
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

CHECKPOINT=$(ls -v checkpoints/$RUN_NAME/checkpoint-*.pth.tar | tail -n1)
# CHECKPOINT=checkpoints/simba_l_replace_rms/checkpoint-50.pth.tar
echo "Resuming from checkpoint: $CHECKPOINT"

DATA_PATH="/volume1/users/rgeens/simba/dataset/ILSVRC2012"
TOKEN_LABEL_PATH="/volume1/users/rgeens/simba/dataset/label_top5_train_nfnet/"

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
   # --resume $CHECKPOINT \ # <- TODO restore this if you have an initial checkpoint
   # --drop-path 0.05 \
   # --finetune $CHECKPOINT \
