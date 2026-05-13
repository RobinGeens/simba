#!/bin/bash

MODEL="simba_l_bf16"

# Extract RUN_NAME from the config file
# RUN_NAME=$(python3 -c "
# import os
# import sys
# sys.path.insert(0, 'config')
# from $MODEL import cfg
# print(os.path.basename(cfg['output_dir']))
# ")
RUN_NAME="exp_approx"

echo "Running on GPU $CUDA_VISIBLE_DEVICES"
nvidia-smi
source env/bin/activate

# ! Change this in next run
# CHECKPOINT=$(ls -v checkpoints/$RUN_NAME/checkpoint-*.pth.tar | tail -n1)
CHECKPOINT=checkpoints/simba_l_bf16_TL/checkpoint-316.pth.tar
echo "Resuming from checkpoint: $CHECKPOINT"

DATA_PATH="dataset/ILSVRC2012"
TOKEN_LABEL_PATH="dataset/label_top5_train_nfnet/"

CUDA_VISIBLE_DEVICES=0 torchrun  \
   --nproc_per_node=1 \
   simba/main.py \
   --config config/$MODEL.py \
   --run-name $RUN_NAME \
   --output_dir checkpoints/$RUN_NAME \
   --data-path $DATA_PATH \
   --epochs 346  \
   --batch-size 128 \
   --drop-path 0.05 \
   --weight-decay 0.05 \
   --lr 1e-3 \
   --num_workers 12\
   --pin-mem \
   --token-label \
   --token-label-size 7 \
   --token-label-data $TOKEN_LABEL_PATH \
   --resume $CHECKPOINT \
