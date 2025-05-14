#!/bin/bash

MODEL="simba_l_bf16"

echo "Running on GPU $CUDA_VISIBLE_DEVICES"
nvidia-smi
source env/bin/activate

# LATEST_CHECKPOINT=$(ls -v checkpoints/$MODEL/checkpoint-*.pth.tar | tail -n1)
# echo "Resuming from checkpoint: $LATEST_CHECKPOINT"

DATA_PATH="dataset/ILSVRC2012"
# LABEL_PATH="dataset/Image_net/imagenet_efficientnet_l2_sz475_top5/"


CUDA_VISIBLE_DEVICES=1 torchrun  \
   --nproc_per_node=1 \
   simba/main.py \
   --config config/$MODEL.py \
   --data-path $DATA_PATH \
   --epochs 310 \
   --batch-size 256 \
   --drop-path 0.05 \
   --weight-decay 0.05 \
   --lr 2e-3 \
   --num_workers 12\
   --pin-mem \
   # --resume $LATEST_CHECKPOINT \
   
   # --token-label \
   # --master_port=12346 \
   # --nnodes=1 \
   # --node_rank=0 \
   # --master_addr="localhost" \
   # --token-label-size 7 \
   # --token-label-data $LABEL_PATH
