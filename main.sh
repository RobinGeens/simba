#!/bin/bash

# echo "Current path is $PATH"
echo "Running"
nvidia-smi
echo $CUDA_VISIBLE_DEVICES


DATA_PATH="dataset/ILSVRC2012"
# LABEL_PATH="dataset/Image_net/imagenet_efficientnet_l2_sz475_top5/"
export DATA_PATH
export LABEL_PATH

torchrun  \
   --nproc_per_node=2 \
   main.py \
   --config config/simba_s.py \
   --data-path $DATA_PATH \
   --epochs 310 \
   --batch-size 128 \
   --drop-path 0.05 \
   --weight-decay 0.05 \
   --lr 1e-3 \
   --num_workers 24\
   --token-label \
   # --master_port=12346 \
   # --nnodes=1 \
   # --node_rank=0 \
   # --master_addr="localhost" \
   # --token-label-size 7 \
   # --token-label-data $LABEL_PATH
