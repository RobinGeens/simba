#!/bin/bash


MODEL="simba_l_bf16"
RUN_NAME="simba_l_bf16_B"
AUTOCAST_DTYPE="bfloat16"


echo "Running on GPU $CUDA_VISIBLE_DEVICES"
nvidia-smi
source env/bin/activate


# #### Set aggressive GPU memory management ####
# # export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:1024"
# export CUDA_MEMORY_FRACTION=0.95
# # Pre-allocate GPU memory to prevent other processes from taking it
# python3 -c "
# import torch
# torch.cuda.empty_cache()
# torch.cuda.memory.set_per_process_memory_fraction(0.95)
# torch.cuda.memory.set_per_process_memory_fraction(0.95)
# "
# #### ####


LATEST_CHECKPOINT=$(ls -v checkpoints/$RUN_NAME/checkpoint-*.pth.tar | tail -n1)
echo "Resuming from checkpoint: $LATEST_CHECKPOINT"

DATA_PATH="dataset/ILSVRC2012"

CUDA_VISIBLE_DEVICES=1 torchrun  \
   --nproc_per_node=1 \
   simba/main.py \
   --config config/$MODEL.py \
   --run-name $RUN_NAME \
   --autocast-dtype $AUTOCAST_DTYPE \
   --output_dir checkpoints/$RUN_NAME \
   --data-path $DATA_PATH \
   --epochs 310 \
   --batch-size 128 \
   --drop-path 0.05 \
   --weight-decay 0.05 \
   --lr 1e-3 \
   --num_workers 12\
   --pin-mem \
   --resume $LATEST_CHECKPOINT \
