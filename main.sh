#!/bin/bash
#SBATCH --cluster=wice
#SBATCH --partition=gpu_h100,gpu_a100
#SBATCH --account=lp_marianslab
#SBATCH --job-name=simba_b_finetune_fp8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=robin.geens@kuleuven.be
#SBATCH --output=slurm_logs/slurm-simba-%j.out
#SBATCH --error=slurm_logs/slurm-simba-%j.err
# 
# Submit:  mkdir -p slurm_logs && sbatch main.sh
# Local:   NGPUS=1 ./main.sh

set -e

MODEL="simba_b_fp8"
RUN_NAME="simba_b_finetune_fp8"
FINETUNE_CKPT="checkpoints/simba_b_bf16_rms/checkpoint-346.pth.tar"

# Multi-GPU config. Total batch is held constant
TOTAL_BATCH=128
NGPUS=${NGPUS:-2}  # keep in sync with #SBATCH --gres=gpu: above
PER_GPU_BATCH=$(( TOTAL_BATCH / NGPUS ))
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(( NGPUS - 1 )))
export CUDA_VISIBLE_DEVICES

# Paths: server-agnostic
pick() { for p in "$@"; do [ -e "$p" ] && { echo "$p"; return; }; done; }
DATA_PATH=$(pick /volume1/users/rgeens/simba/dataset/ILSVRC2012 \
                 /lustre1/project/stg_00218/simba/dataset/ILSVRC2012)
TOKEN_LABEL_PATH=$(pick /volume1/users/rgeens/simba/dataset/label_top5_train_nfnet/ \
                     /scratch/leuven/391/vsc39144/simba/dataset/label_top5_train_nfnet/)

# pick() only checks that a path exists. Fail here, before the chain queues the
# next job, if the data is missing or a partial copy (empty class dirs).
for p in "$DATA_PATH/train" "$DATA_PATH/val" "$TOKEN_LABEL_PATH"; do
    if [ -z "$(find -L "$p" -type f -print -quit 2>/dev/null)" ]; then
        echo "ERROR: no files under '$p' (DATA_PATH='$DATA_PATH', TOKEN_LABEL_PATH='$TOKEN_LABEL_PATH')" >&2
        exit 1
    fi
done

# Self-resubmit chain
OUTPUT_DIR="checkpoints/$RUN_NAME"
if [ -f ../slurm_chain.sh ]; then
    source ../slurm_chain.sh
    self_resubmit main.sh "$OUTPUT_DIR"
fi

# Random port so concurrent runs on the same host don't collide on the rendezvous endpoint.
MASTER_PORT=${MASTER_PORT:-$(( 29500 + RANDOM % 1000 ))}

echo "Running on GPU(s) $CUDA_VISIBLE_DEVICES (NGPUS=$NGPUS, per-GPU batch=$PER_GPU_BATCH, total batch=$TOTAL_BATCH)"
nvidia-smi
source env/bin/activate

# Resume from the run's own latest checkpoint if one exists; otherwise start
# the finetune from the bf16 checkpoint (first run of the chain).
CHECKPOINT=$(ls -v checkpoints/$RUN_NAME/checkpoint-*.pth.tar 2>/dev/null | tail -n1 || true)
if [ -n "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $CHECKPOINT"
    CKPT_ARGS=(--resume "$CHECKPOINT")
else
    echo "Finetuning from checkpoint: $FINETUNE_CKPT"
    CKPT_ARGS=(--finetune "$FINETUNE_CKPT")
fi

torchrun  \
   --nproc_per_node=$NGPUS \
   --master_port=$MASTER_PORT \
   simba/main.py \
   --config config/$MODEL.py \
   --run-name $RUN_NAME \
   --output_dir checkpoints/$RUN_NAME \
   --data-path $DATA_PATH \
   --epochs 10  \
   --batch-size $PER_GPU_BATCH \
   --weight-decay 0.05 \
   --lr 2e-5 \
   --warmup-lr 1e-7 \
   --warmup-epochs 1 \
   --min-lr 1e-7 \
   --num_workers 32\
   --pin-mem \
   --token-label \
   --token-label-size 7 \
   --token-label-data $TOKEN_LABEL_PATH \
   "${CKPT_ARGS[@]}" 

touch "$OUTPUT_DIR/DONE"
