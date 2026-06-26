#!/bin/bash
#SBATCH --cluster=wice
#SBATCH --partition=gpu_h100
#SBATCH --account=lp_marianslab
#SBATCH --job-name=simba_s_bf16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=chao.fang@kuleuven.be
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err

# Submit:  mkdir -p slurm_logs && sbatch main.sh
# Local:   NGPUS=2 ./main.sh
# When submitted via sbatch, NGPUS defaults to 2 to match --gres=gpu:2 below.

MODEL="simba_s_bf16"
RUN_NAME="simba_s_bf16"

# Self-resubmit chain. Queue the next job at start so it accumulates queue
# priority while we run. Several gates can break the chain:
#   - DONE flag: training has finished
#   - STOP flag: manual escape hatch ('touch checkpoints/$RUN_NAME/STOP' to halt)
#   - No-progress guard: previous attempt produced no new checkpoint (likely a crash)
#   - sbatch error: detected and surfaced. The previous version used $(...) which
#     swallows stderr and never checked the return code, so failures like
#     'insufficient credits' silently broke the chain.
DONE_FLAG="checkpoints/$RUN_NAME/DONE"
STOP_FLAG="checkpoints/$RUN_NAME/STOP"
LAST_ATTEMPT="checkpoints/$RUN_NAME/.last_attempt"

if [ -f "$DONE_FLAG" ]; then
    echo "Found $DONE_FLAG — training already complete, exiting."
    exit 0
fi
if [ -f "$STOP_FLAG" ]; then
    echo "Found $STOP_FLAG — chain manually halted (rm to resume). Exiting."
    exit 0
fi

if [ -f "$LAST_ATTEMPT" ]; then
    LATEST_CKPT=$(ls -t checkpoints/$RUN_NAME/checkpoint-*.pth.tar 2>/dev/null | head -1)
    if [ -z "$LATEST_CKPT" ] || [ "$LAST_ATTEMPT" -nt "$LATEST_CKPT" ]; then
        echo "ERROR: previous attempt produced no new checkpoint — likely a crash. Breaking chain." >&2
        echo "ERROR: investigate, then 'rm $LAST_ATTEMPT' and resubmit to restart." >&2
        exit 1
    fi
fi
mkdir -p "checkpoints/$RUN_NAME"
touch "$LAST_ATTEMPT"

if [ -n "$SLURM_JOB_ID" ]; then
    NEXT_OUT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID main.sh 2>&1)
    NEXT_RC=$?
    if [ $NEXT_RC -ne 0 ] || [ -z "$NEXT_OUT" ]; then
        if echo "$NEXT_OUT" | grep -q "insufficient available credits"; then
            BAL=$(sam-balance 2>/dev/null | tail -1)
            echo "ERROR: resubmit failed — insufficient credits. Balance: $BAL" >&2
            echo "ERROR: top up credits and manually 'sbatch main.sh' to restart the chain." >&2
        else
            echo "ERROR: resubmit sbatch failed (rc=$NEXT_RC): $NEXT_OUT" >&2
        fi
        exit 1  # mark job FAILED so --mail-type=FAIL notifies us
    fi
    echo "Queued next job: $NEXT_OUT"
fi

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
GRAD_LOG_PATH="/scratch/leuven/379/vsc37999/grad_stats_s.log"

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
   --grad-log-path $GRAD_LOG_PATH \
   $RESUME_ARG
