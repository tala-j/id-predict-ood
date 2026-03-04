#!/bin/bash
# Launch sweep workers, one per GPU, in parallel.
# Usage: bash influence/launch_parallel.sh [sweep_id]

set -e

NUM_GPUS=$(nvidia-smi -L | wc -l)
SWEEP_ID=${1:-$(python -c "import uuid; print(uuid.uuid4().hex[:8])")}
LOG_DIR=/root/id-predict-ood/influence/logs

mkdir -p "$LOG_DIR"
echo "Sweep $SWEEP_ID | $NUM_GPUS GPUs"

for ((i=0; i<NUM_GPUS; i++)); do
    echo "Launching worker $i on GPU $i"
    CUDA_VISIBLE_DEVICES=$i PYTHONPATH=/root/id-predict-ood \
        python /root/id-predict-ood/influence/train.py \
        --sweep --sweep_id "$SWEEP_ID" --worker_id "$i" --num_workers "$NUM_GPUS" \
        > "$LOG_DIR/worker_$i.log" 2>&1 &
done

echo "All $NUM_GPUS workers launched. Monitor with:"
echo "  tail -f $LOG_DIR/worker_*.log"
echo "  watch -n 10 'wc -l $LOG_DIR/1Mexp2_bin_40_05_Transformer.csv 2>/dev/null'"

wait
echo "All workers finished."
