#!/bin/bash
# Launch attention-head susceptibility computation across multiple GPUs.
# Usage: bash launch.sh [NUM_GPUS]
#   e.g. bash launch.sh 8

NUM_GPUS=${1:-8}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$SCRIPT_DIR/results"

echo "Launching $NUM_GPUS workers..."

for ((i=0; i<NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i python "$SCRIPT_DIR/susceptibility.py" \
        --worker_id $i --num_workers $NUM_GPUS \
        > "$SCRIPT_DIR/results/worker_${i}.log" 2>&1 &
    echo "  Worker $i on GPU $i (PID $!)"
done

echo "All workers launched. Monitor: tail -f $SCRIPT_DIR/results/worker_*.log"
wait
echo "All workers finished."

# Run CKA analysis once all workers complete
echo "Running CKA analysis..."
python "$SCRIPT_DIR/cka.py"
echo "CKA done. Results in $SCRIPT_DIR/cka_results/"
