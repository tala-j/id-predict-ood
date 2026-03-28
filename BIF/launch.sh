#!/bin/bash
# Launch BIF computation across multiple GPUs, then run CKA.
# Usage: bash launch.sh [NUM_GPUS]
#   e.g. bash launch.sh 8

NUM_GPUS=${1:-8}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$SCRIPT_DIR/results"
mkdir -p "$SCRIPT_DIR/cka_results"

echo "Launching $NUM_GPUS workers for BIF computation..."

for ((i=0; i<NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i python "$SCRIPT_DIR/run_bif.py" \
        --worker_id $i --num_workers $NUM_GPUS \
        > "$SCRIPT_DIR/results/worker_${i}.log" 2>&1 &
    echo "  Worker $i on GPU $i (PID $!)"
done

echo "All workers launched. Logs: $SCRIPT_DIR/results/worker_*.log"
echo "Monitor: tail -f $SCRIPT_DIR/results/worker_*.log"
wait
echo "All BIF workers finished."

echo ""
echo "Running CKA..."
python "$SCRIPT_DIR/cka_bif.py" 2>&1 | tee "$SCRIPT_DIR/cka_results/cka.log"
echo "Done."
