#!/bin/bash
# For each lambda: pass 1 -> shared indices -> pass 2 -> CKA -> delete matrices.
# Usage: bash influence/launch_influence.sh

set -e

NUM_GPUS=$(nvidia-smi -L | wc -l)
LOG_DIR=/root/id-predict-ood/influence/logs
LAMBDAS="1e-10 1e-8 1e-6 1e-4 1e-2 1.0"

mkdir -p "$LOG_DIR"
mkdir -p /workspace/lambda_sweep_tmp

for LAM in $LAMBDAS; do
    echo ""
    echo "========================================"
    echo "Lambda = $LAM"
    echo "========================================"

    # ---- Pass 1: variance vectors ----
    echo "Pass 1: variances..."
    for ((i=0; i<NUM_GPUS; i++)); do
        CUDA_VISIBLE_DEVICES=$i PYTHONPATH=/root/id-predict-ood \
            python /root/id-predict-ood/influence/influence.py \
            --mode variances --lam "$LAM" --worker_id "$i" --num_workers "$NUM_GPUS" \
            >> "$LOG_DIR/influence_lam${LAM}.log" 2>&1 &
    done
    wait
    echo "Pass 1 done."

    # ---- Shared top-1000 indices ----
    echo "Computing shared indices..."
    PYTHONPATH=/root/id-predict-ood python /root/id-predict-ood/influence/influence.py \
        --mode shared_indices --lam "$LAM" >> "$LOG_DIR/influence_lam${LAM}.log" 2>&1
    echo "Shared indices saved."

    # ---- Pass 2: [1000x1000] matrices ----
    echo "Pass 2: matrices..."
    for ((i=0; i<NUM_GPUS; i++)); do
        CUDA_VISIBLE_DEVICES=$i PYTHONPATH=/root/id-predict-ood \
            python /root/id-predict-ood/influence/influence.py \
            --mode matrices --lam "$LAM" --worker_id "$i" --num_workers "$NUM_GPUS" \
            >> "$LOG_DIR/influence_lam${LAM}.log" 2>&1 &
    done
    wait
    echo "Pass 2 done."

    # ---- CKA + delete matrices ----
    echo "Computing CKA..."
    PYTHONPATH=/root/id-predict-ood python /root/id-predict-ood/influence/influence.py \
        --mode cka --lam "$LAM" >> "$LOG_DIR/influence_lam${LAM}.log" 2>&1
    echo "CKA done for lambda=$LAM. Matrices deleted."

done

echo ""
echo "All lambdas done. Results in: $LOG_DIR/lambda_sweep/"
