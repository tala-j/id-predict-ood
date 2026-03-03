#!/bin/bash
# Launch a wandb sweep and run one agent per available GPU.
#
# Usage:
#   # Start a new sweep:
#   bash influence/launch_sweep.sh
#
#   # Resume an existing sweep:
#   bash influence/launch_sweep.sh <sweep_id>

set -e

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ -n "$1" ]; then
    SWEEP_ID="$1"
    echo "Resuming sweep: $SWEEP_ID"
else
    # Create a new sweep and capture the sweep ID
    SWEEP_ID=$(python -c "
import yaml, wandb
config = yaml.safe_load(open('influence/transformer_sweep.yaml', 'r'))
sweep_id = wandb.sweep(config, project='hp_sweep', entity='trdy')
print(sweep_id)
")
    echo "Created new sweep: $SWEEP_ID"
fi

# Launch one agent per GPU
for ((i=0; i<NUM_GPUS; i++)); do
    echo "Launching agent on GPU $i"
    CUDA_VISIBLE_DEVICES=$i python influence/train.py --sweep --sweep_id "$SWEEP_ID" &
done

echo "All $NUM_GPUS agents launched for sweep $SWEEP_ID"
echo "Monitor at: https://wandb.ai/trdy/hp_sweep/sweeps/$SWEEP_ID"

wait
echo "All agents finished."
