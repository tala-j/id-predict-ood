"""
Phase 2: Compute final 1000×1000 BIF matrices using BIF-selected top-1000 train points.

Requires shared_top1000_bif.npy (produced by find_top1000.py --aggregate).

Produces one [1000_test × 1000_train] matrix per model.

Usage:
  python run_bif_top1000.py --worker_id 0 --num_workers 2
  python run_bif_top1000.py --worker_id 1 --num_workers 2
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BIF.bif import BIFConfig, compute_bif
from BIF.run_bif import (
    get_run_ids, load_model, load_train_subset, load_test_data,
    BracketsSubset, brackets_loss_fn,
    DEFAULT_EPSILON, DEFAULT_GAMMA, DEFAULT_NBETA,
    DEFAULT_NUM_STEPS, DEFAULT_BURN_IN, DEFAULT_SGLD_BS,
)

_HERE = os.path.dirname(os.path.abspath(__file__))

TOP1000_IDX = os.path.join(_HERE, "shared_top1000_bif.npy")
OUTPUT_DIR  = os.path.join(_HERE, "results_top1000")


def matrix_path(run_id):
    return os.path.join(OUTPUT_DIR, f"{run_id}_bif_top1000.npy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id",       type=int,   default=0)
    parser.add_argument("--num_workers",     type=int,   default=1)
    parser.add_argument("--epsilon",         type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--gamma",           type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--nbeta",           type=float, default=DEFAULT_NBETA)
    parser.add_argument("--num_steps",       type=int,   default=DEFAULT_NUM_STEPS)
    parser.add_argument("--burn_in",         type=int,   default=DEFAULT_BURN_IN)
    parser.add_argument("--sgld_batch_size", type=int,   default=DEFAULT_SGLD_BS)
    args = parser.parse_args()

    if not os.path.exists(TOP1000_IDX):
        print(f"ERROR: {TOP1000_IDX} not found. Run find_top1000.py --aggregate first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda")

    top1000 = np.load(TOP1000_IDX)
    print(f"Loaded BIF top-1000 indices from {TOP1000_IDX}", flush=True)

    run_ids = get_run_ids()
    my_runs = [r for i, r in enumerate(run_ids) if i % args.num_workers == args.worker_id]
    pending = [r for r in my_runs if not os.path.exists(matrix_path(r))]
    print(f"Worker {args.worker_id}/{args.num_workers}: "
          f"{len(pending)}/{len(my_runs)} models pending", flush=True)

    if not pending:
        print("Nothing to do.")
        return

    print("Loading training data (100K)...", flush=True)
    train_inputs, train_seq_lens, train_targets = load_train_subset(device)

    print("Loading test data (1000 OOD)...", flush=True)
    test_inputs, test_seq_lens, test_targets = load_test_data(device)

    # SGLD dataset: full 100K for gradient steps
    sgld_ds = BracketsSubset(train_inputs, train_seq_lens, train_targets)

    # Eval dataset: BIF-selected top-1000 train examples
    train_eval_ds = BracketsSubset(
        train_inputs[top1000],
        train_seq_lens[top1000],
        train_targets[top1000],
    )

    # Query dataset: 1000 OOD test examples
    query_ds = BracketsSubset(test_inputs, test_seq_lens, test_targets)

    config = BIFConfig(
        epsilon         = args.epsilon,
        gamma           = args.gamma,
        nbeta           = args.nbeta,
        n_chains        = 1,
        n_steps         = args.num_steps,
        burn_in         = args.burn_in,
        sgld_batch_size = args.sgld_batch_size,
        eval_batch_size = 512,
        param_subset    = "all",
        device          = "cuda",
    )
    print(f"BIFConfig: eps={config.epsilon}, gamma={config.gamma}, "
          f"nbeta={config.nbeta}, steps={config.n_steps}, "
          f"burn_in={config.burn_in}", flush=True)

    for i, run_id in enumerate(pending):
        print(f"\n[{i+1}/{len(pending)}] {run_id}", flush=True)

        model = load_model(run_id, device)

        results = compute_bif(
            model              = model,
            train_dataset      = sgld_ds,
            query_dataset      = query_ds,
            loss_fn            = brackets_loss_fn,
            config             = config,
            train_eval_dataset = train_eval_ds,
        )
        # bif_matrix: [1000_test, 1000_train]
        np.save(matrix_path(run_id), results.bif_matrix.numpy())
        print(f"  Saved {results.bif_matrix.shape} → {matrix_path(run_id)}", flush=True)

        del model
        torch.cuda.empty_cache()

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
