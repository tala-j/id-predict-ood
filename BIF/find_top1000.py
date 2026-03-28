"""
Phase 1: Find the shared top-1000 training examples by BIF magnitude.

For each model:
  - Run SGLD, evaluating per-example losses on ALL 100K training examples
    and the 1000 OOD test examples at every post-burn-in draw.
  - Use online cross-product accumulation (no trace storage) to compute
    BIF scores for all [1000_test × 100K_train] pairs.
  - Save per-train score vector: mean |BIF| over test examples → [100K].

After all workers finish, run with --aggregate to:
  - Load all 50 score vectors, average across models, take global top-1000.
  - Save BIF/shared_top1000_bif.npy.

Usage:
  # Phase 1a — per-model scoring (parallel across GPUs)
  python find_top1000.py --worker_id 0 --num_workers 2
  python find_top1000.py --worker_id 1 --num_workers 2

  # Phase 1b — aggregate after all workers done
  python find_top1000.py --aggregate
"""

import os, sys, argparse, math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import BracketsDataset
from utils.model import get_transformer, N_CLASSES
from BIF.run_bif import (
    get_run_ids, load_model, load_train_subset, load_test_data,
    BracketsSubset, brackets_loss_fn,
    TARGET_BS, TARGET_LR,
)

_HERE = os.path.dirname(os.path.abspath(__file__))

SCORES_DIR   = os.path.join(_HERE, "rank_scores")
OUTPUT_IDX   = os.path.join(_HERE, "shared_top1000_bif.npy")

DEFAULT_EPSILON   = 3e-6
DEFAULT_GAMMA     = 500.0
DEFAULT_NBETA     = 100.0
DEFAULT_NUM_STEPS = 2000
DEFAULT_BURN_IN   = 500
DEFAULT_SGLD_BS   = 256
EVAL_BATCH_SIZE   = 1024    # larger batch for the full 100K eval pass


def score_path(run_id):
    return os.path.join(SCORES_DIR, f"{run_id}_bif_scores.npy")


# ---------------------------------------------------------------------------
# Online BIF magnitude accumulator
# ---------------------------------------------------------------------------

def compute_bif_scores(
    run_id,
    train_inputs, train_seq_lens, train_targets,   # full 100K (on GPU)
    test_inputs, test_seq_lens, test_targets,       # 1000 test (on GPU)
    epsilon, gamma, nbeta, num_steps, burn_in, sgld_batch_size,
    device,
):
    """
    Run SGLD and compute mean |BIF(test_i, train_j)| for all j in [100K].

    Uses online cross-product accumulation — no trace tensors stored.

    Returns: score [n_train] on CPU, mean |BIF| averaged over test examples.
    """
    model = load_model(run_id, device)
    w_star = [p.data.clone() for p in model.parameters()]

    n_train = len(train_inputs)
    n_test  = len(test_inputs)
    n_draws = num_steps - burn_in

    noise_std = math.sqrt(epsilon)
    half_eps  = epsilon * 0.5

    # Online accumulators (kept on GPU)
    cross_sum  = torch.zeros(n_test, n_train, device=device)  # Σ ℓ_test_i * ℓ_train_j
    sum_test   = torch.zeros(n_test,  device=device)
    sum_train  = torch.zeros(n_train, device=device)

    sgld_ds     = BracketsSubset(train_inputs, train_seq_lens, train_targets)
    sgld_loader = DataLoader(sgld_ds, batch_size=sgld_batch_size, shuffle=True, drop_last=True)
    sgld_iter   = _infinite(sgld_loader)

    train_ds     = BracketsSubset(train_inputs, train_seq_lens, train_targets)
    train_loader = DataLoader(train_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, drop_last=False)
    test_ds      = BracketsSubset(test_inputs,  test_seq_lens,  test_targets)
    test_loader  = DataLoader(test_ds,  batch_size=EVAL_BATCH_SIZE, shuffle=False, drop_last=False)

    draw_idx = 0

    for step in range(num_steps):
        # --- SGLD gradient step ---
        model.zero_grad()
        batch = next(sgld_iter)
        inp, sl, tgt = batch
        preds = _get_preds(model, inp, sl)
        F.cross_entropy(preds, tgt, reduction='mean').backward()

        with torch.no_grad():
            for p, ws in zip(model.parameters(), w_star):
                drift = nbeta * p.grad + gamma * (p.data - ws)
                p.data.add_(drift, alpha=-half_eps)
                p.data.add_(torch.randn_like(p), alpha=noise_std)

        if step < burn_in:
            continue

        # --- Evaluate losses (no grad) ---
        with torch.no_grad():
            tr_loss = _eval_losses(model, train_loader, device)   # [100K]
            te_loss = _eval_losses(model, test_loader,  device)   # [1000]

        # Online accumulation: cross_sum[i,j] += te[i] * tr[j]
        cross_sum.add_(torch.outer(te_loss, tr_loss))
        sum_test.add_(te_loss)
        sum_train.add_(tr_loss)

        draw_idx += 1
        if draw_idx % 200 == 0:
            print(f"    draw {draw_idx}/{n_draws}", flush=True)

    T = n_draws
    # BIF[i,j] = -(cross_sum[i,j]/T - mean_test[i]*mean_train[j]) * T/(T-1)
    mean_test  = sum_test  / T
    mean_train = sum_train / T
    bif = -(cross_sum / T - torch.outer(mean_test, mean_train)) * (T / (T - 1))  # [n_test, n_train]

    # Per-train score: mean |BIF| over test examples
    score = bif.abs().mean(dim=0)   # [n_train]
    return score.cpu().numpy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_preds(model, inputs, seq_lens):
    outputs = model(inputs)[0]
    last_idx = (seq_lens - 1).unsqueeze(1).repeat(1, N_CLASSES)
    return outputs.gather(1, last_idx.unsqueeze(1))[:, 0, :]


@torch.no_grad()
def _eval_losses(model, loader, device):
    parts = []
    for inp, sl, tgt in loader:
        inp, sl, tgt = inp.to(device), sl.to(device), tgt.to(device)
        preds = _get_preds(model, inp, sl)
        parts.append(F.cross_entropy(preds, tgt, reduction='none').cpu())
    return torch.cat(parts).to(device)


def _infinite(loader):
    while True:
        yield from loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate",      action="store_true",
                        help="Aggregate saved score files → shared_top1000_bif.npy")
    parser.add_argument("--worker_id",       type=int,   default=0)
    parser.add_argument("--num_workers",     type=int,   default=1)
    parser.add_argument("--epsilon",         type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--gamma",           type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--nbeta",           type=float, default=DEFAULT_NBETA)
    parser.add_argument("--num_steps",       type=int,   default=DEFAULT_NUM_STEPS)
    parser.add_argument("--burn_in",         type=int,   default=DEFAULT_BURN_IN)
    parser.add_argument("--sgld_batch_size", type=int,   default=DEFAULT_SGLD_BS)
    args = parser.parse_args()

    os.makedirs(SCORES_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Aggregation mode
    # ------------------------------------------------------------------
    if args.aggregate:
        run_ids = get_run_ids()
        scores = []
        missing = []
        for run_id in run_ids:
            p = score_path(run_id)
            if os.path.exists(p):
                scores.append(np.load(p))
            else:
                missing.append(run_id)
        if missing:
            print(f"WARNING: missing scores for {len(missing)} models: {missing}")
        if not scores:
            print("No score files found.")
            return

        mean_score = np.stack(scores).mean(axis=0)   # [n_train]
        top1000 = np.argsort(mean_score)[::-1][:1000].copy()
        np.save(OUTPUT_IDX, top1000)
        print(f"Saved top-1000 indices → {OUTPUT_IDX}")
        print(f"  Score range of top-1000: "
              f"{mean_score[top1000].min():.4e} – {mean_score[top1000].max():.4e}")
        print(f"  Score range overall:     "
              f"{mean_score.min():.4e} – {mean_score.max():.4e}")
        return

    # ------------------------------------------------------------------
    # Per-model scoring mode
    # ------------------------------------------------------------------
    device = torch.device("cuda")

    run_ids = get_run_ids()
    my_runs = [r for i, r in enumerate(run_ids) if i % args.num_workers == args.worker_id]
    pending = [r for r in my_runs if not os.path.exists(score_path(r))]
    print(f"Worker {args.worker_id}/{args.num_workers}: "
          f"{len(pending)}/{len(my_runs)} models pending", flush=True)

    if not pending:
        print("Nothing to do.")
        return

    print("Loading training data (100K)...", flush=True)
    train_inputs, train_seq_lens, train_targets = load_train_subset(device)

    print("Loading test data (1000 OOD)...", flush=True)
    test_inputs, test_seq_lens, test_targets = load_test_data(device)

    print(f"Config: eps={args.epsilon}, gamma={args.gamma}, nbeta={args.nbeta}, "
          f"steps={args.num_steps}, burn_in={args.burn_in}, "
          f"eval_batch={EVAL_BATCH_SIZE}", flush=True)

    for i, run_id in enumerate(pending):
        print(f"\n[{i+1}/{len(pending)}] {run_id}", flush=True)

        score = compute_bif_scores(
            run_id,
            train_inputs, train_seq_lens, train_targets,
            test_inputs,  test_seq_lens,  test_targets,
            args.epsilon, args.gamma, args.nbeta,
            args.num_steps, args.burn_in, args.sgld_batch_size,
            device,
        )
        np.save(score_path(run_id), score)
        print(f"  Saved scores [{score.shape[0]}] → {score_path(run_id)}", flush=True)

        torch.cuda.empty_cache()

    print("\nDone. Run with --aggregate to compute shared top-1000.", flush=True)


if __name__ == "__main__":
    main()
