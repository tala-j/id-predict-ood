"""
Susceptibility estimation via SGLD posterior sampling.

For each of the 50 models (bs=8, lr=1e-4):
1. Load checkpoint w*
2. Run SGLD to sample from localized posterior
   p(w; w*, β, γ) ∝ exp{-nβL(w) - γ/2 ||w-w*||²}
3. Collect per-sample loss traces on 1000 test + union(train) samples
4. Compute susceptibility: χ_{i,j} = -Cov[ℓ_test_i(w), ℓ_train_j(w)]

Runs SGLD once per model, evaluates on the union of all lambdas' shared
top-1000 train indices, then slices per-lambda to produce one 1000×1000
susceptibility matrix per (model, lambda).

Usage:
  python susceptibility.py --worker_id 0 --num_workers 8
  python susceptibility.py --lambda_val 1e-6    # single lambda only
"""

import os, sys, argparse, math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import BracketsDataset
from utils.model import get_transformer, N_CLASSES

# -------------------------
# Paths
# -------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

INFLUENCE_DIR = os.path.join(_ROOT, "influence")
TRAIN_CSV = os.path.join(INFLUENCE_DIR, "data/train_binomial(40,0.5).csv")
OOD_CSV = os.path.join(INFLUENCE_DIR, "data/test_binomial(40,0.5).csv")
RESULTS_CSV = os.path.join(INFLUENCE_DIR, "logs/1Mexp2_bin_40_05_Transformer.csv")
MODELS_DIR = os.path.join(_HERE, "models")
OUTPUT_DIR = os.path.join(_HERE, "results")
LAMBDA_SWEEP_DIR = os.path.join(INFLUENCE_DIR, "logs/lambda_sweep")

# -------------------------
# Config
# -------------------------
TARGET_BS = 8
TARGET_LR = 0.0001
TRAIN_SUBSET = 100_000
TRAIN_SEED = 42

ALL_LAMBDAS = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]

# SGLD defaults (from Baker et al. / friend's code)
DEFAULT_EPSILON = 3e-6
DEFAULT_GAMMA = 500.0
DEFAULT_NBETA = 100.0
DEFAULT_NUM_STEPS = 2000
DEFAULT_BURN_IN = 500
DEFAULT_SGLD_BS = 256


# -------------------------
# Helpers
# -------------------------
def lambda_tag(lam):
    return f"lambda{lam:.0e}".replace("e-0", "e-").replace("e+0", "e")


def matrix_path(run_id, lam):
    return os.path.join(OUTPUT_DIR, f"{run_id}_susc_{lambda_tag(lam)}.npy")


def get_run_ids():
    df = pd.read_csv(RESULTS_CSV)
    final = df[(df["batch_size"] == TARGET_BS) & (df["lr"] == TARGET_LR) &
               (df["datapoints_seen"] == df["datapoints_seen"].max())]
    return sorted(final["run_id"].tolist())


def load_model(run_id):
    ckpt = os.path.join(MODELS_DIR, f"run_{run_id}_final.pt")
    model = get_transformer(n_layer=3, n_head=4, n_embd=64,
                            embd_pdrop=0, resid_pdrop=0, attn_pdrop=0)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model


def get_preds(model, inputs, seq_lens):
    outputs = model(inputs)[0]
    last_idx = (seq_lens - 1).unsqueeze(1).repeat(1, N_CLASSES)
    return outputs.gather(1, last_idx.unsqueeze(1))[:, 0, :]


def load_shared_indices(lam):
    path = os.path.join(LAMBDA_SWEEP_DIR, f"shared_top1000_{lambda_tag(lam)}.npy")
    return np.load(path)


def load_all_shared_indices(lambdas):
    """Load shared indices for given lambdas; return union array + per-lambda index maps."""
    per_lambda = {}
    all_idx = set()
    for lam in lambdas:
        idx = load_shared_indices(lam)
        per_lambda[lam] = idx
        all_idx.update(idx.tolist())

    union_idx = np.array(sorted(all_idx))
    # Map: for each lambda, positions within the union array
    union_pos = {v: i for i, v in enumerate(union_idx)}
    lambda_positions = {}
    for lam, idx in per_lambda.items():
        lambda_positions[lam] = np.array([union_pos[i] for i in idx])

    return union_idx, lambda_positions


def load_train_subset():
    """Load the same 100K training subset used by Kronfluence (seed=42)."""
    df = pd.read_csv(TRAIN_CSV)
    df = df.sample(n=TRAIN_SUBSET, random_state=TRAIN_SEED).reset_index(drop=True)
    bd = BracketsDataset(df)
    inputs = bd.toks
    seq_lens = torch.tensor([len(s) + 2 for s in bd.strs])
    targets = bd.ylabels.long()
    return inputs, seq_lens, targets


def load_test_data():
    """Load full OOD test set (1000 samples)."""
    df = pd.read_csv(OOD_CSV)
    bd = BracketsDataset(df)
    inputs = bd.toks
    seq_lens = torch.tensor([len(s) + 2 for s in bd.strs])
    targets = bd.ylabels.long()
    return inputs, seq_lens, targets


# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def eval_per_sample_losses(model, inputs, seq_lens, targets, batch_size=512):
    """Compute per-sample cross-entropy losses."""
    n = len(inputs)
    losses = torch.zeros(n, device=inputs.device)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        preds = get_preds(model, inputs[i:j], seq_lens[i:j])
        losses[i:j] = F.cross_entropy(preds, targets[i:j], reduction='none')
    return losses


# -------------------------
# SGLD
# -------------------------
def run_sgld(run_id, train_inputs, train_seq_lens, train_targets,
             eval_train_inputs, eval_train_seq_lens, eval_train_targets,
             test_inputs, test_seq_lens, test_targets,
             epsilon, gamma, nbeta, num_steps, burn_in, sgld_batch_size):
    """
    Run SGLD from checkpoint w* and collect per-sample loss traces.

    SGLD update:
        w_{t+1} = w_t - ε[nβ·∇L_batch(w_t) + γ(w_t - w*)] + √(2ε)·η

    Returns:
        test_traces:  [num_draws, n_test]
        train_traces: [num_draws, n_train_eval]
    """
    device = torch.device("cuda")
    model = load_model(run_id).to(device)

    # Store w* for the localization prior
    w_star = [p.data.clone() for p in model.parameters()]

    n_train = len(train_inputs)
    num_draws = num_steps - burn_in
    n_test = len(test_inputs)
    n_train_eval = len(eval_train_inputs)

    test_traces = torch.zeros(num_draws, n_test, device=device)
    train_traces = torch.zeros(num_draws, n_train_eval, device=device)

    noise_scale = math.sqrt(2 * epsilon)
    draw_idx = 0

    for step in range(num_steps):
        # --- SGLD gradient step ---
        model.zero_grad()

        idx = torch.randint(0, n_train, (sgld_batch_size,), device=device)
        preds = get_preds(model, train_inputs[idx], train_seq_lens[idx])
        loss = F.cross_entropy(preds, train_targets[idx], reduction='mean')
        loss.backward()

        with torch.no_grad():
            for p, ps in zip(model.parameters(), w_star):
                grad = nbeta * p.grad + gamma * (p.data - ps)
                p.data -= epsilon * grad
                p.data += noise_scale * torch.randn_like(p)

        # --- Record loss traces after burn-in ---
        if step >= burn_in:
            test_traces[draw_idx] = eval_per_sample_losses(
                model, test_inputs, test_seq_lens, test_targets)
            train_traces[draw_idx] = eval_per_sample_losses(
                model, eval_train_inputs, eval_train_seq_lens, eval_train_targets)
            draw_idx += 1

            if draw_idx % 200 == 0:
                print(f"    Draw {draw_idx}/{num_draws}", flush=True)

    return test_traces.cpu().numpy(), train_traces.cpu().numpy()


# -------------------------
# Susceptibility
# -------------------------
def compute_susceptibility(test_traces, train_traces):
    """
    χ_{i,j} = -Cov[ℓ_test_i(w), ℓ_train_j(w)]

    test_traces:  [num_draws, n_test]
    train_traces: [num_draws, n_train_eval]

    Returns: [n_test, n_train_eval]
    """
    num_draws = test_traces.shape[0]
    test_mean = test_traces.mean(axis=0)
    train_mean = train_traces.mean(axis=0)
    cross = (test_traces.T @ train_traces) / num_draws
    cov = cross - np.outer(test_mean, train_mean)
    return -cov


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_val", type=float, default=None,
                        help="Single lambda value (default: all 6 lambdas)")
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--nbeta", type=float, default=DEFAULT_NBETA)
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--burn_in", type=int, default=DEFAULT_BURN_IN)
    parser.add_argument("--sgld_batch_size", type=int, default=DEFAULT_SGLD_BS)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda")

    lambdas = [args.lambda_val] if args.lambda_val is not None else ALL_LAMBDAS

    # Partition runs across workers
    run_ids = get_run_ids()
    my_runs = [r for i, r in enumerate(run_ids) if i % args.num_workers == args.worker_id]
    # A model is pending if ANY lambda matrix is missing
    pending = [r for r in my_runs
               if any(not os.path.exists(matrix_path(r, lam)) for lam in lambdas)]
    print(f"Worker {args.worker_id}/{args.num_workers}: "
          f"{len(pending)}/{len(my_runs)} models pending "
          f"({len(lambdas)} lambdas)", flush=True)

    if not pending:
        print("Nothing to do.")
        return

    # Load data
    print("Loading training data (100K subset, seed=42)...", flush=True)
    train_inputs, train_seq_lens, train_targets = load_train_subset()
    train_inputs = train_inputs.to(device)
    train_seq_lens = train_seq_lens.to(device)
    train_targets = train_targets.to(device)

    print("Loading test data (1000 OOD)...", flush=True)
    test_inputs, test_seq_lens, test_targets = load_test_data()
    test_inputs = test_inputs.to(device)
    test_seq_lens = test_seq_lens.to(device)
    test_targets = test_targets.to(device)

    # Load shared indices — union for eval, per-lambda for slicing
    union_idx, lambda_positions = load_all_shared_indices(lambdas)
    print(f"Union of shared train indices: {len(union_idx)} "
          f"(from {len(lambdas)} lambdas)", flush=True)

    eval_train_inputs = train_inputs[union_idx]
    eval_train_seq_lens = train_seq_lens[union_idx]
    eval_train_targets = train_targets[union_idx]

    print(f"SGLD: eps={args.epsilon}, gamma={args.gamma}, nbeta={args.nbeta}, "
          f"steps={args.num_steps}, burn_in={args.burn_in}, "
          f"sgld_bs={args.sgld_batch_size}", flush=True)

    for i, run_id in enumerate(pending):
        print(f"\n[{i+1}/{len(pending)}] run {run_id}", flush=True)

        test_traces, train_traces = run_sgld(
            run_id, train_inputs, train_seq_lens, train_targets,
            eval_train_inputs, eval_train_seq_lens, eval_train_targets,
            test_inputs, test_seq_lens, test_targets,
            args.epsilon, args.gamma, args.nbeta,
            args.num_steps, args.burn_in, args.sgld_batch_size)

        # Compute & save one susceptibility matrix per lambda
        for lam in lambdas:
            out = matrix_path(run_id, lam)
            if os.path.exists(out):
                continue
            pos = lambda_positions[lam]
            susc = compute_susceptibility(test_traces, train_traces[:, pos])
            np.save(out, susc)
            print(f"  lambda={lam:.0e}: saved [{susc.shape[0]}x{susc.shape[1]}]", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
