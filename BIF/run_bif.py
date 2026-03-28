"""
BIF matrix computation for each trained model.

For each of the 50 models (bs=8, lr=1e-4):
1. Load checkpoint w*
2. Run SGLD (gradient steps on full 100K training set)
3. Collect per-sample loss traces on:
     - union of shared top-1000 train indices (across all lambdas)
     - 1000 OOD test samples
4. Compute BIF matrix: -Cov[ℓ_test_i(w), ℓ_train_j(w)]
5. Slice per-lambda to produce one [1000 x 1000] matrix per (model, lambda)

Mirrors susceptibility/susceptibility.py but uses the generic BIF module.

Usage:
  python run_bif.py --worker_id 0 --num_workers 8
  python run_bif.py --lambda_val 1e-6    # single lambda only
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import BracketsDataset
from utils.model import get_transformer, N_CLASSES
from BIF.bif import BIFConfig, BIFResults, compute_bif, compute_covariance

# -------------------------
# Paths
# -------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

INFLUENCE_DIR    = os.path.join(_ROOT, "influence")
TRAIN_CSV        = os.path.join(INFLUENCE_DIR, "data/train_binomial(40,0.5).csv")
OOD_CSV          = os.path.join(INFLUENCE_DIR, "data/test_binomial(40,0.5).csv")
RESULTS_CSV      = os.path.join(INFLUENCE_DIR, "logs/1Mexp2_bin_40_05_Transformer.csv")
MODELS_DIR       = os.path.join(_ROOT, "susceptibility", "models")
OUTPUT_DIR       = os.path.join(_HERE, "results")
LAMBDA_SWEEP_DIR = os.path.join(INFLUENCE_DIR, "logs/lambda_sweep")

# -------------------------
# Experiment config
# -------------------------
TARGET_BS    = 8
TARGET_LR    = 0.0001
TRAIN_SUBSET = 100_000
TRAIN_SEED   = 42

ALL_LAMBDAS = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]

DEFAULT_EPSILON   = 3e-6
DEFAULT_GAMMA     = 500.0
DEFAULT_NBETA     = 100.0
DEFAULT_NUM_STEPS = 2000
DEFAULT_BURN_IN   = 500
DEFAULT_SGLD_BS   = 256


# -------------------------
# Helpers
# -------------------------
def lambda_tag(lam):
    return f"lambda{lam:.0e}".replace("e-0", "e-").replace("e+0", "e")


def matrix_path(run_id, lam):
    return os.path.join(OUTPUT_DIR, f"{run_id}_bif_{lambda_tag(lam)}.npy")


def get_run_ids():
    df = pd.read_csv(RESULTS_CSV)
    final = df[
        (df["batch_size"] == TARGET_BS) &
        (df["lr"] == TARGET_LR) &
        (df["datapoints_seen"] == df["datapoints_seen"].max())
    ]
    return sorted(final["run_id"].tolist())


def load_model(run_id, device):
    ckpt = os.path.join(MODELS_DIR, f"{run_id}.pt")
    model = get_transformer(n_layer=3, n_head=4, n_embd=64,
                            embd_pdrop=0, resid_pdrop=0, attn_pdrop=0)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.to(device)


def load_shared_indices(lam):
    path = os.path.join(LAMBDA_SWEEP_DIR, f"shared_top1000_{lambda_tag(lam)}.npy")
    return np.load(path)


def load_all_shared_indices(lambdas):
    """Union of shared indices across lambdas + per-lambda positions within the union."""
    per_lambda = {}
    all_idx = set()
    for lam in lambdas:
        idx = load_shared_indices(lam)
        per_lambda[lam] = idx
        all_idx.update(idx.tolist())
    union_idx = np.array(sorted(all_idx))
    union_pos = {v: i for i, v in enumerate(union_idx)}
    lambda_positions = {
        lam: np.array([union_pos[i] for i in idx])
        for lam, idx in per_lambda.items()
    }
    return union_idx, lambda_positions


# -------------------------
# Dataset wrapper
# -------------------------
class BracketsSubset(Dataset):
    """Wraps pre-loaded GPU tensors (inputs, seq_lens, targets)."""

    def __init__(self, inputs, seq_lens, targets):
        self.inputs   = inputs
        self.seq_lens = seq_lens
        self.targets  = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.seq_lens[idx], self.targets[idx]


def load_train_subset(device):
    df = pd.read_csv(TRAIN_CSV)
    df = df.sample(n=TRAIN_SUBSET, random_state=TRAIN_SEED).reset_index(drop=True)
    bd = BracketsDataset(df)
    inputs   = bd.toks.to(device)
    seq_lens = torch.tensor([len(s) + 2 for s in bd.strs], device=device)
    targets  = bd.ylabels.long().to(device)
    return inputs, seq_lens, targets


def load_test_data(device):
    df = pd.read_csv(OOD_CSV)
    bd = BracketsDataset(df)
    inputs   = bd.toks.to(device)
    seq_lens = torch.tensor([len(s) + 2 for s in bd.strs], device=device)
    targets  = bd.ylabels.long().to(device)
    return inputs, seq_lens, targets


# -------------------------
# Loss function
# -------------------------
def get_preds(model, inputs, seq_lens):
    outputs = model(inputs)[0]
    last_idx = (seq_lens - 1).unsqueeze(1).repeat(1, N_CLASSES)
    return outputs.gather(1, last_idx.unsqueeze(1))[:, 0, :]


def brackets_loss_fn(model, batch):
    """Per-example cross-entropy for the brackets classification task."""
    inputs, seq_lens, targets = batch
    preds = get_preds(model, inputs, seq_lens)
    return F.cross_entropy(preds, targets, reduction="none")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_val",      type=float, default=None)
    parser.add_argument("--worker_id",       type=int,   default=0)
    parser.add_argument("--num_workers",     type=int,   default=1)
    parser.add_argument("--epsilon",         type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--gamma",           type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--nbeta",           type=float, default=DEFAULT_NBETA)
    parser.add_argument("--num_steps",       type=int,   default=DEFAULT_NUM_STEPS)
    parser.add_argument("--burn_in",         type=int,   default=DEFAULT_BURN_IN)
    parser.add_argument("--sgld_batch_size", type=int,   default=DEFAULT_SGLD_BS)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda")

    lambdas = [args.lambda_val] if args.lambda_val is not None else ALL_LAMBDAS

    run_ids = get_run_ids()
    my_runs = [r for i, r in enumerate(run_ids) if i % args.num_workers == args.worker_id]
    pending = [r for r in my_runs
               if any(not os.path.exists(matrix_path(r, lam)) for lam in lambdas)]
    print(f"Worker {args.worker_id}/{args.num_workers}: "
          f"{len(pending)}/{len(my_runs)} models pending "
          f"({len(lambdas)} lambdas)", flush=True)

    if not pending:
        print("Nothing to do.")
        return

    # -----------------------------------------------------------------
    # Load data (once, on GPU)
    # -----------------------------------------------------------------
    print("Loading training data (100K subset, seed=42)...", flush=True)
    train_inputs, train_seq_lens, train_targets = load_train_subset(device)

    print("Loading test data (1000 OOD)...", flush=True)
    test_inputs, test_seq_lens, test_targets = load_test_data(device)

    # Union of shared indices: eval traces cover all lambdas in one SGLD run
    union_idx, lambda_positions = load_all_shared_indices(lambdas)
    print(f"Union of shared train indices: {len(union_idx)} "
          f"(from {len(lambdas)} lambdas)", flush=True)

    # Full 100K — used for SGLD gradient steps
    sgld_dataset = BracketsSubset(train_inputs, train_seq_lens, train_targets)

    # Union subset (~1K-6K) — used for trace collection
    train_eval_ds = BracketsSubset(
        train_inputs[union_idx],
        train_seq_lens[union_idx],
        train_targets[union_idx],
    )

    # 1K OOD test — query set
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
          f"burn_in={config.burn_in}, sgld_bs={config.sgld_batch_size}", flush=True)

    # -----------------------------------------------------------------
    # Per-model loop
    # -----------------------------------------------------------------
    for i, run_id in enumerate(pending):
        print(f"\n[{i+1}/{len(pending)}] run {run_id}", flush=True)

        model = load_model(run_id, device)

        # SGLD runs on full 100K; trace collection on union subset + query
        results = compute_bif(
            model              = model,
            train_dataset      = sgld_dataset,
            query_dataset      = query_ds,
            loss_fn            = brackets_loss_fn,
            config             = config,
            train_eval_dataset = train_eval_ds,
            retain_traces      = True,
        )
        # results.train_trace: [len(union_idx), n_draws]
        # results.query_trace: [1000,           n_draws]

        # Slice per-lambda and save
        for lam in lambdas:
            out = matrix_path(run_id, lam)
            if os.path.exists(out):
                continue
            pos = lambda_positions[lam]                       # indices into union
            train_trace_lam = results.train_trace[pos, :]    # [1000, n_draws]
            bif_mat, _ = compute_covariance(train_trace_lam, results.query_trace)
            # Shape: [n_query=1000, n_train=1000] — same convention as susceptibility
            np.save(out, bif_mat.numpy())
            print(f"  lambda={lam:.0e}: saved {bif_mat.shape}", flush=True)

        del model
        torch.cuda.empty_cache()

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
