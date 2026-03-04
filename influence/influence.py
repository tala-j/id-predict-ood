"""
Compute influence matrices for the 50 models with batch_size=8, lr=0.0001.

Iterated per lambda — for each lambda value, run both passes then CKA:

  python influence.py --mode variances      --lam 1e-6 [--worker_id N --num_workers M]
  python influence.py --mode shared_indices --lam 1e-6
  python influence.py --mode matrices       --lam 1e-6 [--worker_id N --num_workers M]
  python influence.py --mode cka            --lam 1e-6

The launch script iterates over all lambda values automatically.
"""

import os
import sys
import shutil
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import ScoreArguments
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs

from utils.data import BracketsDataset
from utils.model import get_transformer, N_CLASSES

# -------------------------
# Config
# -------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

TRAIN_CSV   = os.path.join(_HERE, "data/train_binomial(40,0.5).csv")
OOD_CSV     = os.path.join(_HERE, "data/test_binomial(40,0.5).csv")
RESULTS_CSV = os.path.join(_HERE, "logs/1Mexp2_bin_40_05_Transformer.csv")
MODELS_DIR  = os.path.join(_HERE, "logs/models")
OUTPUT_DIR  = os.path.join(_HERE, "logs/lambda_sweep")

TARGET_LR    = 0.0001
TARGET_BS    = 8
TRAIN_SUBSET = 100_000
TOP_K        = 1000

LAMBDA_VALUES = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]

# Kronfluence intermediate files go to /workspace to avoid filling overlay fs
KRONFLUENCE_CACHE_DIR = "/workspace/lambda_sweep_tmp"


def lambda_tag(lam):
    return f"lambda{lam:.0e}".replace("e-0", "e-").replace("e+0", "e")


def shared_idx_path(lam):
    return os.path.join(OUTPUT_DIR, f"shared_top1000_{lambda_tag(lam)}.npy")


def variances_path(run_id, lam):
    return os.path.join(OUTPUT_DIR, f"{run_id}_var_{lambda_tag(lam)}.npy")


def matrix_path(run_id, lam):
    return os.path.join(OUTPUT_DIR, f"{run_id}_inf_{lambda_tag(lam)}.npy")


# -------------------------
# Dataset
# -------------------------
class GPTDataset(Dataset):
    def __init__(self, csv_path, max_samples=None, seed=42):
        df = pd.read_csv(csv_path)
        if max_samples is not None and max_samples < len(df):
            df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
        bd = BracketsDataset(df)
        self.inputs   = bd.toks
        self.seq_lens = torch.tensor([len(s) + 2 for s in bd.strs])
        self.targets  = bd.ylabels.long()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.seq_lens[idx], self.targets[idx]


# -------------------------
# Task
# -------------------------
class GPTTask(Task):
    def _get_preds(self, model, inputs, seq_lens):
        outputs = model(inputs)[0]
        last_idx = (seq_lens - 1).unsqueeze(1).repeat(1, N_CLASSES)
        return outputs.gather(1, last_idx.unsqueeze(1))[:, 0, :]

    def compute_train_loss(self, batch, model, sample=False):
        inputs, seq_lens, targets = batch
        preds = self._get_preds(model, inputs, seq_lens)
        if sample:
            with torch.no_grad():
                probs = torch.softmax(preds.detach(), dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            return F.cross_entropy(preds, sampled, reduction="sum")
        return F.cross_entropy(preds, targets, reduction="sum")

    def compute_measurement(self, batch, model):
        inputs, seq_lens, targets = batch
        preds = self._get_preds(model, inputs, seq_lens)
        return -F.cross_entropy(preds, targets, reduction="sum")


# -------------------------
# Model loading
# -------------------------
def load_model(ckpt_path):
    model = get_transformer(n_layer=3, n_head=4, n_embd=64,
                            embd_pdrop=0, resid_pdrop=0, attn_pdrop=0)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model


# -------------------------
# Helpers
# -------------------------
def get_run_ids():
    df = pd.read_csv(RESULTS_CSV)
    final = df[(df["batch_size"] == TARGET_BS) & (df["lr"] == TARGET_LR) &
               (df["datapoints_seen"] == df["datapoints_seen"].max())]
    return final["run_id"].tolist()


def find_ckpt(run_id):
    for sweep_dir in os.listdir(MODELS_DIR):
        candidate = os.path.join(MODELS_DIR, sweep_dir,
                                 f"run_{run_id}", f"run_{run_id}_final.pt")
        if os.path.exists(candidate):
            return candidate
    return None


def compute_scores_for_model(run_id, train_dataset, query_dataset, device, lam):
    """Fit EKFAC factors and compute scores for a single lambda value."""
    ckpt_path = find_ckpt(run_id)
    if ckpt_path is None:
        print(f"  WARNING: checkpoint not found for {run_id}, skipping.")
        return None

    model = load_model(ckpt_path)
    model = prepare_model(model=model, task=GPTTask())
    model.to(device)

    os.makedirs(KRONFLUENCE_CACHE_DIR, exist_ok=True)
    analyzer = Analyzer(
        analysis_name=run_id,
        model=model,
        task=GPTTask(),
        output_dir=KRONFLUENCE_CACHE_DIR,
    )
    analyzer.set_dataloader_kwargs(DataLoaderKwargs(num_workers=0))

    print(f"  Fitting EKFAC factors (lambda={lam:.0e})...")
    analyzer.fit_all_factors(
        factors_name="ekfac",
        dataset=train_dataset,
        overwrite_output_dir=True,
    )

    scores_name = f"pairwise_{lambda_tag(lam)}"
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name="ekfac",
        query_dataset=query_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=256,
        score_args=ScoreArguments(damping_factor=lam),
        overwrite_output_dir=True,
    )
    scores_dict = analyzer.load_pairwise_scores(scores_name)
    scores = list(scores_dict.values())[0]
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()
    scores = np.array(scores, dtype=np.float32)  # force copy into RAM

    # Delete score files and factors from /workspace immediately
    run_cache = os.path.join(KRONFLUENCE_CACHE_DIR, run_id)
    if os.path.isdir(run_cache):
        shutil.rmtree(run_cache)

    return scores  # [n_queries, n_train]


# -------------------------
# Mode: variances (pass 1 for one lambda)
# -------------------------
def run_variances(worker_id, num_workers, lam):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_ids = get_run_ids()
    my_runs = [r for i, r in enumerate(run_ids) if i % num_workers == worker_id]
    pending = [r for r in my_runs if not os.path.exists(variances_path(r, lam))]
    print(f"Worker {worker_id}/{num_workers}: {len(pending)}/{len(my_runs)} models pending "
          f"(lambda={lam:.0e}, pass 1)")

    if not pending:
        print("  Nothing to do.")
        return

    train_dataset = GPTDataset(TRAIN_CSV, max_samples=TRAIN_SUBSET)
    query_dataset = GPTDataset(OOD_CSV)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, run_id in enumerate(pending):
        print(f"\n[{i+1}/{len(pending)}] {run_id}")
        scores = compute_scores_for_model(run_id, train_dataset, query_dataset, device, lam)
        if scores is None:
            continue
        np.save(variances_path(run_id, lam), scores.var(axis=0))
        print(f"  Saved variance vector.")

    print("\nPass 1 done.")


# -------------------------
# Mode: shared_indices (for one lambda)
# -------------------------
def compute_shared_indices(lam):
    run_ids = get_run_ids()
    var_files = [variances_path(r, lam) for r in run_ids]
    missing = [f for f in var_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} variance files for lambda={lam:.0e}")
    avg_var = np.mean([np.load(f) for f in var_files], axis=0)
    top_idx = np.argsort(avg_var)[-TOP_K:]
    top_idx = top_idx[np.argsort(top_idx)]
    np.save(shared_idx_path(lam), top_idx)
    print(f"lambda={lam:.0e}: shared top-{TOP_K} indices saved "
          f"(var range [{avg_var[top_idx].min():.2e}, {avg_var[top_idx].max():.2e}])")


# -------------------------
# Mode: matrices (pass 2 for one lambda)
# -------------------------
def run_matrices(worker_id, num_workers, lam):
    if not os.path.exists(shared_idx_path(lam)):
        raise FileNotFoundError(
            f"Shared indices not found for lambda={lam:.0e}. Run --mode shared_indices first.")
    shared_idx = np.load(shared_idx_path(lam))

    run_ids = get_run_ids()
    my_runs = [r for i, r in enumerate(run_ids) if i % num_workers == worker_id]
    pending = [r for r in my_runs if not os.path.exists(matrix_path(r, lam))]
    print(f"Worker {worker_id}/{num_workers}: {len(pending)}/{len(my_runs)} models pending "
          f"(lambda={lam:.0e}, pass 2)")

    if not pending:
        print("  Nothing to do.")
        return

    train_dataset = GPTDataset(TRAIN_CSV, max_samples=TRAIN_SUBSET)
    query_dataset = GPTDataset(OOD_CSV)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, run_id in enumerate(pending):
        print(f"\n[{i+1}/{len(pending)}] {run_id}")
        scores = compute_scores_for_model(run_id, train_dataset, query_dataset, device, lam)
        if scores is None:
            continue
        matrix = scores[:, shared_idx]   # [1000, 1000]
        np.save(matrix_path(run_id, lam), matrix)
        print(f"  Saved [1000x1000] matrix.")

    print("\nPass 2 done.")


# -------------------------
# CKA utilities
# -------------------------
def zscore_columns(M):
    mu = M.mean(axis=0, keepdims=True)
    sigma = M.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    return (M - mu) / sigma


def rankify_columns(M):
    return rankdata(M, axis=0, method="average").astype(np.float64)


def centered_gram(M):
    M_c = M - M.mean(axis=0, keepdims=True)
    return M_c @ M_c.T


def linear_cka_gram(K_X, K_Y):
    hsic_xy = (K_X * K_Y).sum()
    hsic_xx = (K_X * K_X).sum()
    hsic_yy = (K_Y * K_Y).sum()
    if hsic_xx == 0 or hsic_yy == 0:
        return 0.0
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def plot_heatmap(cka_mat, names, title, path):
    n = len(names)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cka_mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    for i in range(n):
        for j in range(n):
            color = "white" if cka_mat[i, j] < 0.5 else "black"
            ax.text(j, i, f"{cka_mat[i, j]:.2f}",
                    ha="center", va="center", color=color, fontsize=5)
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# -------------------------
# Mode: cka (for one lambda, then delete matrices)
# -------------------------
def get_ood_accuracies(run_ids):
    """Return OOD accuracy for each run_id (final checkpoint)."""
    df = pd.read_csv(RESULTS_CSV)
    final = df[(df["batch_size"] == TARGET_BS) & (df["lr"] == TARGET_LR) &
               (df["datapoints_seen"] == df["datapoints_seen"].max())]
    acc_map = dict(zip(final["run_id"], final["ood_acc"]))
    return [acc_map.get(r, 0.0) for r in run_ids]


def run_cka(lam):
    run_ids = get_run_ids()
    n = len(run_ids)
    tag = lambda_tag(lam)

    missing = [matrix_path(r, lam) for r in run_ids
               if not os.path.exists(matrix_path(r, lam))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} matrices missing for lambda={lam:.0e}")

    # Sort models by OOD accuracy for interpretable heatmap ordering
    ood_accs = get_ood_accuracies(run_ids)
    sort_order = np.argsort(ood_accs)
    run_ids_sorted = [run_ids[i] for i in sort_order]
    ood_sorted = [ood_accs[i] for i in sort_order]

    print(f"Computing CKA for {n} models (lambda={lam:.0e})...")
    grams_z, grams_r = [], []
    for run_id in run_ids_sorted:
        M = np.load(matrix_path(run_id, lam)).astype(np.float64)
        grams_z.append(centered_gram(zscore_columns(M)))
        grams_r.append(centered_gram(rankify_columns(M)))

    cka_z = np.zeros((n, n))
    cka_r = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cka_z[i, j] = linear_cka_gram(grams_z[i], grams_z[j])
            cka_r[i, j] = linear_cka_gram(grams_r[i], grams_r[j])

    mask = ~np.eye(n, dtype=bool)
    print(f"  Z-score CKA: mean={cka_z[mask].mean():.4f}, "
          f"min={cka_z[mask].min():.4f}, max={cka_z[mask].max():.4f}")
    print(f"  Rank    CKA: mean={cka_r[mask].mean():.4f}, "
          f"min={cka_r[mask].min():.4f}, max={cka_r[mask].max():.4f}")

    np.savez(os.path.join(OUTPUT_DIR, f"cka_{tag}.npz"),
             cka_zscore=cka_z, cka_rank=cka_r,
             run_ids=np.array(run_ids_sorted), ood_accs=np.array(ood_sorted))

    # Labels: OOD accuracy rounded to 2 decimal places
    labels = [f"{acc:.2f}" for acc in ood_sorted]
    plot_heatmap(cka_z, labels, f"CKA z-score  λ={lam:.0e}  (sorted by OOD acc)",
                 os.path.join(OUTPUT_DIR, f"cka_z_{tag}.png"))
    plot_heatmap(cka_r, labels, f"CKA rank  λ={lam:.0e}  (sorted by OOD acc)",
                 os.path.join(OUTPUT_DIR, f"cka_r_{tag}.png"))

    # Delete matrices to free storage
    for run_id in run_ids:
        p = matrix_path(run_id, lam)
        if os.path.exists(p):
            os.remove(p)
    print(f"  Deleted matrices for lambda={lam:.0e}.")


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["variances", "shared_indices", "matrices", "cka"],
                        required=True)
    parser.add_argument("--lam",  type=float, required=True,
                        help="Lambda (damping factor) value, e.g. 1e-6")
    parser.add_argument("--worker_id",   type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    lam = args.lam
    if args.mode == "variances":
        run_variances(args.worker_id, args.num_workers, lam)
    elif args.mode == "shared_indices":
        compute_shared_indices(lam)
    elif args.mode == "matrices":
        run_matrices(args.worker_id, args.num_workers, lam)
    elif args.mode == "cka":
        run_cka(lam)
