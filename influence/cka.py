"""
Pairwise CKA between influence matrices.

Two normalizations (per column = per training point):
  1. Z-score: subtract mean, divide by std
  2. Rank: convert to ranks

Uses the Gram-matrix formulation of linear CKA for efficiency:
  K = M_centered @ M_centered^T  (1000 x 1000 instead of 100K x 100K)
  CKA(X, Y) = sum(K_X * K_Y) / sqrt(sum(K_X^2) * sum(K_Y^2))
"""

import os
import glob
import time

import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "influence_results")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "cka_results")


# ------------------------------------------------------------------
# CKA
# ------------------------------------------------------------------

def linear_cka_gram(K_X, K_Y):
    """Linear CKA from precomputed centered Gram matrices."""
    hsic_xy = (K_X * K_Y).sum()
    hsic_xx = (K_X * K_X).sum()
    hsic_yy = (K_Y * K_Y).sum()
    if hsic_xx == 0 or hsic_yy == 0:
        return 0.0
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def centered_gram(M):
    """Column-center M, then compute Gram matrix K = M_c @ M_c^T."""
    M_c = M - M.mean(axis=0, keepdims=True)
    return M_c @ M_c.T


# ------------------------------------------------------------------
# Normalizations
# ------------------------------------------------------------------

def zscore_columns(M):
    """Z-score each column (subtract mean, divide by std)."""
    mu = M.mean(axis=0, keepdims=True)
    sigma = M.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    return (M - mu) / sigma


def rankify_columns(M):
    """Rank-transform each column independently."""
    return rankdata(M, axis=0, method="average").astype(np.float64)


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_heatmap(cka_mat, names, title, path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cka_mat, vmin=0, vmax=1, cmap="viridis")

    n = len(names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)

    for i in range(n):
        for j in range(n):
            color = "white" if cka_mat[i, j] < 0.5 else "black"
            ax.text(j, i, f"{cka_mat[i, j]:.3f}",
                    ha="center", va="center", color=color, fontsize=8)

    ax.set_title(title, fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def print_table(cka_mat, names, title):
    n = len(names)
    print(f"\n{title}")
    header = f"  {'':>20s}" + "".join(f"{nm:>14s}" for nm in names)
    print(header)
    for i in range(n):
        row = f"  {names[i]:>20s}" + "".join(
            f"{cka_mat[i, j]:14.4f}" for j in range(n)
        )
        print(row)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_influence.npy")))
    if not files:
        raise FileNotFoundError(f"No *_influence.npy in {RESULTS_DIR}")

    names = [os.path.basename(f).replace("_influence.npy", "") for f in files]
    n = len(names)

    print(f"Found {n} influence matrices:")
    for name in names:
        print(f"  {name}")

    # Compute Gram matrices (load one influence matrix at a time)
    grams_zscore = []
    grams_rank = []

    for i, (f, name) in enumerate(zip(files, names)):
        t0 = time.time()
        print(f"\n[{i+1}/{n}] {name}")
        raw = np.load(f, allow_pickle=True)
        # kronfluence saves as {module_name: tensor} wrapped in a numpy object array
        if raw.dtype == object:
            d = raw.item()
            M = list(d.values())[0]
            if hasattr(M, "numpy"):
                M = M.numpy()
        else:
            M = raw
        M = M.astype(np.float64)
        print(f"  Shape: {M.shape}  ({M.nbytes / 1e9:.2f} GB)")

        # Z-score normalization → Gram
        t1 = time.time()
        grams_zscore.append(centered_gram(zscore_columns(M)))
        print(f"  Z-score Gram: {time.time() - t1:.1f}s")

        # Rank transformation → Gram
        t1 = time.time()
        grams_rank.append(centered_gram(rankify_columns(M)))
        print(f"  Rank Gram:    {time.time() - t1:.1f}s")

        del M
        print(f"  Total: {time.time() - t0:.1f}s")

    # Pairwise CKA
    cka_z = np.zeros((n, n))
    cka_r = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cka_z[i, j] = linear_cka_gram(grams_zscore[i], grams_zscore[j])
            cka_r[i, j] = linear_cka_gram(grams_rank[i], grams_rank[j])

    # Print
    print_table(cka_z, names, "Z-score normalized CKA:")
    print_table(cka_r, names, "Rank-transformed CKA:")

    # Heatmaps
    short_names = [nm.replace("nesting_seed", "s") for nm in names]
    plot_heatmap(cka_z, short_names, "CKA (Z-score normalized)",
                 os.path.join(OUTPUT_DIR, "cka_zscore.png"))
    plot_heatmap(cka_r, short_names, "CKA (Rank-transformed)",
                 os.path.join(OUTPUT_DIR, "cka_rank.png"))

    # Save raw matrices
    np.savez(os.path.join(OUTPUT_DIR, "cka_matrices.npz"),
             cka_zscore=cka_z, cka_rank=cka_r, names=np.array(names))
    print(f"\nSaved cka_matrices.npz")

    print("\nDone.")


if __name__ == "__main__":
    main()
