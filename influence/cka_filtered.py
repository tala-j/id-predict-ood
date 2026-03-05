"""
CKA on influence matrices after filtering to high-variance training points.

Most of the 100K training points have negligible influence variance across
test queries — they contribute noise that drowns out signal in CKA.
This script keeps only the top-K training points by average variance
(pooled across all models) and recomputes CKA.

Sweeps over multiple K values to show sensitivity.
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "cka_filtered_results")

# Top-K training points to keep (by average cross-model variance)
K_VALUES = [100, 500, 1000, 5000, 10000]


# ------------------------------------------------------------------
# CKA utilities (same as cka.py)
# ------------------------------------------------------------------

def linear_cka_gram(K_X, K_Y):
    hsic_xy = (K_X * K_Y).sum()
    hsic_xx = (K_X * K_X).sum()
    hsic_yy = (K_Y * K_Y).sum()
    if hsic_xx == 0 or hsic_yy == 0:
        return 0.0
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def centered_gram(M):
    M_c = M - M.mean(axis=0, keepdims=True)
    return M_c @ M_c.T


def zscore_columns(M):
    mu = M.mean(axis=0, keepdims=True)
    sigma = M.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    return (M - mu) / sigma


def rankify_columns(M):
    return rankdata(M, axis=0, method="average").astype(np.float64)


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
# Loading helper
# ------------------------------------------------------------------

def load_influence(path):
    raw = np.load(path, allow_pickle=True)
    if raw.dtype == object:
        d = raw.item()
        M = list(d.values())[0]
        if hasattr(M, "numpy"):
            M = M.numpy()
    else:
        M = raw
    return M.astype(np.float64)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_influence.npy")))
    if not files:
        raise FileNotFoundError(f"No *_influence.npy in {RESULTS_DIR}")

    names = [os.path.basename(f).replace("_influence.npy", "") for f in files]
    n_models = len(names)
    short_names = [nm.replace("nesting_seed", "s") for nm in names]

    print(f"Found {n_models} influence matrices")

    # ------------------------------------------------------------------
    # Pass 1: compute per-column variance for each model
    # ------------------------------------------------------------------
    print("\nPass 1: Computing per-column variances...")
    n_train = None
    all_vars = []

    for i, (f, name) in enumerate(zip(files, names)):
        t0 = time.time()
        M = load_influence(f)
        if n_train is None:
            n_train = M.shape[1]
        col_var = M.var(axis=0)  # variance across test points
        all_vars.append(col_var)
        print(f"  [{i+1}/{n_models}] {name}: shape {M.shape}, "
              f"var range [{col_var.min():.2e}, {col_var.max():.2e}], "
              f"{time.time()-t0:.1f}s")
        del M

    # Average variance across models
    avg_var = np.mean(all_vars, axis=0)  # (n_train,)
    del all_vars

    # Rank columns by average variance (descending)
    var_order = np.argsort(avg_var)[::-1]

    print(f"\nAvg variance: min={avg_var.min():.2e}, "
          f"median={np.median(avg_var):.2e}, max={avg_var.max():.2e}")

    # ------------------------------------------------------------------
    # Pass 2: for each K, filter columns and compute CKA
    # ------------------------------------------------------------------
    for K in K_VALUES:
        K = min(K, n_train)
        top_cols = var_order[:K]

        print(f"\n{'='*60}")
        print(f"K = {K:,} (top {100*K/n_train:.1f}% of training points)")
        var_threshold = avg_var[top_cols[-1]]
        print(f"  Variance threshold: {var_threshold:.2e}")
        print(f"{'='*60}")

        grams_z = []
        grams_r = []

        for i, (f, name) in enumerate(zip(files, names)):
            t0 = time.time()
            M = load_influence(f)
            M_filt = M[:, top_cols]  # (1000, K)
            del M

            grams_z.append(centered_gram(zscore_columns(M_filt)))
            grams_r.append(centered_gram(rankify_columns(M_filt)))
            del M_filt
            print(f"  [{i+1}/{n_models}] {name}: {time.time()-t0:.1f}s")

        # CKA matrices
        cka_z = np.zeros((n_models, n_models))
        cka_r = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                cka_z[i, j] = linear_cka_gram(grams_z[i], grams_z[j])
                cka_r[i, j] = linear_cka_gram(grams_r[i], grams_r[j])

        print_table(cka_z, names, f"Z-score CKA (K={K:,}):")
        print_table(cka_r, names, f"Rank CKA (K={K:,}):")

        # Off-diagonal summary
        mask = ~np.eye(n_models, dtype=bool)
        print(f"\n  Off-diagonal summary:")
        print(f"    Z-score: mean={cka_z[mask].mean():.4f}, "
              f"min={cka_z[mask].min():.4f}, max={cka_z[mask].max():.4f}")
        print(f"    Rank:    mean={cka_r[mask].mean():.4f}, "
              f"min={cka_r[mask].min():.4f}, max={cka_r[mask].max():.4f}")

        # Heatmaps
        plot_heatmap(cka_z, short_names,
                     f"CKA Z-score (top {K:,} training pts)",
                     os.path.join(OUTPUT_DIR, f"cka_zscore_K{K}.png"))
        plot_heatmap(cka_r, short_names,
                     f"CKA Rank (top {K:,} training pts)",
                     os.path.join(OUTPUT_DIR, f"cka_rank_K{K}.png"))

        np.savez(os.path.join(OUTPUT_DIR, f"cka_K{K}.npz"),
                 cka_zscore=cka_z, cka_rank=cka_r,
                 names=np.array(names), K=K,
                 top_cols=top_cols)

    # ------------------------------------------------------------------
    # Summary plot: mean off-diagonal CKA vs K
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    means_z, means_r = [], []
    for K in K_VALUES:
        K = min(K, n_train)
        data = np.load(os.path.join(OUTPUT_DIR, f"cka_K{K}.npz"))
        mask = ~np.eye(n_models, dtype=bool)
        means_z.append(data["cka_zscore"][mask].mean())
        means_r.append(data["cka_rank"][mask].mean())

    ax.plot(K_VALUES, means_z, "o-", label="Z-score CKA")
    ax.plot(K_VALUES, means_r, "s-", label="Rank CKA")
    ax.set_xlabel("K (number of top training points kept)")
    ax.set_ylabel("Mean off-diagonal CKA")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("CKA vs. feature dimensionality")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cka_vs_K.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved cka_vs_K.png")

    # ------------------------------------------------------------------
    # Summary table: mean off-diagonal CKA only
    # ------------------------------------------------------------------
    rows = []
    for K in K_VALUES:
        K = min(K, n_train)
        data = np.load(os.path.join(OUTPUT_DIR, f"cka_K{K}.npz"))
        mask = ~np.eye(n_models, dtype=bool)
        cz, cr = data["cka_zscore"], data["cka_rank"]
        rows.append([f"{K:,}", f"{cz[mask].mean():.3f}", f"{cr[mask].mean():.3f}"])

    # Add full 100K baseline
    orig = os.path.join(os.path.dirname(__file__), "cka_results", "cka_matrices.npz")
    if os.path.exists(orig):
        data = np.load(orig)
        mask = ~np.eye(n_models, dtype=bool)
        cz, cr = data["cka_zscore"], data["cka_rank"]
        rows.append(["100,000", f"{cz[mask].mean():.3f}", f"{cr[mask].mean():.3f}"])

    col_labels = ["K (training points)", "Mean Z-score CKA", "Mean Rank CKA"]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.7)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row shading
    for i in range(1, len(rows) + 1):
        color = "#D9E2F3" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

    ax.set_title("Mean pairwise CKA across 8 nesting models\n"
                 "by number of top-variance training points retained",
                 fontsize=12, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cka_summary_table.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cka_summary_table.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
