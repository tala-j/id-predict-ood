"""
Pairwise CKA between BIF matrices across the 50 models.

For each lambda, loads the [1000x1000] BIF matrices, computes pairwise CKA
(z-score and rank normalized), and saves heatmaps sorted by OOD accuracy.

Mirrors susceptibility/cka.py exactly, but reads *_bif_*.npy instead of
*_susc_*.npy.

Usage:
  python cka_bif.py                     # all lambdas
  python cka_bif.py --lambda_val 1e-6   # single lambda
"""

import os, argparse, glob
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
OUTPUT_DIR  = os.path.join(_HERE, "cka_results")
CSV_PATH    = os.path.join(_HERE, "..", "influence", "logs",
                           "1Mexp2_bin_40_05_Transformer.csv")

ALL_LAMBDAS = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]


def lambda_tag(lam):
    return f"lambda{lam:.0e}".replace("e-0", "e-").replace("e+0", "e")


def load_ood_acc():
    df = pd.read_csv(CSV_PATH)
    return dict(zip(df["run_id"], df["ood_acc"]))


# ------------------------------------------------------------------
# CKA
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


# ------------------------------------------------------------------
# Normalizations
# ------------------------------------------------------------------
def zscore_columns(M):
    mu    = M.mean(axis=0, keepdims=True)
    sigma = M.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    return (M - mu) / sigma


def rankify_columns(M):
    return rankdata(M, axis=0, method="average").astype(np.float64)


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def plot_heatmap(cka_mat, names, title, path):
    n = len(names)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cka_mat, vmin=0, vmax=1, cmap="viridis")

    if n <= 20:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        for i in range(n):
            for j in range(n):
                color = "white" if cka_mat[i, j] < 0.5 else "black"
                ax.text(j, i, f"{cka_mat[i, j]:.2f}",
                        ha="center", va="center", color=color, fontsize=5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(title, fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ------------------------------------------------------------------
# Per-lambda
# ------------------------------------------------------------------
def run_cka_for_lambda(lam, ood_acc):
    tag   = lambda_tag(lam)
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"*_bif_{tag}.npy")))
    if not files:
        print(f"  No matrices for {tag}, skipping.")
        return

    names = [os.path.basename(f).replace(f"_bif_{tag}.npy", "") for f in files]

    # Sort by OOD accuracy ascending (low → high, left → right)
    order = sorted(range(len(names)), key=lambda i: ood_acc.get(names[i], 0.0))
    files = [files[i] for i in order]
    names = [names[i] for i in order]

    n = len(names)
    print(f"\n{tag}: {n} matrices")

    grams_z, grams_r = [], []
    for f in files:
        M = np.load(f).astype(np.float64)
        grams_z.append(centered_gram(zscore_columns(M)))
        grams_r.append(centered_gram(rankify_columns(M)))

    cka_z = np.zeros((n, n))
    cka_r = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cka_z[i, j] = linear_cka_gram(grams_z[i], grams_z[j])
            cka_r[i, j] = linear_cka_gram(grams_r[i], grams_r[j])

    mask = ~np.eye(n, dtype=bool)
    print(f"  Z-score CKA — mean: {cka_z[mask].mean():.4f}, "
          f"std: {cka_z[mask].std():.4f}, "
          f"min: {cka_z[mask].min():.4f}, max: {cka_z[mask].max():.4f}")
    print(f"  Rank CKA    — mean: {cka_r[mask].mean():.4f}, "
          f"std: {cka_r[mask].std():.4f}, "
          f"min: {cka_r[mask].min():.4f}, max: {cka_r[mask].max():.4f}")

    labels = [f"{nm}\n({ood_acc.get(nm, 0):.3f})" for nm in names]

    plot_heatmap(
        cka_z, labels,
        f"BIF CKA (Z-score) — {tag}  [sorted by OOD acc]",
        os.path.join(OUTPUT_DIR, f"bif_cka_zscore_{tag}.png"),
    )
    plot_heatmap(
        cka_r, labels,
        f"BIF CKA (Rank) — {tag}  [sorted by OOD acc]",
        os.path.join(OUTPUT_DIR, f"bif_cka_rank_{tag}.png"),
    )

    np.savez(
        os.path.join(OUTPUT_DIR, f"bif_cka_{tag}.npz"),
        cka_zscore=cka_z, cka_rank=cka_r, names=np.array(names),
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_val", type=float, default=None)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ood_acc = load_ood_acc()

    lambdas = [args.lambda_val] if args.lambda_val is not None else ALL_LAMBDAS
    for lam in lambdas:
        run_cka_for_lambda(lam, ood_acc)

    print("\nDone.")


if __name__ == "__main__":
    main()
