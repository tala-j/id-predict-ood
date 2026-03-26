"""
CKA analysis of attention-head susceptibility vectors.

Each model produces a [3, 4] chi matrix (layers x heads).
We flatten to a 12-dim vector, compute pairwise CKA across all models,
and check whether CKA separates models by OOD accuracy.

Outputs (in cka_results/):
  cpt_cka.png                  — pairwise CKA heatmap sorted by OOD accuracy
  cpt_susceptibility_vectors.png — raw [n_models x 12] chi heatmap
  cpt_cka.npz                  — cka_matrix, names, ood_accs

Usage:
  python cka.py
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE      = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
OUTPUT_DIR  = os.path.join(_HERE, "cka_results")
CSV_PATH    = os.path.join(_HERE, "..", "influence", "logs",
                           "1Mexp2_bin_40_05_Transformer.csv")

N_LAYER = 3
N_HEAD  = 4


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def load_ood_acc():
    df = pd.read_csv(CSV_PATH)
    return dict(zip(df["run_id"].astype(str), df["ood_acc"].astype(float)))


def load_chi_matrices():
    """
    Returns (names, S) where S is [n_models, 12] float64.
    If chi is [3, 4, n_obs], aggregate to [3, 4] by taking mean over probe samples.
    """
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_cpt.npy")))
    if not files:
        raise FileNotFoundError(f"No *_cpt.npy files found in {RESULTS_DIR}")

    names, vectors = [], []
    for f in files:
        name = os.path.basename(f).replace("_cpt.npy", "")
        chi  = np.load(f).astype(np.float64)
        if chi.ndim == 3:
            chi = chi.mean(axis=2)   # [3, 4]
        names.append(name)
        vectors.append(chi.flatten())

    return names, np.stack(vectors, axis=0)     # [n_models, 12]


# ------------------------------------------------------------------
# CKA
# ------------------------------------------------------------------
def center_matrix(K):
    """Double-center a square gram matrix."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X, Y):
    """
    Linear CKA between two [n_samples, n_features] matrices.
    Uses the Gram-matrix formulation: CKA(X,Y) = HSIC(XXT, YYT) / sqrt(HSIC(XXT,XXT)*HSIC(YYT,YYT))
    """
    K_X = center_matrix(X @ X.T)
    K_Y = center_matrix(Y @ Y.T)
    hsic_xy = (K_X * K_Y).sum()
    hsic_xx = (K_X * K_X).sum()
    hsic_yy = (K_Y * K_Y).sum()
    if hsic_xx == 0 or hsic_yy == 0:
        return 0.0
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def pairwise_cka(S):
    """
    Compute pairwise CKA treating each row of S as a 1-d feature vector.
    CKA between two scalars is just the squared correlation, so we treat
    each model's 12-dim vector as a point in 12-d feature space and compute
    the normalised dot-product Gram kernel.

    S: [n_models, 12]
    Returns: [n_models, n_models] CKA matrix
    """
    n = S.shape[0]
    # Gram matrix via outer dot-products (each row is a 12-d vector)
    # CKA(i,j): treat model i and model j as singleton "datasets" would collapse.
    # Instead we use the standard approach: treat S itself as a [n x 12] feature
    # matrix and compute column-wise Gram kernels.
    #
    # We want to compare model pairs on their 12-d fingerprints.
    # Standard approach: compute pairwise cosine/CKA of the feature vectors.
    # Here we use the "feature CKA" variant:
    #   K_i = outer(S[i], S[i])  as a 1x1 gram → reduces to cosine similarity.
    # For meaningful CKA we need the n-sample Gram interpretation.
    #
    # We treat the 12 head-susceptibilities as 12 "observations" of a scalar
    # feature.  K_X[i] = outer(S[i], S[i]) ∈ R^{12x12} is the head-Gram for
    # model i.  CKA(i,j) then compares head similarity patterns.
    grams = []
    for i in range(n):
        v = S[i:i+1]          # [1, 12]
        K = center_matrix(v.T @ v)   # [12, 12]
        grams.append(K)

    cka_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            hsic_xy = (grams[i] * grams[j]).sum()
            hsic_xx = (grams[i] * grams[i]).sum()
            hsic_yy = (grams[j] * grams[j]).sum()
            if hsic_xx == 0 or hsic_yy == 0:
                cka_mat[i, j] = 0.0
            else:
                cka_mat[i, j] = hsic_xy / np.sqrt(hsic_xx * hsic_yy)
    return cka_mat


# ------------------------------------------------------------------
# Normalisations
# ------------------------------------------------------------------
def zscore_cols(M):
    mu    = M.mean(axis=0, keepdims=True)
    sigma = M.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1.0
    return (M - mu) / sigma


# ------------------------------------------------------------------
# Pearson r (no scipy needed)
# ------------------------------------------------------------------
def pearson_r(x, y):
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm**2).sum()) * np.sqrt((ym**2).sum())
    return float((xm * ym).sum() / denom) if denom > 0 else 0.0


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
def plot_cka_heatmap(cka_mat, labels, title, path):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.35), max(9, n * 0.3)))
    im = ax.imshow(cka_mat, vmin=0, vmax=1, cmap="viridis")
    if n <= 40:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_vector_heatmap(S, labels, head_labels, title, path):
    """Heatmap of raw chi values: rows=models sorted by OOD acc, cols=heads."""
    fig, ax = plt.subplots(figsize=(max(8, len(head_labels) * 0.5),
                                    max(6, len(labels) * 0.25)))
    vmax = np.abs(S).max()
    im = ax.imshow(S, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(head_labels)))
    ax.set_xticklabels(head_labels, rotation=45, ha="right", fontsize=7)
    if len(labels) <= 60:
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
    else:
        ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Attention head")
    ax.set_ylabel("Model (sorted by OOD acc ↑)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="χ")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ood_acc = load_ood_acc()
    names, S = load_chi_matrices()   # S: [n_models, 12]
    n = len(names)
    print(f"Loaded {n} chi matrices  (shape {S.shape})")

    # Sort by OOD accuracy (ascending)
    order   = sorted(range(n), key=lambda i: ood_acc.get(names[i], 0.0))
    names   = [names[i]  for i in order]
    S       = S[order]
    accs    = np.array([ood_acc.get(nm, 0.0) for nm in names])

    labels      = [f"{nm}\n({accs[i]:.3f})" for i, nm in enumerate(names)]
    head_labels = [f"L{l}H{h}" for l in range(N_LAYER) for h in range(N_HEAD)]

    # ---- raw vector heatmap ----
    plot_vector_heatmap(
        S, labels, head_labels,
        "CPT Susceptibility χ per head  [sorted by OOD acc ↑]",
        os.path.join(OUTPUT_DIR, "cpt_susceptibility_vectors.png"))

    # ---- z-score normalise columns before CKA ----
    S_z = zscore_cols(S)

    # ---- pairwise CKA ----
    print("Computing pairwise CKA...")
    cka_mat = pairwise_cka(S_z)

    mask = ~np.eye(n, dtype=bool)
    print(f"CKA — mean: {cka_mat[mask].mean():.4f}  "
          f"std: {cka_mat[mask].std():.4f}  "
          f"min: {cka_mat[mask].min():.4f}  "
          f"max: {cka_mat[mask].max():.4f}")

    # ---- Pearson r: CKA vs |Δ OOD acc| ----
    i_idx, j_idx = np.triu_indices(n, k=1)
    cka_vals  = cka_mat[i_idx, j_idx]
    delta_acc = np.abs(accs[i_idx] - accs[j_idx])
    r = pearson_r(cka_vals, delta_acc)
    print(f"Pearson r(CKA, |Δ OOD acc|) = {r:.4f}")
    print("  (negative r → higher CKA when models have similar OOD acc — good separation)")

    # ---- plot CKA heatmap ----
    plot_cka_heatmap(
        cka_mat, labels,
        f"CPT Susceptibility CKA  [sorted by OOD acc]  r={r:.3f}",
        os.path.join(OUTPUT_DIR, "cpt_cka.png"))

    # ---- also plot per-layer CKA heatmaps ----
    for layer in range(N_LAYER):
        cols = [layer * N_HEAD + h for h in range(N_HEAD)]
        S_layer = zscore_cols(S[:, cols])
        cka_l   = pairwise_cka(S_layer)
        r_l = pearson_r(cka_l[i_idx, j_idx], delta_acc)
        plot_cka_heatmap(
            cka_l, labels,
            f"CPT Susceptibility CKA — Layer {layer}  r={r_l:.3f}",
            os.path.join(OUTPUT_DIR, f"cpt_cka_layer{layer}.png"))
        print(f"  Layer {layer}: Pearson r = {r_l:.4f}")

    # ---- save ----
    np.savez(
        os.path.join(OUTPUT_DIR, "cpt_cka.npz"),
        cka_matrix=cka_mat,
        chi_matrix=S,
        names=np.array(names),
        ood_accs=accs,
    )
    print(f"  Saved {os.path.join(OUTPUT_DIR, 'cpt_cka.npz')}")
    print("\nDone.")


if __name__ == "__main__":
    main()
