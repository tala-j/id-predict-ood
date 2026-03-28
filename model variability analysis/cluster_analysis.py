"""
Model variability analysis: Do high-OOD-accuracy models (≥95%) exhibit more
disagreement on individual examples than low-OOD-accuracy models (≤5%)?

For each of the 50 trained models (all 3-layer, 4-head, wd=0, bs=8, lr=1e-4),
compute hard (0/1) predictions on every OOD example. Split into low (≤5%)
and high (≥95%) OOD accuracy bins, then compare:
  - Average within-bin pairwise Hamming distance (fraction of examples
    where two models disagree)
  - Hierarchical clustering dendrograms
  - HDBSCAN (if groups large enough)
  - Bootstrap stability of distance statistics
  - Heatmaps of model-by-example predictions
  - UMAP for visualization

Higher pairwise Hamming distance = models disagree on more individual
examples, suggesting multiple behavioral patterns — not necessarily
proof of mechanistically distinct algorithms.

Note: all 50 models share architecture (3-layer, 4-head, wd=0), so no
separate architecture/weight-decay control needed.
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, leaves_list
from scipy.stats import mannwhitneyu
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data import BracketsDataset
from utils.model import get_transformer, N_CLASSES

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "susceptibility" / "models"
RESULTS_CSV = ROOT / "influence" / "logs" / "1Mexp2_bin_40_05_Transformer.csv"
OOD_CSV = ROOT / "influence" / "data" / "test_binomial(40,0.5).csv"
OUT_DIR = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOW_THRESH = 0.05
HIGH_THRESH = 0.95


# -----------------------------------------------------------------------
# Load models and compute OOD hard predictions
# -----------------------------------------------------------------------
def load_model(run_id):
    ckpt = MODELS_DIR / f"{run_id}.pt"
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


def get_ood_hard_preds(run_ids):
    """Compute hard (0/1) predictions on all 1000 OOD examples. Returns [n_models, 1000]."""
    df = pd.read_csv(OOD_CSV)
    bd = BracketsDataset(df)
    inputs = bd.toks.to(DEVICE)
    seq_lens = torch.tensor([len(s) + 2 for s in bd.strs], device=DEVICE)

    all_preds = []
    for run_id in run_ids:
        model = load_model(run_id).to(DEVICE).eval()
        with torch.no_grad():
            logits = get_preds(model, inputs, seq_lens)
            pred = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(pred)
        del model
    torch.cuda.empty_cache()
    return np.stack(all_preds)  # [n_models, 1000], values in {0, 1}


# -----------------------------------------------------------------------
# Analysis helpers
# -----------------------------------------------------------------------
def group_analysis(pred_matrix, names, ood_accs, label, out_dir):
    """Analyze a group of models using Hamming distance on hard predictions."""
    n = pred_matrix.shape[0]
    n_ood = pred_matrix.shape[1]
    res = {"label": label, "n_models": n}

    if n < 2:
        print(f"  [{label}] Only {n} model(s), skipping.")
        res["skip"] = True
        return res

    # --- Pairwise Hamming distance ---
    condensed = pdist(pred_matrix, metric="hamming")  # fraction of disagreements
    dist_mat = squareform(condensed)
    res["mean_dist"] = float(condensed.mean())
    res["std_dist"] = float(condensed.std())
    res["median_dist"] = float(np.median(condensed))
    res["max_dist"] = float(condensed.max())
    res["min_dist"] = float(condensed.min())
    res["mean_disagree"] = float(condensed.mean() * n_ood)
    res["all_dists"] = condensed

    # --- Hierarchical clustering ---
    Z = linkage(condensed, method="average")
    res["linkage"] = Z

    best_sil = -1
    best_k = 1
    for k in range(2, min(n, 6)):
        labels_k = fcluster(Z, k, criterion="maxclust")
        if len(set(labels_k)) < 2:
            continue
        sil = silhouette_score(dist_mat, labels_k, metric="precomputed")
        if sil > best_sil:
            best_sil = sil
            best_k = k

    if best_k > 1:
        hier_labels = fcluster(Z, best_k, criterion="maxclust")
        res["hier_k"] = best_k
        res["hier_silhouette"] = float(best_sil)
        res["hier_sizes"] = sorted([int((hier_labels == c).sum())
                                    for c in set(hier_labels)], reverse=True)
    else:
        res["hier_k"] = 1
        res["hier_silhouette"] = None
        res["hier_sizes"] = [n]

    # --- HDBSCAN ---
    try:
        import hdbscan
        min_cs = max(2, n // 4)
        clusterer = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=min_cs)
        hdb_labels = clusterer.fit_predict(dist_mat.astype(np.float64))
        n_cl = len(set(hdb_labels) - {-1})
        n_noise = int((hdb_labels == -1).sum())
        res["hdbscan_k"] = n_cl
        res["hdbscan_noise"] = n_noise
        if n_cl >= 2:
            mask = hdb_labels != -1
            res["hdbscan_silhouette"] = float(
                silhouette_score(dist_mat[np.ix_(mask, mask)],
                                 hdb_labels[mask], metric="precomputed"))
            res["hdbscan_sizes"] = sorted(
                [int((hdb_labels == c).sum()) for c in set(hdb_labels) - {-1}],
                reverse=True)
        else:
            res["hdbscan_silhouette"] = None
            res["hdbscan_sizes"] = [n - n_noise] if n_cl == 1 else []
    except ImportError:
        res["hdbscan_k"] = None

    # --- Bootstrap stability ---
    n_boot = 1000
    boot_means = []
    for _ in range(n_boot):
        idx = np.random.choice(n_ood, n_ood, replace=True)
        cd = pdist(pred_matrix[:, idx], metric="hamming")
        boot_means.append(cd.mean())
    res["boot_mean"] = float(np.mean(boot_means))
    res["boot_std"] = float(np.std(boot_means))
    res["boot_ci95"] = (float(np.percentile(boot_means, 2.5)),
                        float(np.percentile(boot_means, 97.5)))

    # --- Dendrogram ---
    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), 4))
    short = [f"{nm[:8]}\n({ood_accs[i]:.3f})" for i, nm in enumerate(names)]
    dendrogram(Z, labels=short, ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_title(f"Hierarchical Clustering (Hamming) — {label} OOD (n={n})")
    ax.set_ylabel("Hamming Distance")
    fig.tight_layout()
    fig.savefig(out_dir / f"dendrogram_{label}.png", dpi=150)
    plt.close(fig)

    # --- Heatmap of predictions ---
    order = leaves_list(Z)
    sorted_mat = pred_matrix[order]
    sorted_names = [short[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, max(3, n * 0.35)))
    im = ax.imshow(sorted_mat, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_names, fontsize=7)
    ax.set_xlabel("OOD Example Index")
    ax.set_title(f"Hard Predictions — {label} group (n={n}, "
                 f"mean disagree={res['mean_disagree']:.1f}/{n_ood})")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Prediction (0=False, 1=True)")
    fig.tight_layout()
    fig.savefig(out_dir / f"heatmap_{label}.png", dpi=150)
    plt.close(fig)

    # --- Distance matrix ---
    fig, ax = plt.subplots(figsize=(max(5, n * 0.4), max(4, n * 0.35)))
    ord_dist = dist_mat[np.ix_(order, order)]
    im = ax.imshow(ord_dist * n_ood, cmap="viridis", vmin=0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=6)
    ax.set_yticklabels(sorted_names, fontsize=6)
    ax.set_title(f"Pairwise Disagreements — {label} (n={n})")
    fig.colorbar(im, ax=ax, shrink=0.7, label="# examples disagreed")
    fig.tight_layout()
    fig.savefig(out_dir / f"distmat_{label}.png", dpi=150)
    plt.close(fig)

    return res


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    np.random.seed(42)

    # Load metadata for the 50 models
    df = pd.read_csv(RESULTS_CSV)
    final = df[df["datapoints_seen"] == df["datapoints_seen"].max()]
    model_files = [f.replace(".pt", "") for f in os.listdir(MODELS_DIR)]
    matched = final[final["run_id"].isin(model_files)].copy()
    matched = matched.sort_values("ood_acc").reset_index(drop=True)

    run_ids = matched["run_id"].tolist()
    ood_accs = matched["ood_acc"].values

    print(f"Computing OOD hard predictions for {len(run_ids)} models...", flush=True)
    pred_matrix = get_ood_hard_preds(run_ids)
    n_ood = pred_matrix.shape[1]
    print(f"  Shape: {pred_matrix.shape}", flush=True)

    # --- Define bins ---
    low_mask = ood_accs <= LOW_THRESH
    high_mask = ood_accs >= HIGH_THRESH

    print(f"\nBins: low (≤{LOW_THRESH}): {low_mask.sum()}, "
          f"high (≥{HIGH_THRESH}): {high_mask.sum()}, "
          f"all: {len(run_ids)}")

    # --- Per-group analysis ---
    results = {}
    for mask, label in [(low_mask, "low"), (high_mask, "high")]:
        pm = pred_matrix[mask]
        nms = [run_ids[i] for i in range(len(run_ids)) if mask[i]]
        accs = ood_accs[mask]
        print(f"\n{'='*60}")
        print(f"  {label.upper()} OOD group: {mask.sum()} models "
              f"(acc: {accs.min():.3f}–{accs.max():.3f})")
        print(f"{'='*60}")

        res = group_analysis(pm, nms, accs, label, OUT_DIR)
        results[label] = res

        if not res.get("skip"):
            print(f"  Mean Hamming dist:    {res['mean_dist']:.4f} "
                  f"({res['mean_disagree']:.1f}/{n_ood} examples disagree)")
            print(f"  Std:                  {res['std_dist']:.4f}")
            print(f"  Range:                [{res['min_dist']:.4f}, {res['max_dist']:.4f}] "
                  f"([{res['min_dist']*n_ood:.0f}, {res['max_dist']*n_ood:.0f}] examples)")
            print(f"  Bootstrap 95% CI:     [{res['boot_ci95'][0]:.4f}, "
                  f"{res['boot_ci95'][1]:.4f}]")
            print(f"  Hier clusters:        k={res['hier_k']}, sizes={res['hier_sizes']}")
            if res["hier_silhouette"] is not None:
                print(f"  Hier silhouette:      {res['hier_silhouette']:.4f}")
            if res.get("hdbscan_k") is not None:
                print(f"  HDBSCAN clusters:     {res['hdbscan_k']} "
                      f"(noise={res['hdbscan_noise']}, sizes={res.get('hdbscan_sizes')})")

    # --- Statistical comparison ---
    low_r = results.get("low", {})
    high_r = results.get("high", {})
    if not low_r.get("skip") and not high_r.get("skip"):
        d_low = low_r["all_dists"]
        d_high = high_r["all_dists"]
        stat, p = mannwhitneyu(d_low, d_high, alternative="two-sided")

        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"  Low  (n={low_r['n_models']}): mean {low_r['mean_disagree']:.1f}/{n_ood} "
              f"examples disagree per pair")
        print(f"  High (n={high_r['n_models']}): mean {high_r['mean_disagree']:.1f}/{n_ood} "
              f"examples disagree per pair")
        print(f"  Ratio: {high_r['mean_dist']/low_r['mean_dist']:.1f}x")
        print(f"  Mann-Whitney U: U={stat:.1f}, p={p:.4e}")

        # --- Comparison plot ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram of pairwise distances
        axes[0].hist(d_low * n_ood, bins=15, alpha=0.7, density=True,
                     color="tab:red", label=f"Low ≤{LOW_THRESH} (n={low_r['n_models']})")
        axes[0].hist(d_high * n_ood, bins=15, alpha=0.7, density=True,
                     color="tab:blue", label=f"High ≥{HIGH_THRESH} (n={high_r['n_models']})")
        axes[0].set_xlabel("# Examples Two Models Disagree On (out of 1000)")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Distribution of Pairwise Disagreements")
        axes[0].legend()
        axes[0].axvline(low_r["mean_disagree"], color="tab:red", ls="--", alpha=0.8)
        axes[0].axvline(high_r["mean_disagree"], color="tab:blue", ls="--", alpha=0.8)

        # Bootstrap CI
        labels_plot = [f"Low OOD\n(≤{LOW_THRESH*100:.0f}%)",
                       f"High OOD\n(≥{HIGH_THRESH*100:.0f}%)"]
        means = [low_r["boot_mean"] * n_ood, high_r["boot_mean"] * n_ood]
        cis = [(low_r["boot_ci95"][0] * n_ood, low_r["boot_ci95"][1] * n_ood),
               (high_r["boot_ci95"][0] * n_ood, high_r["boot_ci95"][1] * n_ood)]
        colors = ["tab:red", "tab:blue"]
        for i, (m, ci, c) in enumerate(zip(means, cis, colors)):
            axes[1].errorbar(i, m, yerr=[[m - ci[0]], [ci[1] - m]],
                             fmt="o", color=c, capsize=8, markersize=10)
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(labels_plot)
        axes[1].set_ylabel("Mean Pairwise Disagreements (out of 1000)")
        axes[1].set_title(f"Bootstrap 95% CI (p={p:.4e})")
        fig.suptitle("Model Variability: Low vs High OOD Accuracy", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "distance_comparison.png", dpi=150)
        plt.close(fig)

    # --- UMAP of all 50 models ---
    print("\nGenerating UMAP...", flush=True)
    try:
        import umap
        groups = np.where(low_mask, "low", np.where(high_mask, "high", "mid"))
        reducer = umap.UMAP(metric="hamming",
                            n_neighbors=min(15, len(run_ids) - 1),
                            random_state=42)
        emb = reducer.fit_transform(pred_matrix.astype(np.float32))

        fig, ax = plt.subplots(figsize=(8, 6))
        markers = {"low": "v", "mid": "o", "high": "^"}
        for g, marker in markers.items():
            gmask = groups == g
            ax.scatter(emb[gmask, 0], emb[gmask, 1], c=ood_accs[gmask],
                       cmap="coolwarm", s=80, marker=marker,
                       edgecolors="k", linewidth=0.5, vmin=0, vmax=1,
                       label=f"{g} ({gmask.sum()})")
        for i, nm in enumerate(run_ids):
            ax.annotate(nm[:6], (emb[i, 0], emb[i, 1]), fontsize=4, alpha=0.6)

        sm = plt.cm.ScalarMappable(cmap="coolwarm",
                                    norm=plt.Normalize(vmin=0, vmax=1))
        fig.colorbar(sm, ax=ax, label="OOD Accuracy")
        ax.legend(fontsize=9)
        ax.set_title("UMAP of OOD Hard Predictions (Hamming metric)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "umap_ood_predictions.png", dpi=150)
        plt.close(fig)
        print("  Saved umap_ood_predictions.png")
    except ImportError:
        print("  [warning] umap-learn not installed, skipping UMAP")

    print("\nAll outputs saved to:", OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
