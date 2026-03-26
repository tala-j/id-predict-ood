"""
Generate SGLD chain trace plots for a few representative models and heads.
Runs SGLD and records L_ood at every step (including burn-in) for visualization.
"""

import os, sys, math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data import BracketsDataset
from utils.model import get_transformer, N_CLASSES

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

INFLUENCE_DIR = os.path.join(_ROOT, "influence")
TRAIN_CSV     = os.path.join(INFLUENCE_DIR, "data/train_binomial(40,0.5).csv")
OOD_CSV       = os.path.join(INFLUENCE_DIR, "data/test_binomial(40,0.5).csv")
RESULTS_CSV   = os.path.join(INFLUENCE_DIR, "logs/1Mexp2_bin_40_05_Transformer.csv")
MODELS_DIR    = os.path.join(_HERE, "models")
OUTPUT_DIR    = os.path.join(_HERE, "cka_results")

N_LAYER, N_HEAD, N_EMBD, HEAD_DIM = 3, 4, 64, 16
N_CLASSES = 2

EPSILON   = 3e-6
GAMMA     = 5000.0
NBETA     = 100.0
NUM_STEPS = 1000
BURN_IN   = 200
SGLD_BS   = 256
OOD_SIZE  = 200


def load_model(run_id):
    ckpt = os.path.join(MODELS_DIR, f"{run_id}.pt")
    model = get_transformer(n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
                            embd_pdrop=0, resid_pdrop=0, attn_pdrop=0)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model


def get_preds(model, inputs, seq_lens):
    outputs = model(inputs)[0]
    last_idx = (seq_lens - 1).unsqueeze(1).repeat(1, N_CLASSES)
    return outputs.gather(1, last_idx.unsqueeze(1))[:, 0, :]


@torch.no_grad()
def eval_mean_loss(model, inputs, seq_lens, targets):
    preds = get_preds(model, inputs, seq_lens)
    return F.cross_entropy(preds, targets, reduction='mean').item()


def load_dataset(csv_path, n, seed):
    df = pd.read_csv(csv_path)
    if n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    bd = BracketsDataset(df)
    seq_lens = torch.tensor([len(s) + 2 for s in bd.strs])
    return bd.toks, seq_lens, bd.ylabels.long()


def build_head_mask(model, layer, head):
    hd = HEAD_DIM
    base = f"transformer.h.{layer}.attn"
    param_dict = dict(model.named_parameters())
    masks = {}

    pname = f"{base}.c_attn.weight"
    mask = torch.zeros_like(param_dict[pname], dtype=torch.bool)
    mask[head*hd:(head+1)*hd, :]                       = True
    mask[N_EMBD+head*hd:N_EMBD+(head+1)*hd, :]         = True
    mask[2*N_EMBD+head*hd:2*N_EMBD+(head+1)*hd, :]     = True
    masks[pname] = mask

    pname = f"{base}.c_attn.bias"
    mask = torch.zeros_like(param_dict[pname], dtype=torch.bool)
    mask[head*hd:(head+1)*hd]                   = True
    mask[N_EMBD+head*hd:N_EMBD+(head+1)*hd]     = True
    mask[2*N_EMBD+head*hd:2*N_EMBD+(head+1)*hd] = True
    masks[pname] = mask

    pname = f"{base}.c_proj.weight"
    mask = torch.zeros_like(param_dict[pname], dtype=torch.bool)
    mask[:, head*hd:(head+1)*hd] = True
    masks[pname] = mask

    return masks


def run_chain(run_id, layer, head, sgld_inputs, sgld_seq_lens, sgld_targets,
              ood_inputs, ood_seq_lens, ood_targets,
              id_inputs, id_seq_lens, id_targets, device):
    model = load_model(run_id).to(device)
    model.eval()
    w_star = {n: p.data.clone() for n, p in model.named_parameters()}
    Lood_star = eval_mean_loss(model, ood_inputs, ood_seq_lens, ood_targets)
    Lid_star  = eval_mean_loss(model, id_inputs,  id_seq_lens,  id_targets)

    model.train()
    head_mask = {k: v.to(device) for k, v in build_head_mask(model, layer, head).items()}
    n_train = len(sgld_inputs)
    noise_scale = math.sqrt(2.0 * EPSILON)

    lood_trace = np.zeros(NUM_STEPS, dtype=np.float32)
    lid_trace  = np.zeros(NUM_STEPS, dtype=np.float32)
    dist_trace = np.zeros(NUM_STEPS, dtype=np.float32)

    for step in range(NUM_STEPS):
        model.zero_grad()
        idx   = torch.randint(0, n_train, (SGLD_BS,), device=device)
        preds = get_preds(model, sgld_inputs[idx], sgld_seq_lens[idx])
        loss  = F.cross_entropy(preds, sgld_targets[idx], reduction='mean')
        loss.backward()

        with torch.no_grad():
            for pname, p in model.named_parameters():
                ws = w_star[pname]
                if pname in head_mask:
                    mask = head_mask[pname]
                    eff_grad = NBETA * p.grad + GAMMA * (p.data - ws)
                    noise = noise_scale * torch.randn_like(p)
                    p.data[mask] -= EPSILON * eff_grad[mask]
                    p.data[mask] += noise[mask]
                    p.data[~mask] = ws[~mask]
                else:
                    p.data.copy_(ws)

        with torch.no_grad():
            lood_trace[step] = eval_mean_loss(model, ood_inputs, ood_seq_lens, ood_targets)
            lid_trace[step]  = eval_mean_loss(model, id_inputs,  id_seq_lens,  id_targets)
            dist = sum(
                ((p.data - w_star[pname]) ** 2).sum().item()
                for pname, p in model.named_parameters()
                if pname in head_mask
            )
            dist_trace[step] = dist ** 0.5

    return lood_trace, lid_trace, dist_trace, Lood_star, Lid_star


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(RESULTS_CSV)
    final = df[(df["batch_size"] == 8) & (df["lr"] == 0.0001) &
               (df["datapoints_seen"] == df["datapoints_seen"].max())]
    ood_map = dict(zip(final["run_id"].astype(str), final["ood_acc"].astype(float)))

    # Pick 3 models: low, mid, high OOD accuracy
    run_ids = sorted(final["run_id"].tolist(), key=lambda r: ood_map[r])
    selected = {
        "low OOD":  run_ids[2],
        "mid OOD":  run_ids[len(run_ids)//2],
        "high OOD": run_ids[-1],
    }

    print("Loading data...", flush=True)
    ood_inputs, ood_seq_lens, ood_targets = load_dataset(OOD_CSV, OOD_SIZE, seed=0)
    ood_inputs   = ood_inputs.to(device)
    ood_seq_lens = ood_seq_lens.to(device)
    ood_targets  = ood_targets.to(device)

    id_inputs, id_seq_lens, id_targets = load_dataset(TRAIN_CSV, OOD_SIZE, seed=1)
    id_inputs   = id_inputs.to(device)
    id_seq_lens = id_seq_lens.to(device)
    id_targets  = id_targets.to(device)

    sgld_df = pd.read_csv(TRAIN_CSV)
    sgld_bd = BracketsDataset(sgld_df)
    sgld_inputs   = sgld_bd.toks.to(device)
    sgld_seq_lens_full = torch.tensor([len(s) + 2 for s in sgld_bd.strs]).to(device)
    sgld_targets  = sgld_bd.ylabels.long().to(device)

    steps = np.arange(NUM_STEPS)
    fig_lood, axes_lood = plt.subplots(3, N_LAYER, figsize=(12, 8), sharey=False)
    fig_lid,  axes_lid  = plt.subplots(3, N_LAYER, figsize=(12, 8), sharey=False)
    fig_dist, axes_dist = plt.subplots(3, N_LAYER, figsize=(12, 8), sharey=False)

    for row, (label, run_id) in enumerate(selected.items()):
        acc = ood_map[run_id]
        print(f"\nModel: {run_id}  ({label}, OOD acc={acc:.3f})", flush=True)
        for col, layer in enumerate(range(N_LAYER)):
            head = 0
            print(f"  layer={layer} head={head}", flush=True)
            lood_trace, lid_trace, dist_trace, Lood_star, Lid_star = run_chain(
                run_id, layer, head,
                sgld_inputs, sgld_seq_lens_full, sgld_targets,
                ood_inputs, ood_seq_lens, ood_targets,
                id_inputs, id_seq_lens, id_targets, device)

            # L_ood plot
            ax = axes_lood[row, col]
            ax.plot(steps, lood_trace, lw=0.6, color="steelblue", alpha=0.8)
            ax.axhline(Lood_star, color="red", lw=1.0, linestyle="--", label="w* (MAP)")
            ax.axvline(BURN_IN, color="gray", lw=1.0, linestyle=":", label="burn-in")
            ax.set_title(f"{label} (acc={acc:.2f})\nLayer {layer}, Head {head}", fontsize=8)
            ax.set_xlabel("SGLD step", fontsize=7)
            ax.set_ylabel("L_ood(w)", fontsize=7)
            ax.tick_params(labelsize=6)
            if row == 0 and col == 0:
                ax.legend(fontsize=6)

            # L_id plot
            ax = axes_lid[row, col]
            ax.plot(steps, lid_trace, lw=0.6, color="seagreen", alpha=0.8)
            ax.axhline(Lid_star, color="red", lw=1.0, linestyle="--", label="w* (MAP)")
            ax.axvline(BURN_IN, color="gray", lw=1.0, linestyle=":", label="burn-in")
            ax.set_title(f"{label} (acc={acc:.2f})\nLayer {layer}, Head {head}", fontsize=8)
            ax.set_xlabel("SGLD step", fontsize=7)
            ax.set_ylabel("L_id(w)", fontsize=7)
            ax.tick_params(labelsize=6)
            if row == 0 and col == 0:
                ax.legend(fontsize=6)

            # Distance plot
            ax = axes_dist[row, col]
            ax.plot(steps, dist_trace, lw=0.6, color="darkorange", alpha=0.8)
            ax.axvline(BURN_IN, color="gray", lw=1.0, linestyle=":", label="burn-in")
            ax.set_title(f"{label} (acc={acc:.2f})\nLayer {layer}, Head {head}", fontsize=8)
            ax.set_xlabel("SGLD step", fontsize=7)
            ax.set_ylabel("||w − w*||", fontsize=7)
            ax.tick_params(labelsize=6)
            if row == 0 and col == 0:
                ax.legend(fontsize=6)

    fig_lid.suptitle("SGLD chain traces — L_id(w) over steps\n"
                     f"ε={EPSILON}, γ={GAMMA}, nβ={NBETA}, burn-in={BURN_IN}", fontsize=10)
    fig_lid.tight_layout()
    out_lid = os.path.join(OUTPUT_DIR, "chain_traces_lid.png")
    fig_lid.savefig(out_lid, dpi=150)
    print(f"\nSaved {out_lid}")

    fig_lood.suptitle("SGLD chain traces — L_ood(w) over steps\n"
                      f"ε={EPSILON}, γ={GAMMA}, nβ={NBETA}, burn-in={BURN_IN}", fontsize=10)
    fig_lood.tight_layout()
    out_lood = os.path.join(OUTPUT_DIR, "chain_traces_lood.png")
    fig_lood.savefig(out_lood, dpi=150)
    print(f"\nSaved {out_lood}")

    fig_dist.suptitle("SGLD chain traces — ||w − w*|| over steps\n"
                      f"ε={EPSILON}, γ={GAMMA}, nβ={NBETA}, burn-in={BURN_IN}", fontsize=10)
    fig_dist.tight_layout()
    out_dist = os.path.join(OUTPUT_DIR, "chain_traces_dist.png")
    fig_dist.savefig(out_dist, dpi=150)
    print(f"Saved {out_dist}")


if __name__ == "__main__":
    main()
