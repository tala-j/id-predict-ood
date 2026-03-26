"""
Attention-head susceptibility (Baker et al. Definitions 2.4 & 2.5).

For each model checkpoint and each attention head C = (layer l, head h):
  1. Build a boolean parameter mask for C's weights (Q/K/V rows of c_attn,
     output columns of c_proj).
  2. Run weight-restricted SGLD: only C's parameters move; all others are
     pinned to w* after every gradient step.
  3. At each post-burn-in draw record:
       L_ood(w)    = mean OOD loss
       l_j(w)      = per-sample loss on N_OBS training probe samples
       L_n(w)      = mean training loss
  4. chi_{C,j} = -Cov_beta[ L_ood(w),  l_j(w) - L_n(w) ]

  Instantiates Definition 2.5 with:
    phi_C(w) = L_ood(w)      -- OOD generalization loss under perturbed head C
    (x,y)    = training probe sample j
    L_n(w)   = mean training loss (normalisation)

Saves one [n_layer, n_head, n_obs] array per model in results/.

Usage:
  python susceptibility.py --worker_id 0 --num_workers 8
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
TRAIN_CSV     = os.path.join(INFLUENCE_DIR, "data/train_binomial(40,0.5).csv")
OOD_CSV       = os.path.join(INFLUENCE_DIR, "data/test_binomial(40,0.5).csv")
RESULTS_CSV   = os.path.join(INFLUENCE_DIR, "logs/1Mexp2_bin_40_05_Transformer.csv")
MODELS_DIR    = os.path.join(_HERE, "models")
OUTPUT_DIR    = os.path.join(_HERE, "results")

# -------------------------
# Config
# -------------------------
TARGET_BS  = 8
TARGET_LR  = 0.0001

N_LAYER    = 3
N_HEAD     = 4
N_EMBD     = 64
HEAD_DIM   = N_EMBD // N_HEAD  # 16

N_OBS         = 200   # training probe samples for l_j
OOD_EVAL_SIZE = 200   # OOD samples for L_ood
LN_EVAL_SIZE  = 500   # training samples for L_n

# SGLD defaults
DEFAULT_EPSILON   = 3e-6
DEFAULT_GAMMA     = 5000.0
DEFAULT_NBETA     = 100.0
DEFAULT_NUM_STEPS = 1000
DEFAULT_BURN_IN   = 200
DEFAULT_SGLD_BS   = 256


# -------------------------
# Helpers
# -------------------------
def get_run_ids():
    df = pd.read_csv(RESULTS_CSV)
    final = df[
        (df["batch_size"] == TARGET_BS) &
        (df["lr"] == TARGET_LR) &
        (df["datapoints_seen"] == df["datapoints_seen"].max())
    ]
    return sorted(final["run_id"].tolist())


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
def eval_mean_loss(model, inputs, seq_lens, targets, batch_size=512):
    n = len(inputs)
    total = 0.0
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        preds = get_preds(model, inputs[i:j], seq_lens[i:j])
        total += F.cross_entropy(preds, targets[i:j], reduction='sum').item()
    return total / n


@torch.no_grad()
def eval_per_sample_losses(model, inputs, seq_lens, targets, batch_size=512):
    n = len(inputs)
    losses = torch.zeros(n, device=inputs.device)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        preds = get_preds(model, inputs[i:j], seq_lens[i:j])
        losses[i:j] = F.cross_entropy(preds, targets[i:j], reduction='none')
    return losses.cpu().numpy()


def load_dataset(csv_path, n, seed):
    df = pd.read_csv(csv_path)
    if n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    bd = BracketsDataset(df)
    seq_lens = torch.tensor([len(s) + 2 for s in bd.strs])
    return bd.toks, seq_lens, bd.ylabels.long()


# -------------------------
# Head parameter masks
# -------------------------
def build_head_mask(model, layer, head):
    h, hd = head, HEAD_DIM
    base = f"transformer.h.{layer}.attn"
    param_dict = dict(model.named_parameters())
    masks = {}

    pname = f"{base}.c_attn.weight"
    mask = torch.zeros_like(param_dict[pname], dtype=torch.bool)
    mask[h*hd:(h+1)*hd,                   :] = True
    mask[N_EMBD+h*hd:N_EMBD+(h+1)*hd,     :] = True
    mask[2*N_EMBD+h*hd:2*N_EMBD+(h+1)*hd, :] = True
    masks[pname] = mask

    pname = f"{base}.c_attn.bias"
    mask = torch.zeros_like(param_dict[pname], dtype=torch.bool)
    mask[h*hd:(h+1)*hd]                   = True
    mask[N_EMBD+h*hd:N_EMBD+(h+1)*hd]     = True
    mask[2*N_EMBD+h*hd:2*N_EMBD+(h+1)*hd] = True
    masks[pname] = mask

    pname = f"{base}.c_proj.weight"
    mask = torch.zeros_like(param_dict[pname], dtype=torch.bool)
    mask[:, h*hd:(h+1)*hd] = True
    masks[pname] = mask

    return masks


# -------------------------
# Weight-restricted SGLD
# -------------------------
def run_sgld_restricted(
        run_id, layer, head,
        sgld_inputs,  sgld_seq_lens,  sgld_targets,   # gradient steps
        ood_inputs,   ood_seq_lens,   ood_targets,    # L_ood
        ln_inputs,    ln_seq_lens,    ln_targets,     # L_n
        obs_inputs,   obs_seq_lens,   obs_targets,    # training probe l_j
        epsilon, gamma, nbeta, num_steps, burn_in, sgld_batch_size):
    """
    Returns:
      Lood_traces: [num_draws]          L_ood(w) at each draw
      obs_traces:  [num_draws, n_obs]   per-training-probe losses at each draw
      Ln_traces:   [num_draws]          L_n(w) at each draw
    """
    device = torch.device("cuda")
    model = load_model(run_id).to(device)
    model.eval()

    w_star = {n: p.data.clone() for n, p in model.named_parameters()}

    model.train()
    head_mask = {k: v.to(device) for k, v in build_head_mask(model, layer, head).items()}

    n_train   = len(sgld_inputs)
    n_obs     = len(obs_inputs)
    num_draws = num_steps - burn_in

    Lood_traces = np.zeros(num_draws,          dtype=np.float32)
    obs_traces  = np.zeros((num_draws, n_obs),  dtype=np.float32)
    Ln_traces   = np.zeros(num_draws,           dtype=np.float32)

    noise_scale = math.sqrt(2.0 * epsilon)
    draw_idx = 0

    for step in range(num_steps):
        # --- SGLD gradient step ---
        model.zero_grad()
        idx   = torch.randint(0, n_train, (sgld_batch_size,), device=device)
        preds = get_preds(model, sgld_inputs[idx], sgld_seq_lens[idx])
        loss  = F.cross_entropy(preds, sgld_targets[idx], reduction='mean')
        loss.backward()

        # --- weight-restricted update ---
        with torch.no_grad():
            for pname, p in model.named_parameters():
                ws = w_star[pname]
                if pname in head_mask:
                    mask     = head_mask[pname]
                    eff_grad = nbeta * p.grad + gamma * (p.data - ws)
                    noise    = noise_scale * torch.randn_like(p)
                    p.data[mask]  -= epsilon * eff_grad[mask]
                    p.data[mask]  += noise[mask]
                    p.data[~mask]  = ws[~mask]
                else:
                    p.data.copy_(ws)

        # --- record traces after burn-in ---
        if step >= burn_in:
            L_ood = eval_mean_loss(model, ood_inputs, ood_seq_lens, ood_targets)
            L_n   = eval_mean_loss(model, ln_inputs,  ln_seq_lens,  ln_targets)
            obs   = eval_per_sample_losses(model, obs_inputs, obs_seq_lens, obs_targets)

            Lood_traces[draw_idx] = L_ood
            Ln_traces[draw_idx]   = L_n
            obs_traces[draw_idx]  = obs
            draw_idx += 1

            if draw_idx % 100 == 0:
                print(f"      draw {draw_idx}/{num_draws}  "
                      f"L_ood={L_ood:.4f}  L_n={L_n:.6f}  "
                      f"Lood_std={Lood_traces[:draw_idx].std():.4f}",
                      flush=True)

    return Lood_traces, obs_traces, Ln_traces


# -------------------------
# Susceptibility (Variant 2)
# -------------------------
def compute_chi(Lood_traces, obs_traces, Ln_traces):
    """
    chi_{C,j} = -Cov[ L_ood(w), l_j(w) - L_n(w) ]

    Lood_traces: [num_draws]
    obs_traces:  [num_draws, n_obs]
    Ln_traces:   [num_draws]

    Returns chi: [n_obs]
    """
    phi     = Lood_traces - Lood_traces.mean()                    # [D]
    resid   = obs_traces - Ln_traces[:, None]                     # [D, n_obs]
    resid_c = resid - resid.mean(axis=0, keepdims=True)
    return -(phi[:, None] * resid_c).mean(axis=0)                 # [n_obs]


# -------------------------
# Output path
# -------------------------
def result_path(run_id):
    return os.path.join(OUTPUT_DIR, f"{run_id}_cpt.npy")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id",      type=int,   default=0)
    parser.add_argument("--num_workers",    type=int,   default=1)
    parser.add_argument("--epsilon",        type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--gamma",          type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--nbeta",          type=float, default=DEFAULT_NBETA)
    parser.add_argument("--num_steps",      type=int,   default=DEFAULT_NUM_STEPS)
    parser.add_argument("--burn_in",        type=int,   default=DEFAULT_BURN_IN)
    parser.add_argument("--sgld_bs",        type=int,   default=DEFAULT_SGLD_BS)
    parser.add_argument("--n_obs",          type=int,   default=N_OBS)
    parser.add_argument("--ood_eval_size",  type=int,   default=OOD_EVAL_SIZE)
    parser.add_argument("--ln_eval_size",   type=int,   default=LN_EVAL_SIZE)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda")

    run_ids = get_run_ids()
    my_runs = [r for i, r in enumerate(run_ids) if i % args.num_workers == args.worker_id]
    pending = [r for r in my_runs if not os.path.exists(result_path(r))]

    print(f"Worker {args.worker_id}/{args.num_workers}: "
          f"{len(pending)}/{len(my_runs)} models pending", flush=True)
    if not pending:
        print("Nothing to do.")
        return

    print(f"Loading OOD eval set ({args.ood_eval_size} samples)...", flush=True)
    ood_inputs, ood_seq_lens, ood_targets = load_dataset(OOD_CSV, args.ood_eval_size, seed=0)
    ood_inputs, ood_seq_lens, ood_targets = (ood_inputs.to(device), ood_seq_lens.to(device), ood_targets.to(device))

    print(f"Loading L_n eval set ({args.ln_eval_size} samples)...", flush=True)
    ln_inputs, ln_seq_lens, ln_targets = load_dataset(TRAIN_CSV, args.ln_eval_size, seed=99)
    ln_inputs, ln_seq_lens, ln_targets = (ln_inputs.to(device), ln_seq_lens.to(device), ln_targets.to(device))

    print(f"Loading training probe set ({args.n_obs} samples)...", flush=True)
    obs_inputs, obs_seq_lens, obs_targets = load_dataset(TRAIN_CSV, args.n_obs, seed=42)
    obs_inputs, obs_seq_lens, obs_targets = (obs_inputs.to(device), obs_seq_lens.to(device), obs_targets.to(device))

    print("Loading full train set for SGLD gradient steps...", flush=True)
    sgld_df  = pd.read_csv(TRAIN_CSV)
    sgld_bd  = BracketsDataset(sgld_df)
    sgld_inputs   = sgld_bd.toks.to(device)
    sgld_seq_lens = torch.tensor([len(s) + 2 for s in sgld_bd.strs]).to(device)
    sgld_targets  = sgld_bd.ylabels.long().to(device)

    print(f"SGLD: eps={args.epsilon}, gamma={args.gamma}, nbeta={args.nbeta}, "
          f"steps={args.num_steps}, burn_in={args.burn_in}, sgld_bs={args.sgld_bs}",
          flush=True)

    for mi, run_id in enumerate(pending):
        print(f"\n[{mi+1}/{len(pending)}] model {run_id}", flush=True)
        chi_tensor = np.zeros((N_LAYER, N_HEAD, args.n_obs), dtype=np.float32)

        for layer in range(N_LAYER):
            for head in range(N_HEAD):
                print(f"  layer={layer} head={head}", flush=True)
                Lood_tr, obs_tr, Ln_tr = run_sgld_restricted(
                    run_id, layer, head,
                    sgld_inputs, sgld_seq_lens, sgld_targets,
                    ood_inputs,  ood_seq_lens,  ood_targets,
                    ln_inputs,   ln_seq_lens,   ln_targets,
                    obs_inputs,  obs_seq_lens,  obs_targets,
                    args.epsilon, args.gamma, args.nbeta,
                    args.num_steps, args.burn_in, args.sgld_bs)

                chi = compute_chi(Lood_tr, obs_tr, Ln_tr)   # [n_obs]
                chi_tensor[layer, head] = chi
                print(f"    chi[{layer},{head}]: mean={chi.mean():.2e}  std={chi.std():.2e}  "
                      f"Lood_std={Lood_tr.std():.4f}  Ln_std={Ln_tr.std():.6f}",
                      flush=True)

        out = result_path(run_id)
        np.save(out, chi_tensor)
        print(f"  Saved {out}  shape={chi_tensor.shape}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
