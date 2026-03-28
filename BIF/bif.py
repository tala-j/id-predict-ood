"""
Local Bayesian Influence Functions (BIF) — Hessian-free data attribution.

Theory
------
For train sample z_i and query sample z_j:

    BIF_γ(z_i, z_j) = -Cov_γ(ℓ_i(w), ℓ_j(w))

where the covariance is under the localized posterior:

    p_γ(w | D, w*) ∝ exp(-∑_k ℓ_k(w) - γ/2 ||w - w*||²)

Estimated via SGLD chains starting from the trained checkpoint w*.

SGLD update (Welling & Teh, 2011 / localized posterior):

    w_{t+1} = w_t - ε/2 · [nβ/m · ∑_{k∈B_t} ∇ℓ_k(w_t) + γ(w_t − w*)]
                   + N(0, ε·I)

    n    : training set size (used for attribution background)
    m    : SGLD minibatch size
    β    : inverse temperature
    γ    : localization strength
    ε    : step size
    nβ   : stored as a single pre-multiplied scalar `nbeta`

Design principles
-----------------
- Model-agnostic: works for any differentiable nn.Module.
- Two batching modes are strictly separated:
    * SGLD gradient minibatch (size m) — used only for the SGLD update.
    * Eval forward-pass batch  — used only for recording loss traces.
- Multiple independent chains, each starting from w*.
- Per-token attribution is first-class for autoregressive LMs.

Usage
-----
    config = BIFConfig(epsilon=3e-6, gamma=500.0, nbeta=100.0,
                       n_chains=1, n_steps=2000, burn_in=500)

    def loss_fn(model, batch):
        inputs, targets = batch
        logits = model(inputs)
        return F.cross_entropy(logits, targets, reduction='none')  # [n]

    results = compute_bif(model, train_dataset, query_dataset, loss_fn, config)
    # results.bif_matrix  : Tensor[n_query, n_train]  (raw -Cov)
    # results.corr_matrix : Tensor[n_query, n_train]  (normalized by std)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BIFConfig:
    """Hyperparameters for the BIF estimator."""

    # --- SGLD hyperparameters ---
    epsilon: float = 3e-6
    """Step size ε for SGLD."""

    gamma: float = 500.0
    """Localization strength γ: controls how tightly the posterior is anchored to w*."""

    nbeta: float = 100.0
    """Pre-multiplied inverse temperature nβ (= n_train · β).
    Scales the data likelihood gradient relative to the localization prior."""

    # --- Chain settings ---
    n_chains: int = 1
    """Number of independent SGLD chains. Each starts from w*."""

    n_steps: int = 2000
    """Total SGLD steps per chain (including burn-in)."""

    burn_in: int = 500
    """Steps discarded at the start of each chain before recording draws."""

    # --- Batching (kept strictly separate) ---
    sgld_batch_size: int = 256
    """Size m of the SGLD gradient minibatch. Independent of eval batching."""

    eval_batch_size: int = 512
    """Batch size for forward-pass loss evaluation over train/query sets."""

    # --- Parameter subset ---
    param_subset: str = "all"
    """Which parameters SGLD updates.
    Options:
      'all'          — all trainable parameters (default)
      'attention'    — attention projection weights/biases only
      'attention_mlp'— attention + MLP / feed-forward layers
    Or pass a callable: (param_name: str) -> bool
    """

    # --- Misc ---
    device: str = "cuda"
    seed: Optional[int] = None
    collect_diagnostics: bool = False
    """If True, also return per-draw mean and variance of each loss trace."""

    # --- Per-token mode ---
    per_token: bool = False
    """If True, also compute token-level covariance using token_loss_fn."""


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class ModelWrapper:
    """
    Wraps a trained PyTorch model, freezes a reference checkpoint w*, and
    exposes the active (SGLD-updated) parameter subset.

    Parameters
    ----------
    model       : trained nn.Module (will be moved to `device`).
    param_subset: see BIFConfig.param_subset.
    device      : 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model: nn.Module,
        param_subset: str | Callable[[str], bool] = "all",
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self._active: List[nn.Parameter] = self._select_params(param_subset)
        if not self._active:
            raise ValueError(
                f"param_subset={param_subset!r} selected zero parameters. "
                "Check your model's named_parameters."
            )
        # Freeze w* on CPU to save GPU memory
        self._w_star: List[torch.Tensor] = [p.data.clone().cpu() for p in self._active]

    # ------------------------------------------------------------------
    # Param selection
    # ------------------------------------------------------------------

    def _select_params(
        self, subset: str | Callable[[str], bool]
    ) -> List[nn.Parameter]:
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        if callable(subset):
            return [p for n, p in named if subset(n)]
        if subset == "all":
            return [p for _, p in named]
        if subset == "attention":
            return [p for n, p in named if _is_attn_param(n)]
        if subset == "attention_mlp":
            return [p for n, p in named if _is_attn_param(n) or _is_mlp_param(n)]
        raise ValueError(
            f"Unknown param_subset={subset!r}. "
            "Expected 'all', 'attention', 'attention_mlp', or a callable."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def active_params(self) -> List[nn.Parameter]:
        """Parameters that SGLD will update."""
        return self._active

    def n_active_params(self) -> int:
        return sum(p.numel() for p in self._active)

    def reset_to_wstar(self) -> None:
        """Copy w* back into the active parameters (in-place)."""
        for p, w in zip(self._active, self._w_star):
            p.data.copy_(w.to(self.device))


# ---------------------------------------------------------------------------
# Parameter name helpers
# ---------------------------------------------------------------------------

_ATTN_KEYWORDS = (
    "attn", "attention", "self_attn", "cross_attn",
    "q_proj", "k_proj", "v_proj", "out_proj",
    "query", "key", "value",
)
_MLP_KEYWORDS = (
    "mlp", "ffn", "feed_forward", "fc1", "fc2",
    "linear1", "linear2", "intermediate", "dense", "c_fc", "c_proj",
)


def _is_attn_param(name: str) -> bool:
    low = name.lower()
    return any(kw in low for kw in _ATTN_KEYWORDS)


def _is_mlp_param(name: str) -> bool:
    low = name.lower()
    return any(kw in low for kw in _MLP_KEYWORDS)


# ---------------------------------------------------------------------------
# SGLD sampler
# ---------------------------------------------------------------------------

def _sgld_step(
    wrapper: ModelWrapper,
    batch: object,
    loss_fn: Callable,
    w_star_gpu: List[torch.Tensor],
    epsilon: float,
    gamma: float,
    nbeta: float,
    noise_std: float,
) -> None:
    """
    One SGLD update:
        w_{t+1} = w_t - ε/2 · [nbeta · ∇L̄(w_t) + γ(w_t − w*)] + N(0, ε·I)

    where L̄ = (1/m) ∑_{k∈B} ℓ_k (mean over SGLD minibatch).
    """
    wrapper.model.zero_grad()
    batch = _to_device(batch, wrapper.device)
    losses = loss_fn(wrapper.model, batch)      # [m]
    losses.mean().backward()

    half_eps = epsilon * 0.5
    with torch.no_grad():
        for p, ws in zip(wrapper.active_params, w_star_gpu):
            drift = nbeta * p.grad + gamma * (p.data - ws)
            p.data.add_(drift, alpha=-half_eps)
            p.data.add_(torch.randn_like(p), alpha=noise_std)


def run_chain(
    wrapper: ModelWrapper,
    sgld_loader: DataLoader,
    train_eval_loader: DataLoader,
    query_loader: DataLoader,
    loss_fn: Callable,
    config: BIFConfig,
    token_loss_fn: Optional[Callable] = None,
) -> Dict[str, torch.Tensor]:
    """
    Run one SGLD chain from w* and record per-example losses at each post-burn-in step.

    Returns
    -------
    dict with keys:
        'train' : Tensor[n_train, n_draws]
        'query' : Tensor[n_query, n_draws]
        'train_token' (optional): Tensor[n_train_tokens, n_draws]
        'query_token' (optional): Tensor[n_query_tokens, n_draws]
    """
    device = wrapper.device
    n_draws = config.n_steps - config.burn_in
    if n_draws <= 0:
        raise ValueError("n_steps must be greater than burn_in.")

    epsilon = config.epsilon
    gamma = config.gamma
    nbeta = config.nbeta
    noise_std = math.sqrt(epsilon)

    # Reset to w* at chain start
    wrapper.reset_to_wstar()

    # Pre-pin w* to GPU for the localization gradient
    w_star_gpu = [w.to(device) for w in wrapper._w_star]

    # Pre-allocate trace tensors (stored on CPU to save GPU memory)
    n_train = len(train_eval_loader.dataset)
    n_query = len(query_loader.dataset)
    train_trace = torch.zeros(n_train, n_draws)
    query_trace = torch.zeros(n_query, n_draws)

    # Optional token-level traces; sizes determined lazily on first draw
    train_tok_trace: Optional[torch.Tensor] = None
    query_tok_trace: Optional[torch.Tensor] = None

    sgld_iter = _infinite(sgld_loader)
    draw_idx = 0

    for step in range(config.n_steps):
        batch = next(sgld_iter)
        _sgld_step(wrapper, batch, loss_fn, w_star_gpu,
                   epsilon, gamma, nbeta, noise_std)

        if step < config.burn_in:
            continue

        # Record loss traces (no gradient needed)
        with torch.no_grad():
            t_losses = _eval_losses(wrapper.model, train_eval_loader, loss_fn, device)
            q_losses = _eval_losses(wrapper.model, query_loader, loss_fn, device)

        train_trace[:, draw_idx] = t_losses.cpu()
        query_trace[:, draw_idx] = q_losses.cpu()

        # Per-token traces (optional)
        if token_loss_fn is not None and config.per_token:
            with torch.no_grad():
                t_tok = _eval_token_losses(
                    wrapper.model, train_eval_loader, token_loss_fn, device
                )
                q_tok = _eval_token_losses(
                    wrapper.model, query_loader, token_loss_fn, device
                )

            if train_tok_trace is None:
                train_tok_trace = torch.zeros(t_tok.shape[0], n_draws)
                query_tok_trace = torch.zeros(q_tok.shape[0], n_draws)

            train_tok_trace[:, draw_idx] = t_tok.cpu()
            query_tok_trace[:, draw_idx] = q_tok.cpu()

        draw_idx += 1

    result = {"train": train_trace, "query": query_trace}
    if train_tok_trace is not None:
        result["train_token"] = train_tok_trace
        result["query_token"] = query_tok_trace

    return result


# ---------------------------------------------------------------------------
# Loss evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_losses(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    device: torch.device,
) -> torch.Tensor:
    """
    Evaluate per-example losses over the entire loader.

    Returns: Tensor[n] on CPU.
    """
    parts: List[torch.Tensor] = []
    for batch in loader:
        batch = _to_device(batch, device)
        losses = loss_fn(model, batch)          # [batch_size]
        parts.append(losses.detach().cpu())
    return torch.cat(parts, dim=0)


@torch.no_grad()
def _eval_token_losses(
    model: nn.Module,
    loader: DataLoader,
    token_loss_fn: Callable,
    device: torch.device,
) -> torch.Tensor:
    """
    Evaluate per-token losses and flatten to [n_tokens_total].

    token_loss_fn(model, batch) -> Tensor[batch_size, seq_len] (per-token losses,
    with padding positions already zeroed or masked).
    """
    parts: List[torch.Tensor] = []
    for batch in loader:
        batch = _to_device(batch, device)
        tok_losses = token_loss_fn(model, batch)    # [B, T]
        # Flatten: each row is a sequence; we concatenate all tokens
        parts.append(tok_losses.detach().cpu().reshape(-1))
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------

def compute_covariance(
    train_trace: torch.Tensor,
    query_trace: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute BIF matrices from loss traces.

    Parameters
    ----------
    train_trace : Tensor[n_train, T]  — per-train-example losses across T draws.
    query_trace : Tensor[n_query, T]  — per-query-example losses across T draws.

    Returns
    -------
    bif_matrix  : Tensor[n_query, n_train]
        -Cov[ℓ_query_i, ℓ_train_j]  (raw covariance, sign-flipped)
    corr_matrix : Tensor[n_query, n_train]
        Normalized version: -Corr[ℓ_query_i, ℓ_train_j]
    """
    T = train_trace.shape[1]
    if query_trace.shape[1] != T:
        raise ValueError(
            f"train_trace and query_trace must have the same number of draws, "
            f"got {T} vs {query_trace.shape[1]}."
        )

    # Work in float64 for numerical stability
    L = train_trace.double()   # [n_train, T]
    Q = query_trace.double()   # [n_query, T]

    # Center
    L_c = L - L.mean(dim=1, keepdim=True)   # [n_train, T]
    Q_c = Q - Q.mean(dim=1, keepdim=True)   # [n_query, T]

    # Sample covariance: -1/(T-1) * Q_c @ L_c.T → [n_query, n_train]
    bif = -(Q_c @ L_c.t()) / (T - 1)

    # Normalized BIF (posterior correlation)
    std_L = L_c.std(dim=1)   # [n_train]
    std_Q = Q_c.std(dim=1)   # [n_query]

    # Avoid division by zero for constant traces
    eps_std = torch.finfo(torch.float64).eps
    std_L = std_L.clamp(min=eps_std)
    std_Q = std_Q.clamp(min=eps_std)

    corr = bif / (std_Q.unsqueeze(1) * std_L.unsqueeze(0))

    return bif.float(), corr.float()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class BIFResults:
    """Output of compute_bif()."""

    bif_matrix: torch.Tensor
    """Shape [n_query, n_train]. Raw -Cov(ℓ_query, ℓ_train)."""

    corr_matrix: torch.Tensor
    """Shape [n_query, n_train]. Normalized posterior correlation."""

    metadata: Dict = field(default_factory=dict)
    """Config, chain counts, draw counts, etc."""

    train_trace: Optional[torch.Tensor] = None
    """Shape [n_train, n_total_draws]. Full loss trace (if retained)."""

    query_trace: Optional[torch.Tensor] = None
    """Shape [n_query, n_total_draws]. Full loss trace (if retained)."""

    train_token_trace: Optional[torch.Tensor] = None
    """Shape [n_train_tokens, n_total_draws]. Per-token train trace (per_token mode)."""

    query_token_trace: Optional[torch.Tensor] = None
    """Shape [n_query_tokens, n_total_draws]. Per-token query trace (per_token mode)."""

    bif_token_matrix: Optional[torch.Tensor] = None
    """Per-token BIF [n_query_tokens, n_train_tokens] (per_token mode)."""

    corr_token_matrix: Optional[torch.Tensor] = None
    """Per-token normalized BIF (per_token mode)."""

    diagnostics: Optional[Dict] = None
    """Mean and variance traces per example, if collect_diagnostics=True."""


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def compute_bif(
    model: nn.Module,
    train_dataset: Dataset,
    query_dataset: Dataset,
    loss_fn: Callable[[nn.Module, object], torch.Tensor],
    config: BIFConfig,
    token_loss_fn: Optional[Callable[[nn.Module, object], torch.Tensor]] = None,
    retain_traces: bool = False,
    train_eval_dataset: Optional[Dataset] = None,
) -> BIFResults:
    """
    Compute Bayesian Influence Function scores between all train/query pairs.

    Parameters
    ----------
    model              : trained nn.Module at checkpoint w*.
    train_dataset      : dataset used for SGLD gradient minibatches (typically the
                         full training set, e.g. 100K examples).
    query_dataset      : query examples to attribute.
    loss_fn            : ``(model, batch) -> Tensor[n]``
                         Returns per-example scalar losses for a batch.
                         Must be differentiable w.r.t. model parameters (SGLD update).
    config             : BIFConfig with all hyperparameters.
    token_loss_fn      : (optional) ``(model, batch) -> Tensor[n, T]``
                         Per-token losses for autoregressive LMs. Required if
                         config.per_token=True.
    retain_traces      : if True, store the full loss trace matrices in BIFResults.
    train_eval_dataset : (optional) subset of train examples over which loss traces
                         are collected. Defaults to train_dataset if not provided.
                         Use this when SGLD should run on the full training set but
                         trace collection should cover only a smaller attribution subset.

    Returns
    -------
    BIFResults
        .bif_matrix  [n_query, n_train_eval]  — raw -Cov scores
        .corr_matrix [n_query, n_train_eval]  — normalized posterior correlation
        (plus optional token-level matrices and diagnostics)
    """
    if config.per_token and token_loss_fn is None:
        raise ValueError("token_loss_fn must be provided when config.per_token=True.")

    if config.seed is not None:
        torch.manual_seed(config.seed)

    wrapper = ModelWrapper(model, config.param_subset, config.device)

    _train_eval_ds = train_eval_dataset if train_eval_dataset is not None else train_dataset

    # Eval DataLoaders — independent of SGLD minibatch size
    train_eval_loader = DataLoader(
        _train_eval_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        drop_last=False,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        drop_last=False,
    )
    # SGLD gradient minibatch loader — uses the full train_dataset
    sgld_loader = DataLoader(
        train_dataset,
        batch_size=config.sgld_batch_size,
        shuffle=True,
        drop_last=True,
    )

    n_draws_per_chain = config.n_steps - config.burn_in
    n_total_draws = config.n_chains * n_draws_per_chain

    # Accumulate traces across chains: [n_examples, n_total_draws]
    all_train_traces: List[torch.Tensor] = []
    all_query_traces: List[torch.Tensor] = []
    all_train_tok_traces: List[torch.Tensor] = []
    all_query_tok_traces: List[torch.Tensor] = []

    for chain_idx in range(config.n_chains):
        if config.seed is not None:
            torch.manual_seed(config.seed + chain_idx * 1000)

        print(
            f"Chain {chain_idx + 1}/{config.n_chains}  "
            f"({n_draws_per_chain} draws after {config.burn_in} burn-in steps) ...",
            flush=True,
        )

        chain_result = run_chain(
            wrapper=wrapper,
            sgld_loader=sgld_loader,
            train_eval_loader=train_eval_loader,
            query_loader=query_loader,
            loss_fn=loss_fn,
            config=config,
            token_loss_fn=token_loss_fn,
        )

        all_train_traces.append(chain_result["train"])
        all_query_traces.append(chain_result["query"])

        if config.per_token and "train_token" in chain_result:
            all_train_tok_traces.append(chain_result["train_token"])
            all_query_tok_traces.append(chain_result["query_token"])

    # Concatenate across chains: [n_examples, n_total_draws]
    train_trace = torch.cat(all_train_traces, dim=1)
    query_trace = torch.cat(all_query_traces, dim=1)

    print(
        f"Computing covariance over {n_total_draws} draws  "
        f"({len(_train_eval_ds)} train_eval × {len(query_dataset)} query) ...",
        flush=True,
    )

    bif_matrix, corr_matrix = compute_covariance(train_trace, query_trace)

    result = BIFResults(
        bif_matrix=bif_matrix,
        corr_matrix=corr_matrix,
        metadata={
            "n_train_sgld": len(train_dataset),
            "n_train_eval": len(_train_eval_ds),
            "n_query": len(query_dataset),
            "n_chains": config.n_chains,
            "n_steps": config.n_steps,
            "burn_in": config.burn_in,
            "n_total_draws": n_total_draws,
            "epsilon": config.epsilon,
            "gamma": config.gamma,
            "nbeta": config.nbeta,
            "param_subset": config.param_subset,
            "n_active_params": wrapper.n_active_params(),
        },
    )

    if retain_traces:
        result.train_trace = train_trace
        result.query_trace = query_trace

    # Token-level BIF
    if all_train_tok_traces:
        train_tok_trace = torch.cat(all_train_tok_traces, dim=1)
        query_tok_trace = torch.cat(all_query_tok_traces, dim=1)

        bif_tok, corr_tok = compute_covariance(train_tok_trace, query_tok_trace)

        result.bif_token_matrix = bif_tok
        result.corr_token_matrix = corr_tok

        if retain_traces:
            result.train_token_trace = train_tok_trace
            result.query_token_trace = query_tok_trace

    # Diagnostics
    if config.collect_diagnostics:
        result.diagnostics = {
            "train_loss_mean": train_trace.mean(dim=1),   # [n_train]
            "train_loss_var":  train_trace.var(dim=1),    # [n_train]
            "query_loss_mean": query_trace.mean(dim=1),   # [n_query]
            "query_loss_var":  query_trace.var(dim=1),    # [n_query]
        }

    return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_device(batch: object, device: torch.device) -> object:
    """Recursively move tensors in batch to device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        moved = [_to_device(x, device) for x in batch]
        return type(batch)(moved)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch  # non-tensor (e.g. string labels): pass through


def _infinite(loader: DataLoader):
    """Cycle through a DataLoader indefinitely."""
    while True:
        yield from loader


# ---------------------------------------------------------------------------
# Convenience: pre-built loss functions for common model types
# ---------------------------------------------------------------------------

def classification_loss_fn(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Standard cross-entropy loss for a classification model.

    Expects batch = (inputs, targets) where model(inputs) returns logits.
    """
    inputs, targets = batch
    logits = model(inputs)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return torch.nn.functional.cross_entropy(logits, targets, reduction="none")


def autoregressive_loss_fn(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Per-sequence mean token-level cross-entropy for an autoregressive LM.

    Expects batch = (input_ids, target_ids) where:
        input_ids  : [B, T] integer token ids
        target_ids : [B, T] integer token ids (next-token targets)
    Returns: [B] mean loss per sequence.
    """
    input_ids, target_ids = batch
    logits = model(input_ids)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    B, T, V = logits.shape
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(B * T, V),
        target_ids.reshape(B * T),
        reduction="none",
    ).reshape(B, T)
    return loss.mean(dim=1)   # [B]


def autoregressive_token_loss_fn(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Per-token cross-entropy losses for an autoregressive LM.

    Returns: [B, T]
    """
    input_ids, target_ids = batch
    logits = model(input_ids)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    B, T, V = logits.shape
    return torch.nn.functional.cross_entropy(
        logits.reshape(B * T, V),
        target_ids.reshape(B * T),
        reduction="none",
    ).reshape(B, T)
