"""Measurements that answer the three questions this study poses.

  who trains whom   `grad_attribution` -- the norm of the gradient each expert
                    receives from *each modality's* loss. Off-diagonal entries
                    can only be non-zero because attention is global, so this
                    quantifies the cross-modal training path directly.
  where fusion is   `attention_mass` -- how much attention probability flows
                    from one modality's queries onto another's keys, per layer.
  how specialised   `expert_divergence` (weights) and `layerwise_cka`
                    (representations).

All probes run on a fixed held-out batch so their variation over time reflects
the model changing, not the data changing.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from mot.data import PAIRED_IT, PAIRED_TI, Batch
from mot.model import SUBMODULE_NAMES, MoTModel
from mot.vocab import IMAGE, TEXT


def per_modality_losses(
    logits: torch.Tensor, batch: Batch, n_modalities: int
) -> dict[int, torch.Tensor]:
    """Cross-entropy restricted to each modality's *target* positions.

    Each is a mean over its own positions, so the two are on a comparable scale
    regardless of how lopsided the token mixture is.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_y = batch.y.reshape(-1)
    flat_mod = batch.mod_y.reshape(-1)
    token_loss = F.cross_entropy(flat_logits, flat_y, reduction="none")

    out: dict[int, torch.Tensor] = {}
    for m in range(n_modalities):
        mask = flat_mod == m
        if not bool(mask.any()):
            raise ValueError(f"batch contains no target tokens for modality {m}")
        out[m] = token_loss[mask].mean()
    return out


def pure_modality_losses(
    logits: torch.Tensor, batch: Batch, n_modalities: int
) -> dict[int, torch.Tensor]:
    """Loss on positions whose input *and* target are both modality m.

    Used only by `grad_attribution`. Modality-boundary positions -- a caption's
    <eot> predicting <boi>, say -- are routed to one expert while their loss
    belongs to the other modality, so they leak gradient across the diagonal for
    reasons that have nothing to do with attention. Excluding them is what makes
    the blocked-attention control exact: with cross-modal attention off, the
    off-diagonal must be *identically* zero.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    token_loss = F.cross_entropy(flat_logits, batch.y.reshape(-1), reduction="none")
    mod_x, mod_y = batch.mod_x.reshape(-1), batch.mod_y.reshape(-1)

    out: dict[int, torch.Tensor] = {}
    for m in range(n_modalities):
        mask = (mod_x == m) & (mod_y == m)
        if not bool(mask.any()):
            raise ValueError(f"batch has no pure positions for modality {m}")
        out[m] = token_loss[mask].mean()
    return out


def conditional_losses(logits: torch.Tensor, batch: Batch) -> dict[str, float]:
    """Losses on positions that are only predictable using the *other* modality.

    In a caption->image document the image tokens are determined by the caption
    and nothing else, so `image_given_text` is the sharpest test of whether
    cross-modal information is actually being used.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    token_loss = F.cross_entropy(flat_logits, batch.y.reshape(-1), reduction="none")
    mod, doc = batch.mod_y.reshape(-1), batch.doc_type_y.reshape(-1)

    def masked_mean(mask: torch.Tensor) -> float:
        return float(token_loss[mask].mean()) if bool(mask.any()) else float("nan")

    return {
        "image_given_text": masked_mean((mod == IMAGE) & (doc == PAIRED_TI)),
        "text_given_image": masked_mean((mod == TEXT) & (doc == PAIRED_IT)),
    }


@torch.no_grad()
def _expert_grid(model: MoTModel) -> tuple[int, int]:
    return model.cfg.n_layers, model.cfg.n_experts


def grad_attribution(model: MoTModel, batch: Batch) -> np.ndarray:
    """G[layer, expert, loss_modality] = || d L_modality / d theta_expert ||.

    One backward pass per modality loss. Gradients are cleared afterwards, so a
    probe never contributes to the optimiser step that follows it.
    """
    n_layers, n_experts = _expert_grid(model)
    n_mod = model.cfg.n_modalities
    grid = model.expert_params()
    out = np.zeros((n_layers, n_experts, n_mod), dtype=np.float64)

    logits, _ = model(batch.x, batch.mod_x)
    losses = pure_modality_losses(logits, batch, n_mod)

    for m_loss in range(n_mod):
        model.zero_grad(set_to_none=True)
        losses[m_loss].backward(retain_graph=(m_loss < n_mod - 1))
        for (layer, expert), params in grid.items():
            total = sum(float(p.grad.pow(2).sum()) for p in params if p.grad is not None)
            out[layer, expert, m_loss] = float(np.sqrt(total))

    model.zero_grad(set_to_none=True)
    return out


@torch.no_grad()
def attention_mass(model: MoTModel, batch: Batch) -> np.ndarray:
    """A[layer, query_modality, key_modality]: mean attention probability mass.

    Rows sum to 1, so the off-diagonal is literally the fraction of each
    modality's attention that is spent reading the other modality.
    """
    n_mod = model.cfg.n_modalities
    _, aux = model(batch.x, batch.mod_x, need_probs=True)
    out = np.zeros((model.cfg.n_layers, n_mod, n_mod), dtype=np.float64)

    for layer, probs in enumerate(aux["probs"]):          # (B, H, T, T)
        for m_key in range(n_mod):
            key_mask = (batch.mod_x == m_key)[:, None, None, :]
            mass = (probs * key_mask).sum(-1)             # (B, H, T)
            for m_query in range(n_mod):
                q_mask = (batch.mod_x == m_query)[:, None, :].expand_as(mass)
                out[layer, m_query, m_key] = float(mass[q_mask].mean())
    return out


@torch.no_grad()
def expert_divergence(model: MoTModel) -> dict[str, np.ndarray]:
    """Relative L2 distance between expert 0 and expert 1, per layer.

    Zero at step 0 by construction (identical init), so this curve is a clean
    readout of specialisation caused purely by training.
    """
    n_layers = model.cfg.n_layers
    names = ("overall",) + SUBMODULE_NAMES
    out = {name: np.zeros(n_layers, dtype=np.float64) for name in names}
    if model.cfg.n_experts < 2:
        return {name: np.full(n_layers, np.nan) for name in names}

    for layer in range(n_layers):
        a = model.expert_submodule_params(layer, 0)
        b = model.expert_submodule_params(layer, 1)
        diff_sq = ref_sq = 0.0
        for name in SUBMODULE_NAMES:
            if name not in a:
                # Shared between the modalities, so the two experts hold the
                # same tensor and the divergence is zero by construction rather
                # than by measurement.
                out[name][layer] = 0.0
                continue
            d = sum(float((p - q).pow(2).sum()) for p, q in zip(a[name], b[name]))
            r = sum(float(p.pow(2).sum()) + float(q.pow(2).sum())
                    for p, q in zip(a[name], b[name])) / 2.0
            out[name][layer] = float(np.sqrt(d / r)) if r > 0 else 0.0
            diff_sq += d
            ref_sq += r
        out["overall"][layer] = float(np.sqrt(diff_sq / ref_sq)) if ref_sq > 0 else 0.0
    return out


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, "fro") ** 2
    denom = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(cross / denom) if denom > 0 else float("nan")


@torch.no_grad()
def layerwise_cka(model: MoTModel, batch: Batch) -> np.ndarray:
    """Linear CKA between each document's pooled text and pooled image states.

    Both views describe the *same* scene, so a rise means the two experts are
    mapping their own modality into a shared representation -- synthesis.
    """
    _, aux = model(batch.x, batch.mod_x, need_hidden=True)
    doc_ids = batch.doc_id
    paired = (batch.doc_type_x == PAIRED_TI) | (batch.doc_type_x == PAIRED_IT)

    out = np.full(model.cfg.n_layers, np.nan, dtype=np.float64)
    for layer, hidden in enumerate(aux["hidden"]):        # (B, T, d)
        text_rows, image_rows = [], []
        for b in range(hidden.shape[0]):
            for doc in doc_ids[b].unique():
                sel = (doc_ids[b] == doc) & paired[b]
                t_sel = sel & (batch.mod_x[b] == TEXT)
                i_sel = sel & (batch.mod_x[b] == IMAGE)
                if int(t_sel.sum()) < 2 or int(i_sel.sum()) < 2:
                    continue
                text_rows.append(hidden[b][t_sel].mean(0))
                image_rows.append(hidden[b][i_sel].mean(0))
        if len(text_rows) >= 8:
            out[layer] = _linear_cka(
                torch.stack(text_rows).numpy(), torch.stack(image_rows).numpy())
    return out


class GradientSNR:
    """Per-expert gradient signal-to-noise, tracked with EMAs of g and g*g.

    This is the quantity that distinguishes 'this expert is being trained less'
    from 'this expert is being trained more noisily' -- under Adam a starved
    expert keeps its step size but loses SNR.
    """

    def __init__(self, model: MoTModel, beta: float = 0.95):
        self.beta = beta
        self.grid = model.expert_params()
        self.mean = {k: [torch.zeros_like(p) for p in v] for k, v in self.grid.items()}
        self.sq = {k: [torch.zeros_like(p) for p in v] for k, v in self.grid.items()}
        self.steps = 0

    @torch.no_grad()
    def update(self) -> None:
        self.steps += 1
        for key, params in self.grid.items():
            for i, p in enumerate(params):
                g = torch.zeros_like(p) if p.grad is None else p.grad
                self.mean[key][i].mul_(self.beta).add_(g, alpha=1 - self.beta)
                self.sq[key][i].mul_(self.beta).addcmul_(g, g, value=1 - self.beta)

    @torch.no_grad()
    def snr(self) -> dict[tuple[int, int], float]:
        """||E[g]|| / sqrt(sum Var[g]); bias-corrected, NaN before warm-up."""
        correction = 1 - self.beta ** self.steps if self.steps else 0.0
        out: dict[tuple[int, int], float] = {}
        for key in self.grid:
            if correction <= 0:
                out[key] = float("nan")
                continue
            signal = sum(float(m.pow(2).sum()) for m in self.mean[key]) / correction**2
            second = sum(float(s.sum()) for s in self.sq[key]) / correction
            noise = max(second - signal, 0.0)
            out[key] = float(np.sqrt(signal) / np.sqrt(noise)) if noise > 0 else float("nan")
        return out
