"""What the two towers do to each other.

Cosmos 3 runs an autoregressive reasoner and a diffusion generator through one
attention operator, with the generator conditioned on the reasoner. That
arrangement raises exactly the question this repo was built to answer, only
sharper than the caption/image study could put it: the reasoner is a working
vision-language model, the generator's objective is not even the same kind of
function, and the two share an attention. So does the diffusion objective train
the reasoner, or damage it, or both?

  `tower_attribution`  the gradient each tower receives from each objective,
                       across a cross-entropy and a mean squared error. The
                       off-diagonal is the whole cross-tower path, and it is
                       non-zero only because the attention is joint.
  `generation_quality` the flow loss, and -- separately -- whether the decoded
                       frame is actually the right frame.
  `reasoner_ablation`  cut the generator's view of the text or of the image and
                       re-measure. What the generator is really conditioned on.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from mot.cosmos_data import CosmosBatch, decode_batch
from mot.model import MoTModel
from mot.vocab import GENERATOR, IMAGE, TEXT

OBJECTIVES = ("text", "image", "generator")


def flow_loss(velocity: torch.Tensor, batch: CosmosBatch) -> torch.Tensor:
    """Mean squared error against `u = noise - z`, over every generator slot."""
    return (velocity - batch.velocity_target()).pow(2).mean()


def lm_losses(logits: torch.Tensor, batch: CosmosBatch) -> dict[int, torch.Tensor]:
    """Cross-entropy per reasoner modality, over positions with a real target."""
    flat = logits.reshape(-1, logits.shape[-1])
    token_loss = F.cross_entropy(flat, batch.y.reshape(-1), reduction="none")
    mod_y, valid = batch.mod_y.reshape(-1), batch.lm_mask.reshape(-1)
    out: dict[int, torch.Tensor] = {}
    for m in (TEXT, IMAGE):
        mask = valid & (mod_y == m)
        if bool(mask.any()):
            out[m] = token_loss[mask].mean()
    return out


def pure_lm_losses(logits: torch.Tensor, batch: CosmosBatch
                   ) -> dict[int, torch.Tensor]:
    """As above, but only where input and target share a modality.

    Boundary positions -- the `<eot>` that predicts a `<boi>` -- are routed to
    one tower while their loss belongs to another modality, and would put mass
    on the attribution off-diagonal for reasons that have nothing to do with
    attention. Excluding them is what lets the blocked control come out at
    exactly zero rather than merely small.
    """
    flat = logits.reshape(-1, logits.shape[-1])
    token_loss = F.cross_entropy(flat, batch.y.reshape(-1), reduction="none")
    mod_x, mod_y = batch.mod_x.reshape(-1), batch.mod_y.reshape(-1)
    valid = batch.lm_mask.reshape(-1)
    out: dict[int, torch.Tensor] = {}
    for m in (TEXT, IMAGE):
        mask = valid & (mod_x == m) & (mod_y == m)
        if bool(mask.any()):
            out[m] = token_loss[mask].mean()
    return out


def tower_attribution(model: MoTModel, batch: CosmosBatch) -> np.ndarray:
    """G[layer, tower, objective] = || d L_objective / d theta_tower ||.

    Objectives are ordered as `OBJECTIVES`. The entry that matters is
    [:, reasoner, generator]: gradient arriving in the vision-language tower
    from a diffusion loss computed on positions it never processes.
    """
    n_layers, n_towers = model.cfg.n_layers, model.cfg.n_experts
    grid = model.expert_params()
    out = np.full((n_layers, n_towers, len(OBJECTIVES)), np.nan)

    logits, aux = model(batch.x, batch.mod_x,
                        generator=batch.generator_inputs())
    losses = dict(pure_lm_losses(logits, batch))
    losses[GENERATOR] = flow_loss(aux["velocity"], batch)

    present = [m for m in (TEXT, IMAGE, GENERATOR) if m in losses]
    for i, m in enumerate(present):
        model.zero_grad(set_to_none=True)
        losses[m].backward(retain_graph=(i < len(present) - 1))
        for (layer, tower), params in grid.items():
            total = sum(float(p.grad.pow(2).sum())
                        for p in params if p.grad is not None)
            out[layer, tower, m] = float(np.sqrt(total))

    model.zero_grad(set_to_none=True)
    return out


@torch.no_grad()
def decode_future(model: MoTModel, batch: CosmosBatch, n_steps: int = 8,
                  attn_allowed: torch.Tensor | None = None) -> torch.Tensor:
    """Integrate the learned velocity field from noise to a predicted frame.

    Euler, starting from the batch's own noise draw, so two calls on the same
    probe batch differ only by the model.
    """
    z = batch.noise.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        tau = torch.full((z.shape[0],), 1.0 - i * dt, dtype=torch.float32,
                         device=z.device)
        _, aux = model(batch.x, batch.mod_x,
                       generator=batch.generator_inputs(noisy=z, flow_t=tau),
                       attn_allowed=attn_allowed)
        z = z - dt * aux["velocity"]
    return z


@torch.no_grad()
def generation_quality(model: MoTModel, batch: CosmosBatch, n_steps: int = 8
                       ) -> dict[str, float]:
    """The loss, and the thing the loss is a proxy for."""
    _, aux = model(batch.x, batch.mod_x, generator=batch.generator_inputs())
    decoded = decode_future(model, batch, n_steps)
    cell, exact = decode_batch(decoded, batch)
    return {"flow_loss": float(flow_loss(aux["velocity"], batch)),
            "cell_accuracy": cell,
            "exact_frame": exact}


@torch.no_grad()
def reasoner_ablation(model: MoTModel, batch: CosmosBatch) -> dict[str, float]:
    """Flow loss when the generator is forbidden to read one reasoner modality.

    The next frame is determined by the rule and the board together and by
    neither alone, so a generator that has learned to condition on its reasoner
    must get worse when either is cut. Read it as dependence rather than skill:
    removing an attention edge is an out-of-distribution perturbation, not a
    marginalisation, and it degrades further than the missing information alone
    accounts for.
    """
    base = model.tower_mask(batch.is_generator)
    reader = batch.is_generator[:, :, None]

    def without(modality: int | None) -> float:
        allowed = base
        if modality is not None:
            keys = (batch.mod_x == modality)[:, None, :]
            allowed = base & ~(reader & keys).unsqueeze(1)
        _, aux = model(batch.x, batch.mod_x,
                       generator=batch.generator_inputs(), attn_allowed=allowed)
        return float(flow_loss(aux["velocity"], batch))

    return {"full": without(None), "no_text": without(TEXT),
            "no_image": without(IMAGE)}


@torch.no_grad()
def tower_attention_mass(model: MoTModel, batch: CosmosBatch) -> np.ndarray:
    """A[layer, query_modality, key_modality]: mean attention probability mass.

    The generator row is where the conditioning actually happens, layer by
    layer.
    """
    _, aux = model(batch.x, batch.mod_x, generator=batch.generator_inputs(),
                   need_probs=True)
    n_mod = 3
    out = np.full((model.cfg.n_layers, n_mod, n_mod), np.nan)
    for layer, probs in enumerate(aux["probs"]):
        for key in range(n_mod):
            key_mask = (batch.mod_x == key)[:, None, None, :]
            if not bool(key_mask.any()):
                continue
            mass = (probs * key_mask).sum(-1)
            for query in range(n_mod):
                q = (batch.mod_x == query)[:, None, :].expand_as(mass)
                if bool(q.any()):
                    out[layer, query, key] = float(mass[q].mean())
    return out
