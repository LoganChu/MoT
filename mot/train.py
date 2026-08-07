"""Training loop, instrumented per step and per probe interval."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mot.data import Batch, make_batch, measure_token_ratio, solve_text_only_prob
from mot.model import MoTConfig, MoTModel
from mot.probes import (
    GradientSNR, attention_mass, conditional_losses, expert_divergence,
    grad_attribution, layerwise_cka, per_modality_losses,
)
from mot.vocab import MODALITY_NAMES, N_MODALITIES, VOCAB_SIZE


@dataclass
class RunConfig:
    name: str
    description: str = ""
    steps: int = 1500
    batch_size: int = 12
    seq_len: int = 192
    target_text_frac: float = 0.5
    optimizer: str = "adamw"                 # adamw | sgd
    lr: float = 3e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup: int = 100
    loss_weighting: str = "token_mean"       # token_mean | modality_normalized
    blocked_attention: bool = False
    dense: bool = False
    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 4
    probe_every: int = 25
    log_every: int = 5
    probe_batch: int = 8
    seed: int = 0

    def model_config(self) -> MoTConfig:
        return MoTConfig(
            vocab_size=VOCAB_SIZE,
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_modalities=N_MODALITIES,
            expert_of_modality=(0, 0) if self.dense else tuple(range(N_MODALITIES)),
            seq_len=self.seq_len,
            blocked_attention=self.blocked_attention,
        )


def _lr_at(cfg: RunConfig, step: int) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def _build_optimizer(cfg: RunConfig, model: MoTModel) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9,
                               weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


class ExpertTracker:
    """Tracks per-expert weight movement: how 'trained' each transformer is."""

    def __init__(self, model: MoTModel):
        self.grid = model.expert_params()
        self.init = {k: [p.detach().clone() for p in v] for k, v in self.grid.items()}
        self.prev = {k: [p.detach().clone() for p in v] for k, v in self.grid.items()}
        self.init_norm = {
            k: float(np.sqrt(sum(float(p.pow(2).sum()) for p in v)))
            for k, v in self.init.items()
        }

    @torch.no_grad()
    def grad_norms(self) -> dict[tuple[int, int], float]:
        return {
            key: float(np.sqrt(sum(float(p.grad.pow(2).sum())
                                   for p in params if p.grad is not None)))
            for key, params in self.grid.items()
        }

    @torch.no_grad()
    def movement(self) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
        """(update norm since last call, cumulative displacement relative to init)."""
        update, displacement = {}, {}
        for key, params in self.grid.items():
            step_sq = sum(float((p - q).pow(2).sum())
                          for p, q in zip(params, self.prev[key]))
            init_sq = sum(float((p - q).pow(2).sum())
                          for p, q in zip(params, self.init[key]))
            update[key] = float(np.sqrt(step_sq))
            displacement[key] = float(np.sqrt(init_sq)) / self.init_norm[key]
            for p, q in zip(params, self.prev[key]):
                q.copy_(p)
        return update, displacement


def _grid_to_nested(values: dict[tuple[int, int], float], n_layers: int, n_experts: int):
    return [[round(values[(l, e)], 8) for e in range(n_experts)] for l in range(n_layers)]


def _compute_loss(cfg: RunConfig, logits: torch.Tensor, batch: Batch
                  ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    per_modality = per_modality_losses(logits, batch, N_MODALITIES)
    if cfg.loss_weighting == "token_mean":
        # The default: one mean over every position, so each modality's weight in
        # the objective is proportional to how many tokens it contributed.
        total = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch.y.reshape(-1))
    elif cfg.loss_weighting == "modality_normalized":
        # The mitigation: each modality contributes equally regardless of count.
        total = sum(per_modality.values()) / N_MODALITIES
    else:
        raise ValueError(f"unknown loss_weighting {cfg.loss_weighting!r}")
    return total, per_modality


def train(cfg: RunConfig, out_dir: Path, verbose: bool = True) -> dict:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    probe_rng = np.random.default_rng(10_000 + cfg.seed)
    val_rng = np.random.default_rng(20_000 + cfg.seed)

    p_text_only = solve_text_only_prob(cfg.target_text_frac)
    measured_frac = measure_token_ratio(
        np.random.default_rng(999), p_text_only, n_seq=48, seq_len=cfg.seq_len)

    model = MoTModel(cfg.model_config())
    optimizer = _build_optimizer(cfg, model)
    tracker = ExpertTracker(model)
    snr = GradientSNR(model)

    # Fixed batches: probe variation over time is the model changing, not the data.
    probe_batch = make_batch(probe_rng, cfg.probe_batch, cfg.seq_len, p_text_only)
    val_batch = make_batch(val_rng, cfg.batch_size, cfg.seq_len, p_text_only)

    n_layers, n_experts = cfg.n_layers, model.cfg.n_experts
    log: dict = {
        "config": asdict(cfg),
        "meta": {
            "param_summary": model.param_summary(),
            "measured_text_frac": round(measured_frac, 5),
            "p_text_only": round(p_text_only, 5),
            "n_experts": n_experts,
            "n_layers": n_layers,
            "modality_names": list(MODALITY_NAMES),
            "vocab_size": VOCAB_SIZE,
        },
        "steps": [],
        "probes": [],
    }

    started = time.time()
    for step in range(cfg.steps):
        lr = _lr_at(cfg, step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % cfg.probe_every == 0:
            log["probes"].append(_run_probes(model, probe_batch, val_batch, snr, step,
                                             n_layers, n_experts))

        batch = make_batch(rng, cfg.batch_size, cfg.seq_len, p_text_only)
        logits, _ = model(batch.x, batch.mod_x)
        loss, per_modality = _compute_loss(cfg, logits, batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norms = tracker.grad_norms()
        snr.update()
        if cfg.grad_clip > 0:
            # Uniform rescaling, so per-expert gradient *ratios* are unaffected.
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            update, displacement = tracker.movement()
            counts = torch.bincount(batch.mod_y.reshape(-1), minlength=N_MODALITIES)
            log["steps"].append({
                "step": step,
                "lr": round(lr, 8),
                "loss": round(float(loss), 6),
                "loss_text": round(float(per_modality[0]), 6),
                "loss_image": round(float(per_modality[1]), 6),
                "tokens": counts.tolist(),
                "grad_norm": _grid_to_nested(grad_norms, n_layers, n_experts),
                "update_norm": _grid_to_nested(update, n_layers, n_experts),
                "displacement": _grid_to_nested(displacement, n_layers, n_experts),
            })
            if verbose and step % (cfg.log_every * 10) == 0:
                print(f"  [{cfg.name}] step {step:5d}/{cfg.steps}  loss {float(loss):.4f}  "
                      f"text {float(per_modality[0]):.4f}  image {float(per_modality[1]):.4f}  "
                      f"({time.time()-started:.0f}s)", flush=True)

    log["probes"].append(_run_probes(model, probe_batch, val_batch, snr, cfg.steps,
                                     n_layers, n_experts))
    log["meta"]["wall_seconds"] = round(time.time() - started, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg.name}.json"
    path.write_text(json.dumps(log), encoding="utf-8")
    if verbose:
        print(f"  [{cfg.name}] done in {log['meta']['wall_seconds']}s -> {path}", flush=True)
    return log


def _run_probes(model, probe_batch, val_batch, snr, step, n_layers, n_experts) -> dict:
    attribution = grad_attribution(model, probe_batch)
    mass = attention_mass(model, probe_batch)
    divergence = expert_divergence(model)
    cka = layerwise_cka(model, probe_batch)

    with torch.no_grad():
        val_logits, _ = model(val_batch.x, val_batch.mod_x)
        val_modality = per_modality_losses(val_logits, val_batch, N_MODALITIES)
        conditional = conditional_losses(val_logits, val_batch)

    snr_values = snr.snr()
    return {
        "step": step,
        "grad_attribution": np.round(attribution, 10).tolist(),
        "attention_mass": np.round(mass, 8).tolist(),
        "divergence": {k: np.round(v, 8).tolist() for k, v in divergence.items()},
        "cka": np.round(cka, 6).tolist(),
        "val_loss_text": round(float(val_modality[0]), 6),
        "val_loss_image": round(float(val_modality[1]), 6),
        "conditional": {k: round(v, 6) for k, v in conditional.items()},
        "grad_snr": _grid_to_nested(snr_values, n_layers, n_experts),
    }
