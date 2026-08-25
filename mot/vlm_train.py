"""Staged VLM training, instrumented at every stage boundary.

VILA's recipe is three stages on top of a language model, and every one of its
findings is a comparison *across* a stage boundary, so the training loop has to
be built out of stages rather than have them bolted on:

  0  text pretraining    the language model. Shared by every arm and cached on
                         disk, so all arms start from bit-identical weights and
                         the comparison is exact rather than approximate.
  1  align               freeze the language model, train the modality-specific
                         image weights. The projector-initialisation analogue --
                         in a Mixture-of-Transformers the image expert *is* the
                         projector, since it is the only thing an image token
                         ever passes through.
  2  multimodal pretrain the arm under test: pairs, interleaved, or blended.
  3  instruction tuning  with or without text-only data mixed back in.

Capabilities are measured on fixed held-out batches at every probe interval and
at every boundary, so "what did stage 2 cost the language model" is a
subtraction rather than an inference.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mot.model import SUBMODULE_NAMES, MoTConfig, MoTModel
from mot import optim as muon_optim
from mot.probes import (
    GradientSNR, attention_mass, conditional_losses, expert_divergence,
    grad_attribution, layerwise_cka,
)
from mot.train import ExpertTracker
from mot.vlm_data import REGIMES, Regime, association_eval_batch, make_batch, \
    statement_eval_batch, token_shares
from mot.vlm_probes import ExpertDrift, text_capability
from mot.vocab import IMAGE, MODALITY_NAMES, N_MODALITIES, TEXT, VLM_VOCAB_SIZE

SHOTS = (0, 4)


@dataclass(frozen=True)
class Stage:
    """One phase of the recipe."""

    name: str
    regime: str
    steps: int
    freeze: str = "none"        # none | text_expert | text_expert_and_embed
    lr: float | None = None     # None = the run's default


@dataclass
class VLMRunConfig:
    name: str
    description: str = ""
    stages: tuple[Stage, ...] = ()
    batch_size: int = 16
    seq_len: int = 192
    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 4
    optimizer: str = "adamw"
    lr: float = 3e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup: int = 50
    dense: bool = False
    blocked_attention: bool = False
    # Which sub-modules get one copy per modality. None means all of them,
    # which is MoT as published.
    decoupled_submodules: tuple[str, ...] | None = None
    shared_from_layer: int | None = None
    insulate_image_gradient: bool = False
    upcycle: bool = False
    probe_every: int = 50
    log_every: int = 10
    probe_batch: int = 8
    eval_batch: int = 256
    seed: int = 0

    # Shared stage 0. Identical across arms, so it is trained once and cached.
    text_steps: int = 1200
    text_lr: float = 3e-3

    def model_config(self) -> MoTConfig:
        return MoTConfig(
            vocab_size=VLM_VOCAB_SIZE,
            n_layers=self.n_layers, d_model=self.d_model, n_heads=self.n_heads,
            n_modalities=N_MODALITIES,
            expert_of_modality=(0, 0) if self.dense else (0, 1),
            seq_len=self.seq_len,
            blocked_attention=self.blocked_attention,
            insulate_modality=IMAGE if self.insulate_image_gradient else None,
            decoupled_submodules=(tuple(self.decoupled_submodules)
                                  if self.decoupled_submodules is not None
                                  else SUBMODULE_NAMES),
            shared_from_layer=self.shared_from_layer,
        )

    def text_base_key(self) -> str:
        """Anything that changes the pretrained language model changes this.

        Deliberately *not* the decoupling or the optimiser. A real VLM starts
        from one language model and the choice of how to attach a vision side is
        made afterwards, so every arm here loads the same checkpoint into
        whatever architecture it is testing. That also makes the architecture
        arms comparable to each other, which they would not be if each had
        pretrained its own language model.
        """
        return (f"d{self.d_model}_l{self.n_layers}_h{self.n_heads}"
                f"_t{self.text_steps}_b{self.batch_size}_s{self.seq_len}"
                f"_lr{self.text_lr}_seed{self.seed}")

    def base_model_config(self) -> MoTConfig:
        """The architecture the shared language model is pretrained in.

        Fully decoupled and two-expert, so its checkpoint is a superset: any arm
        can take expert 0 out of it and ignore the rest.
        """
        return MoTConfig(
            vocab_size=VLM_VOCAB_SIZE,
            n_layers=self.n_layers, d_model=self.d_model, n_heads=self.n_heads,
            n_modalities=N_MODALITIES, expert_of_modality=(0, 1),
            seq_len=self.seq_len,
        )


# --- parameter groups -------------------------------------------------------

def expert_parameters(model: MoTModel, expert: int) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for layer in model.layers:
        for module in layer.submodules(expert).values():
            params += list(module.parameters())
    params += list(model.final_norm[expert].parameters())
    return params


def set_freeze(model: MoTModel, spec: str) -> None:
    """Apply a freezing specification, from scratch each time."""
    for p in model.parameters():
        p.requires_grad_(True)
    if spec == "none":
        return
    if spec not in ("text_expert", "text_expert_and_embed"):
        raise ValueError(f"unknown freeze spec {spec!r}")
    for p in expert_parameters(model, model.cfg.expert_of_modality[TEXT]):
        p.requires_grad_(False)
    if spec == "text_expert_and_embed":
        model.embed.weight.requires_grad_(False)


def upcycle_image_expert(model: MoTModel) -> None:
    """Copy the pretrained text expert onto the image expert.

    Sparse upcycling. The image expert comes out of text pretraining still at
    its random initialisation, because no image token ever reached it; this
    starts it from a trained transformer instead, which is what every real VLM
    does when it initialises a vision tower from something pretrained.
    """
    if model.cfg.n_experts < 2:
        return
    with torch.no_grad():
        for layer in model.layers:
            for name, module in layer.submodules(0).items():
                getattr(layer, name)[1].load_state_dict(module.state_dict())
        model.final_norm[1].load_state_dict(model.final_norm[0].state_dict())


# --- optimisation -----------------------------------------------------------

def build_optimizer(cfg: VLMRunConfig, model: MoTModel, lr: float
                    ) -> torch.optim.Optimizer:
    """Built fresh per stage, over the currently trainable parameters only.

    Each stage of a real recipe is a separate training run with its own
    optimiser state, and carrying Adam moments across a freeze boundary would
    let a stage be influenced by gradients from a phase in which those weights
    were not even being trained.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError(
            "this stage has nothing to train: every parameter is frozen. A "
            "dense model has no modality-specific weights, so freezing the "
            "language model freezes all of it -- see `vlm_dense` in "
            "mot/vlm_configs.py for how that is handled.")
    if cfg.optimizer == "muon":
        return muon_optim.build(params, lr=lr, weight_decay=cfg.weight_decay,
                                adamw_lr=cfg.lr)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay,
                                 betas=(0.9, 0.95))
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9,
                               weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def lr_at(base_lr: float, step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def modality_losses(logits: torch.Tensor, batch) -> dict[int, torch.Tensor]:
    """Per-modality cross-entropy, skipping modalities absent from the batch.

    The text-only stage has no image tokens at all, so this cannot be the
    caption/image study's version, which treats an absent modality as a bug.
    """
    flat = logits.reshape(-1, logits.shape[-1])
    token_loss = F.cross_entropy(flat, batch.y.reshape(-1), reduction="none")
    mod = batch.mod_y.reshape(-1)
    out: dict[int, torch.Tensor] = {}
    for m in range(N_MODALITIES):
        mask = mod == m
        if bool(mask.any()):
            out[m] = token_loss[mask].mean()
    return out


# --- stage 0, shared and cached ---------------------------------------------

def ensure_text_base(cfg: VLMRunConfig, cache_dir: Path, verbose: bool = True
                     ) -> tuple[dict, dict]:
    """Train the language model once, or load it if it is already on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"text_base_{cfg.text_base_key()}.pt"
    if path.exists():
        blob = torch.load(path, weights_only=False)
        return blob["state_dict"], blob["capability"]

    # `build_text_evals` is defined below; the language model is scored on the
    # same fixed batches every arm will later be scored on.

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = MoTModel(cfg.base_model_config())
    set_freeze(model, "none")
    optimizer = build_optimizer(cfg, model, cfg.text_lr)
    regime = REGIMES["text_only"]
    evals = build_text_evals(cfg)
    curve: list[dict] = []

    started = time.time()
    for step in range(cfg.text_steps):
        if step % max(cfg.probe_every, 1) == 0:
            curve.append({"step": step,
                          **{k: round(v, 6)
                             for k, v in text_capability(model, evals).items()}})
        lr = lr_at(cfg.text_lr, step, cfg.text_steps, cfg.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = make_batch(rng, cfg.batch_size, cfg.seq_len, regime)
        logits, _ = model(batch.x, batch.mod_x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               batch.y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if verbose and step % 200 == 0:
            print(f"  [text-base] step {step:5d}/{cfg.text_steps} "
                  f"loss {float(loss.detach()):.4f} ({time.time()-started:.0f}s)",
                  flush=True)

    capability = text_capability(model, evals)
    curve.append({"step": cfg.text_steps, **{k: round(v, 6)
                                             for k, v in capability.items()}})
    torch.save({"state_dict": model.state_dict(), "capability": capability,
                "curve": curve}, path)
    if verbose:
        print(f"  [text-base] cached -> {path}", flush=True)
        for row in curve[::max(len(curve) // 8, 1)]:
            print(f"      step {row['step']:5d}  fact {row['statement_acc']:.3f}  "
                  f"0-shot {row['assoc0_acc']:.3f}  4-shot {row['assoc4_acc']:.3f}",
                  flush=True)
    return model.state_dict(), capability


def load_text_base(model: MoTModel, state: dict) -> None:
    """Put the pretrained language model into expert 0 of any architecture.

    Expert 0 is index 0 in every layout -- decoupled, partially shared, or
    dense -- so the parameter names line up and the only keys that go unmatched
    are the image expert's, which is exactly what should stay at its own
    initialisation. Anything else missing would mean the language model did not
    actually transfer, so it is checked rather than trusted.
    """
    missing, unexpected = model.load_state_dict(state, strict=False)
    stranded = [k for k in missing if ".1." not in k]
    if stranded:
        raise RuntimeError(
            f"the pretrained language model did not transfer: {stranded[:5]}")


def build_text_evals(cfg: VLMRunConfig) -> dict:
    """Fixed held-out batches for the text capabilities. Same for every arm."""
    rng = np.random.default_rng(90_000 + cfg.seed)
    return {
        "statement": statement_eval_batch(rng, cfg.eval_batch),
        "association": {k: association_eval_batch(rng, cfg.eval_batch, k)
                        for k in SHOTS},
    }


# --- stages 1..n ------------------------------------------------------------

def _grid(values: dict[tuple[int, int], float], n_layers: int, n_experts: int):
    return [[round(values[(l, e)], 8) for e in range(n_experts)]
            for l in range(n_layers)]


def _run_probes(model, evals, probe_batch, val_batch, drift, snr,
                stage_name: str, step: int) -> dict:
    text = text_capability(model, evals)
    attribution = grad_attribution(model, probe_batch)
    mass = attention_mass(model, probe_batch)
    divergence = expert_divergence(model)
    cka = layerwise_cka(model, probe_batch)

    with torch.no_grad():
        logits, _ = model(val_batch.x, val_batch.mod_x)
        per_modality = modality_losses(logits, val_batch)
        conditional = conditional_losses(logits, val_batch)

    n_layers, n_experts = model.cfg.n_layers, model.cfg.n_experts
    return {
        "stage": stage_name,
        "step": step,
        "text": {k: round(v, 6) for k, v in text.items()},
        "visual": {MODALITY_NAMES[m]: round(float(v), 6)
                   for m, v in per_modality.items()},
        "conditional": {k: round(v, 6) for k, v in conditional.items()},
        "drift": _grid(drift.drift(), n_layers, n_experts),
        "grad_attribution": np.round(attribution, 10).tolist(),
        "attention_mass": np.round(mass, 8).tolist(),
        "divergence": {k: np.round(v, 8).tolist() for k, v in divergence.items()},
        "cka": np.round(cka, 6).tolist(),
        "grad_snr": _grid(snr.snr(), n_layers, n_experts),
    }


def train(cfg: VLMRunConfig, out_dir: Path, cache_dir: Path | None = None,
          verbose: bool = True) -> dict:
    torch.manual_seed(cfg.seed)
    cache_dir = cache_dir or (out_dir / "cache")
    state, baseline = ensure_text_base(cfg, cache_dir, verbose=verbose)

    model = MoTModel(cfg.model_config())
    load_text_base(model, state)
    if cfg.upcycle:
        upcycle_image_expert(model)

    evals = build_text_evals(cfg)
    probe_batch = make_batch(np.random.default_rng(10_000 + cfg.seed),
                             cfg.probe_batch, cfg.seq_len, REGIMES["pairs"])
    val_batch = make_batch(np.random.default_rng(20_000 + cfg.seed),
                           cfg.batch_size, cfg.seq_len, REGIMES["pairs"])

    tracker = ExpertTracker(model)
    drift = ExpertDrift(model)
    snr = GradientSNR(model)
    rng = np.random.default_rng(cfg.seed)

    n_layers, n_experts = cfg.n_layers, model.cfg.n_experts
    log: dict = {
        "config": {**asdict(cfg), "stages": [asdict(s) for s in cfg.stages]},
        "meta": {
            "param_summary": model.param_summary(),
            "n_experts": n_experts,
            "n_layers": n_layers,
            "modality_names": list(MODALITY_NAMES),
            "vocab_size": VLM_VOCAB_SIZE,
            "shots": list(SHOTS),
            "regime_shares": {
                s.regime: token_shares(np.random.default_rng(7), REGIMES[s.regime],
                                       n_seq=16, seq_len=cfg.seq_len)
                for s in cfg.stages
            },
        },
        # What the language model could do before it ever saw an image. Every
        # forgetting number below is a subtraction from this.
        "baseline": baseline,
        "steps": [],
        "probes": [],
    }

    started = time.time()
    for stage in cfg.stages:
        set_freeze(model, stage.freeze)
        optimizer = build_optimizer(cfg, model, stage.lr or cfg.lr)
        regime: Regime = REGIMES[stage.regime]
        base_lr = stage.lr or cfg.lr

        for step in range(stage.steps):
            if step % cfg.probe_every == 0:
                log["probes"].append(_run_probes(model, evals, probe_batch,
                                                 val_batch, drift, snr,
                                                 stage.name, step))
            lr = lr_at(base_lr, step, stage.steps, cfg.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr

            batch = make_batch(rng, cfg.batch_size, cfg.seq_len, regime)
            logits, _ = model(batch.x, batch.mod_x)
            per_modality = modality_losses(logits, batch)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   batch.y.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norms = tracker.grad_norms()
            snr.update()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
            optimizer.step()

            if step % cfg.log_every == 0 or step == stage.steps - 1:
                update, displacement = tracker.movement()
                counts = torch.bincount(batch.mod_y.reshape(-1),
                                        minlength=N_MODALITIES)
                log["steps"].append({
                    "stage": stage.name,
                    "step": step,
                    "lr": round(lr, 8),
                    "loss": round(float(loss.detach()), 6),
                    "loss_text": round(float(per_modality[TEXT].detach()), 6),
                    "loss_image": (round(float(per_modality[IMAGE].detach()), 6)
                                   if IMAGE in per_modality else None),
                    "tokens": counts.tolist(),
                    "grad_norm": _grid(grad_norms, n_layers, n_experts),
                    "update_norm": _grid(update, n_layers, n_experts),
                    "displacement": _grid(displacement, n_layers, n_experts),
                })
            if verbose and step % (cfg.log_every * 20) == 0:
                print(f"  [{cfg.name}/{stage.name}] step {step:5d}/{stage.steps} "
                      f"loss {float(loss.detach()):.4f} "
                      f"({time.time()-started:.0f}s)", flush=True)

        # Boundary measurement: this is what the stage cost or bought.
        log["probes"].append(_run_probes(model, evals, probe_batch, val_batch,
                                         drift, snr, stage.name, stage.steps))

    log["meta"]["wall_seconds"] = round(time.time() - started, 1)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The finished model, so a measurement that turns out to be the wrong one
    # can be recomputed without spending an hour retraining fourteen arms.
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)},
               out_dir / f"{cfg.name}.pt")
    path = out_dir / f"{cfg.name}.json"
    path.write_text(json.dumps(log), encoding="utf-8")
    if verbose:
        final = log["probes"][-1]["text"]
        print(f"  [{cfg.name}] done in {log['meta']['wall_seconds']}s -> {path}\n"
              f"      baseline {baseline}\n      final    {final}", flush=True)
    return log
