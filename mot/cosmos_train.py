"""Two-tower training, instrumented per step and per probe interval.

Cosmos 3 initialises *both* towers from a pretrained vision-language model and
then trains them together through one joint attention operator, the reasoner
autoregressively and the generator by diffusion. This reproduces that shape:

  0  vision-language pretraining   a plain VLM on the reasoner prefix. Shared by
                                   every arm, cached on disk, so the arms differ
                                   by their variable and not by initialisation.
  1  joint two-tower training      cross-entropy on the reasoner, flow matching
                                   on the generator, both at once.

The measurements that matter are all about the seam between them: how much of
the reasoner's gradient arrives from a diffusion loss computed on positions it
never processes, whether that helps it or damages it, and what the generator is
really conditioned on.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mot import optim as muon_optim
from mot.cosmos_data import SEQ_LEN, make_batch, token_shares
from mot.cosmos_probes import (
    OBJECTIVES, flow_loss, generation_quality, lm_losses, reasoner_ablation,
    tower_attention_mass, tower_attribution,
)
from mot.model import SUBMODULE_NAMES, GeneratorSpec, MoTConfig, MoTModel
from mot.probes import (
    ExpertDrift, GradientSNR, conditional_losses, expert_divergence,
    layerwise_cka,
)
from mot.train import ExpertTracker
from mot.vocab import (
    COSMOS_MODALITY_NAMES, COSMOS_VOCAB_SIZE, GENERATOR, IMAGE,
    N_COSMOS_MODALITIES, TEXT,
)
from mot.world import LATENT_DIM, N_SLOTS

TOWER_LAYOUTS: dict[str, tuple[int, ...]] = {
    # modality -> tower, for (text, image, generator)
    "split": (0, 0, 1),    # Cosmos 3: one reasoner tower, one generator tower
    "dense": (0, 0, 0),    # one transformer for everything
    "three": (0, 1, 2),    # the reasoner itself decoupled over image and text
}


@dataclass
class CosmosRunConfig:
    name: str
    description: str = ""
    steps: int = 1500
    batch_size: int = 32
    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 4

    towers: str = "split"
    optimizer: str = "adamw"
    # Chosen by measurement, not by habit. On this task 1e-3 leaves the
    # generator badly under-trained -- it reconstructs 0.8% of frames exactly,
    # against 3.9% at 3e-3 and 43.8% at 1e-2 -- and every architecture
    # comparison made down there is a comparison between two models that have
    # not finished learning.
    lr: float = 1e-2
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup: int = 50

    lm_loss: bool = True
    flow_weight: float = 1.0
    upcycle: bool = True
    insulate_generator: bool = False
    blocked_attention: bool = False
    freeze_reasoner: bool = False
    decoupled_submodules: tuple[str, ...] | None = None
    shared_from_layer: int | None = None

    probe_every: int = 50
    log_every: int = 10
    probe_batch: int = 16
    eval_batch: int = 128
    decode_steps: int = 8
    seed: int = 0

    # Stage 0. Identical across arms, so it is trained once and cached.
    pretrain_steps: int = 1500
    pretrain_lr: float = 3e-3

    def model_config(self) -> MoTConfig:
        return MoTConfig(
            vocab_size=COSMOS_VOCAB_SIZE,
            n_layers=self.n_layers, d_model=self.d_model, n_heads=self.n_heads,
            n_modalities=N_COSMOS_MODALITIES,
            expert_of_modality=TOWER_LAYOUTS[self.towers],
            seq_len=SEQ_LEN,
            blocked_attention=self.blocked_attention,
            insulate_modality=GENERATOR if self.insulate_generator else None,
            decoupled_submodules=(tuple(self.decoupled_submodules)
                                  if self.decoupled_submodules is not None
                                  else SUBMODULE_NAMES),
            shared_from_layer=self.shared_from_layer,
            generator=GeneratorSpec(LATENT_DIM, N_SLOTS),
            generator_modality=GENERATOR,
            tower_attention=True,
        )

    def pretrain_config(self) -> MoTConfig:
        """The plain vision-language model both towers start from.

        One tower, no generator path, and the reasoner prefix only -- the
        analogue of the pretrained Qwen3-VL checkpoint Cosmos 3 begins from.
        """
        return MoTConfig(
            vocab_size=COSMOS_VOCAB_SIZE,
            n_layers=self.n_layers, d_model=self.d_model, n_heads=self.n_heads,
            n_modalities=N_COSMOS_MODALITIES, expert_of_modality=(0, 0, 0),
            seq_len=SEQ_LEN,
        )

    def pretrain_key(self) -> str:
        return (f"d{self.d_model}_l{self.n_layers}_h{self.n_heads}"
                f"_p{self.pretrain_steps}_b{self.batch_size}"
                f"_lr{self.pretrain_lr}_seed{self.seed}")


def lr_at(base: float, step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def tower_parameters(model: MoTModel, tower: int) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for layer in model.layers:
        for module in layer.submodules(tower).values():
            params += list(module.parameters())
    params += list(model.final_norm[tower].parameters())
    return params


def build_optimizer(cfg: CosmosRunConfig, model: MoTModel, lr: float
                    ) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("this run has nothing to train: every parameter is frozen")
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


# --- stage 0: the pretrained vision-language model ---------------------------

def ensure_pretrained(cfg: CosmosRunConfig, cache_dir: Path, verbose: bool = True
                      ) -> tuple[dict, dict]:
    """Train the vision-language model once, or load it if already on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"vlm_{cfg.pretrain_key()}.pt"
    if path.exists():
        blob = torch.load(path, weights_only=False)
        return blob["state_dict"], blob["capability"]

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = MoTModel(cfg.pretrain_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.pretrain_lr,
                                  betas=(0.9, 0.95))
    started = time.time()
    for step in range(cfg.pretrain_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(cfg.pretrain_lr, step, cfg.pretrain_steps, cfg.warmup)
        batch = make_batch(rng, cfg.batch_size).reasoner_prefix()
        logits, _ = model(batch.x, batch.mod_x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               batch.y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if verbose and step % 250 == 0:
            print(f"  [vlm-pretrain] step {step:5d}/{cfg.pretrain_steps} "
                  f"loss {float(loss.detach()):.4f} ({time.time()-started:.0f}s)",
                  flush=True)

    capability = pretrained_capability(cfg, model)
    torch.save({"state_dict": model.state_dict(), "capability": capability}, path)
    if verbose:
        print(f"  [vlm-pretrain] cached -> {path} | "
              f"{ {k: round(v, 4) for k, v in capability.items()} }", flush=True)
    return model.state_dict(), capability


@torch.no_grad()
def pretrained_capability(cfg: CosmosRunConfig, model: MoTModel) -> dict[str, float]:
    """What the vision-language model can do before a generator is attached."""
    batch = make_batch(np.random.default_rng(50_000 + cfg.seed), cfg.eval_batch)
    prefix = batch.reasoner_prefix()
    logits, _ = model(prefix.x, prefix.mod_x)
    flat = logits.reshape(-1, logits.shape[-1])
    token_loss = F.cross_entropy(flat, prefix.y.reshape(-1), reduction="none")
    mod = prefix.mod_y.reshape(-1)
    out = {}
    for name, m in (("text", TEXT), ("image", IMAGE)):
        mask = mod == m
        out[f"{name}_loss"] = float(token_loss[mask].mean())
    out.update({k: v for k, v in conditional_losses(logits, prefix).items()
                if v == v})
    return out


def load_pretrained(model: MoTModel, state: dict, upcycle: bool) -> None:
    """Put the pretrained VLM into tower 0, and optionally into every tower.

    Cosmos 3 starts *both* towers from the same pretrained weights, so upcycling
    is the faithful default here and `upcycle=False` is the ablation.
    """
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The only keys allowed to go unmatched are the towers above the first --
    # which should start from their own initialisation or be upcycled below --
    # and the generator's own modules, which the pretrained model does not have.
    generator_own = ("gen_in", "gen_out", "gen_slot", "time_mlp")
    stranded = [k for k in missing
                if not k.startswith(generator_own)
                and not any(f".{t}." in k or k.endswith(f".{t}.weight")
                            for t in range(1, model.cfg.n_experts))]
    if stranded or unexpected:
        raise RuntimeError(
            "the pretrained vision-language model did not transfer: "
            f"missing {stranded[:4]}, unexpected {list(unexpected)[:4]}")
    if not upcycle:
        return
    with torch.no_grad():
        for layer in model.layers:
            for name in SUBMODULE_NAMES:
                group = getattr(layer, name)
                for tower in range(1, len(group)):
                    group[tower].load_state_dict(group[0].state_dict())
        for tower in range(1, len(model.final_norm)):
            model.final_norm[tower].load_state_dict(
                model.final_norm[0].state_dict())


# --- stage 1: joint two-tower training ---------------------------------------

def _grid(values: dict[tuple[int, int], float], n_layers: int, n_towers: int):
    return [[round(values[(l, t)], 8) for t in range(n_towers)]
            for l in range(n_layers)]


def compute_loss(cfg: CosmosRunConfig, logits, velocity, batch):
    """Cross-entropy on the reasoner plus flow matching on the generator.

    The two are not commensurable -- one is a per-token likelihood, the other a
    regression on a continuous latent -- so combining them means choosing a
    weight, and the choice is exposed as `flow_weight` rather than hidden. It is
    also why gradient norm, not loss value, is the currency `tower_attribution`
    reports in.
    """
    flow = flow_loss(velocity, batch)
    total = cfg.flow_weight * flow
    parts = {"flow": float(flow.detach())}

    per_modality = lm_losses(logits, batch)
    if cfg.lm_loss and per_modality:
        flat = logits.reshape(-1, logits.shape[-1])
        mask = batch.lm_mask.reshape(-1)
        total = total + F.cross_entropy(flat[mask], batch.y.reshape(-1)[mask])
    for m, value in per_modality.items():
        parts[COSMOS_MODALITY_NAMES[m]] = float(value.detach())
    return total, parts


def _run_probes(cfg, model, probe_batch, val_batch, drift, snr, step) -> dict:
    attribution = tower_attribution(model, probe_batch)
    mass = tower_attention_mass(model, probe_batch)
    cka = layerwise_cka(model, probe_batch.reasoner_prefix())
    ablation = reasoner_ablation(model, probe_batch)
    quality = generation_quality(model, val_batch, cfg.decode_steps)

    n_towers = model.cfg.n_experts
    divergence = {
        f"{a}{b}": {k: np.round(v, 8).tolist()
                    for k, v in expert_divergence(model, (a, b)).items()}
        for a in range(n_towers) for b in range(a + 1, n_towers)
    }
    with torch.no_grad():
        prefix = val_batch.reasoner_prefix()
        logits, _ = model(prefix.x, prefix.mod_x)
        flat = logits.reshape(-1, logits.shape[-1])
        token_loss = F.cross_entropy(flat, prefix.y.reshape(-1), reduction="none")
        mod = prefix.mod_y.reshape(-1)
        reasoner = {f"{name}_loss": round(float(token_loss[mod == m].mean()), 6)
                    for name, m in (("text", TEXT), ("image", IMAGE))}
        reasoner.update({k: round(v, 6)
                         for k, v in conditional_losses(logits, prefix).items()
                         if v == v})

    return {
        "step": step,
        "tower_attribution": np.round(attribution, 10).tolist(),
        "attention_mass": np.round(mass, 8).tolist(),
        "divergence": divergence,
        "cka": np.round(cka, 6).tolist(),
        "ablation": {k: round(v, 6) for k, v in ablation.items()},
        "quality": {k: round(v, 6) for k, v in quality.items()},
        "reasoner": reasoner,
        "drift": _grid(drift.drift(), model.cfg.n_layers, n_towers),
        "grad_snr": _grid(snr.snr(), model.cfg.n_layers, n_towers),
    }


def train(cfg: CosmosRunConfig, out_dir: Path, cache_dir: Path | None = None,
          verbose: bool = True) -> dict:
    torch.manual_seed(cfg.seed)
    cache_dir = cache_dir or (out_dir / "cache")
    state, baseline = ensure_pretrained(cfg, cache_dir, verbose=verbose)

    model = MoTModel(cfg.model_config())
    load_pretrained(model, state, cfg.upcycle)

    if cfg.freeze_reasoner:
        for p in model.parameters():
            p.requires_grad_(True)
        for p in tower_parameters(model, 0):
            p.requires_grad_(False)
        model.embed.weight.requires_grad_(False)

    optimizer = build_optimizer(cfg, model, cfg.lr)
    tracker = ExpertTracker(model)
    drift = ExpertDrift(model)
    snr = GradientSNR(model)
    rng = np.random.default_rng(cfg.seed)

    probe_batch = make_batch(
        np.random.default_rng(10_000 + cfg.seed), cfg.probe_batch,
        flow_t=np.linspace(0.05, 0.95, cfg.probe_batch).astype(np.float32))
    val_batch = make_batch(np.random.default_rng(20_000 + cfg.seed), cfg.eval_batch)

    n_layers, n_towers = cfg.n_layers, model.cfg.n_experts
    log: dict = {
        "config": asdict(cfg),
        "meta": {
            "param_summary": model.param_summary(),
            "seq_len": SEQ_LEN,
            "token_shares": token_shares(),
            "tower_of_modality": list(TOWER_LAYOUTS[cfg.towers]),
            "n_towers": n_towers,
            "n_layers": n_layers,
            "modality_names": list(COSMOS_MODALITY_NAMES),
            "objectives": list(OBJECTIVES),
            "vocab_size": COSMOS_VOCAB_SIZE,
        },
        # What the vision-language model could do before a generator was
        # attached to it. Every reasoner number below is a comparison to this.
        "baseline": {k: round(v, 6) for k, v in baseline.items()},
        "steps": [],
        "probes": [],
    }

    started = time.time()
    for step in range(cfg.steps):
        if step % cfg.probe_every == 0:
            log["probes"].append(
                _run_probes(cfg, model, probe_batch, val_batch, drift, snr, step))
        for group in optimizer.param_groups:
            group["lr"] = lr_at(cfg.lr, step, cfg.steps, cfg.warmup)

        batch = make_batch(rng, cfg.batch_size)
        logits, aux = model(batch.x, batch.mod_x,
                            generator=batch.generator_inputs())
        loss, parts = compute_loss(cfg, logits, aux["velocity"], batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norms = tracker.grad_norms()
        snr.update()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            update, displacement = tracker.movement()
            record = {"step": step, "loss": round(float(loss.detach()), 6),
                      "grad_norm": _grid(grad_norms, n_layers, n_towers),
                      "update_norm": _grid(update, n_layers, n_towers),
                      "displacement": _grid(displacement, n_layers, n_towers)}
            record.update({f"loss_{k}": round(v, 6) for k, v in parts.items()})
            log["steps"].append(record)
            if verbose and step % (cfg.log_every * 25) == 0:
                detail = "  ".join(f"{k} {v:.4f}" for k, v in parts.items())
                print(f"  [{cfg.name}] step {step:5d}/{cfg.steps}  {detail}  "
                      f"({time.time()-started:.0f}s)", flush=True)

    log["probes"].append(
        _run_probes(cfg, model, probe_batch, val_batch, drift, snr, cfg.steps))

    final = make_batch(np.random.default_rng(30_000 + cfg.seed), 512)
    log["meta"]["final_quality"] = {
        k: round(v, 6)
        for k, v in generation_quality(model, final, cfg.decode_steps).items()}
    log["meta"]["wall_seconds"] = round(time.time() - started, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)},
               out_dir / f"{cfg.name}.pt")
    path = out_dir / f"{cfg.name}.json"
    path.write_text(json.dumps(log), encoding="utf-8")
    if verbose:
        print(f"  [{cfg.name}] done in {log['meta']['wall_seconds']}s -> {path}\n"
              f"      final {log['meta']['final_quality']}", flush=True)
    return log
