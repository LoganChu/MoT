"""Load run logs into arrays, and derive the quantities the write-up argues from."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TEXT, IMAGE = 0, 1


@dataclass
class RunLog:
    name: str
    config: dict
    meta: dict
    step: np.ndarray            # (S,)
    loss: np.ndarray            # (S,)
    loss_text: np.ndarray       # (S,)
    loss_image: np.ndarray      # (S,)
    tokens: np.ndarray          # (S, M)
    grad_norm: np.ndarray       # (S, L, E)
    update_norm: np.ndarray     # (S, L, E)
    displacement: np.ndarray    # (S, L, E)
    probe_step: np.ndarray      # (P,)
    attribution: np.ndarray     # (P, L, E, M)  expert <- loss-modality
    attention_mass: np.ndarray  # (P, L, M, M)  query-modality -> key-modality
    divergence: dict[str, np.ndarray]
    cka: np.ndarray             # (P, L)
    val_loss_text: np.ndarray   # (P,)
    val_loss_image: np.ndarray  # (P,)
    conditional: dict[str, np.ndarray]
    grad_snr: np.ndarray        # (P, L, E)

    @property
    def n_layers(self) -> int:
        return int(self.meta["n_layers"])

    @property
    def n_experts(self) -> int:
        return int(self.meta["n_experts"])

    @property
    def is_dense(self) -> bool:
        return self.n_experts < 2

    @property
    def text_frac(self) -> float:
        return float(self.meta["measured_text_frac"])


def load_run(path: Path) -> RunLog:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    steps, probes = raw["steps"], raw["probes"]

    def stack(records, key):
        return np.array([r[key] for r in records], dtype=np.float64)

    return RunLog(
        name=raw["config"]["name"],
        config=raw["config"],
        meta=raw["meta"],
        step=np.array([s["step"] for s in steps]),
        loss=stack(steps, "loss"),
        loss_text=stack(steps, "loss_text"),
        loss_image=stack(steps, "loss_image"),
        tokens=stack(steps, "tokens"),
        grad_norm=stack(steps, "grad_norm"),
        update_norm=stack(steps, "update_norm"),
        displacement=stack(steps, "displacement"),
        probe_step=np.array([p["step"] for p in probes]),
        attribution=stack(probes, "grad_attribution"),
        attention_mass=stack(probes, "attention_mass"),
        divergence={k: np.array([p["divergence"][k] for p in probes], dtype=np.float64)
                    for k in probes[0]["divergence"]},
        cka=stack(probes, "cka"),
        val_loss_text=stack(probes, "val_loss_text"),
        val_loss_image=stack(probes, "val_loss_image"),
        conditional={k: np.array([p["conditional"][k] for p in probes], dtype=np.float64)
                     for k in probes[0]["conditional"]},
        grad_snr=stack(probes, "grad_snr"),
    )


def load_all(run_dir: Path) -> dict[str, RunLog]:
    return {p.stem: load_run(p) for p in sorted(Path(run_dir).glob("*.json"))}


# --- derived quantities -----------------------------------------------------

def cross_modal_grad_fraction(run: RunLog) -> np.ndarray:
    """(P, E): share of each expert's gradient that comes from the *other* modality.

    The direct answer to "does the starved expert still get trained?" -- if this
    is large for the image expert, joint attention is subsidising it.
    """
    if run.is_dense:
        return np.full((len(run.probe_step), run.n_experts), np.nan)
    total = run.attribution.sum(axis=3)                     # (P, L, E)
    own = np.stack([run.attribution[:, :, e, e] for e in range(run.n_experts)], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(total > 0, 1.0 - own / total, np.nan)
    return np.nanmean(frac, axis=1)                          # mean over layers


def cross_modal_attention_fraction(run: RunLog) -> np.ndarray:
    """(P, L, M): fraction of each modality's attention spent on the other modality."""
    mass = run.attention_mass
    n_mod = mass.shape[-1]
    return np.stack([1.0 - mass[:, :, m, m] for m in range(n_mod)], axis=2)


def expert_balance_ratio(run: RunLog, field: str = "displacement") -> np.ndarray:
    """(S,): image-expert quantity divided by text-expert quantity, averaged over layers.

    1.0 means the two transformers are being trained equally hard.
    """
    values = getattr(run, field)
    if run.is_dense:
        return np.full(values.shape[0], np.nan)
    per_layer = values[:, :, IMAGE] / np.where(values[:, :, TEXT] > 0,
                                               values[:, :, TEXT], np.nan)
    return np.nanmean(per_layer, axis=1)


def summarize(run: RunLog) -> dict:
    """Compact end-of-training summary, used for the report and the artifact."""
    final = -1
    disp = run.displacement[final]                    # (L, E)
    snr = run.grad_snr[final]
    out = {
        "name": run.name,
        "description": run.config.get("description", ""),
        "text_frac": round(run.text_frac, 4),
        "steps": int(run.config["steps"]),
        "wall_seconds": run.meta.get("wall_seconds"),
        "final_loss_text": round(float(run.loss_text[final]), 4),
        "final_loss_image": round(float(run.loss_image[final]), 4),
        "val_loss_text": round(float(run.val_loss_text[-1]), 4),
        "val_loss_image": round(float(run.val_loss_image[-1]), 4),
        "image_given_text": round(float(run.conditional["image_given_text"][-1]), 4),
        "text_given_image": round(float(run.conditional["text_given_image"][-1]), 4),
        "displacement_text": round(float(disp[:, TEXT].mean()), 4),
        "token_share_image": round(float(1 - run.text_frac), 4),
    }
    if not run.is_dense:
        out.update({
            "displacement_image": round(float(disp[:, IMAGE].mean()), 4),
            "displacement_ratio": round(float(disp[:, IMAGE].mean() / disp[:, TEXT].mean()), 4),
            "update_ratio": round(float(np.nanmean(expert_balance_ratio(run, "update_norm")[-20:])), 4),
            "grad_norm_ratio": round(float(np.nanmean(expert_balance_ratio(run, "grad_norm")[-20:])), 4),
            "snr_text": round(float(np.nanmean(snr[:, TEXT])), 4),
            "snr_image": round(float(np.nanmean(snr[:, IMAGE])), 4),
            "snr_ratio": round(float(np.nanmean(snr[:, IMAGE]) / np.nanmean(snr[:, TEXT])), 4),
            "cross_grad_frac_text": round(float(cross_modal_grad_fraction(run)[-1, TEXT]), 4),
            "cross_grad_frac_image": round(float(cross_modal_grad_fraction(run)[-1, IMAGE]), 4),
            "divergence_final": round(float(run.divergence["overall"][-1].mean()), 4),
        })
    return out
