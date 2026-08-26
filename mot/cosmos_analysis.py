"""Load two-tower run logs, and derive the quantities the write-up argues from."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TEXT, IMAGE, GENERATOR = 0, 1, 2


@dataclass
class CosmosRunLog:
    name: str
    config: dict
    meta: dict
    baseline: dict[str, float]
    steps: list[dict]
    probes: list[dict]

    @property
    def n_layers(self) -> int:
        return int(self.meta["n_layers"])

    @property
    def n_towers(self) -> int:
        return int(self.meta["n_towers"])

    @property
    def final(self) -> dict:
        return self.probes[-1]

    @property
    def quality(self) -> dict[str, float]:
        """The end-of-training rollout, scored once on many fresh transitions."""
        return self.meta.get("final_quality", self.final["quality"])

    def tower_of(self, modality: int) -> int:
        return int(self.meta["tower_of_modality"][modality])


def load_cosmos_run(path: Path) -> CosmosRunLog:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return CosmosRunLog(name=raw["config"]["name"], config=raw["config"],
                        meta=raw["meta"], baseline=raw["baseline"],
                        steps=raw["steps"], probes=raw["probes"])


def load_all_cosmos(directory: Path) -> dict[str, CosmosRunLog]:
    return {p.stem: load_cosmos_run(p)
            for p in sorted(Path(directory).glob("*.json"))}


def diffusion_into_reasoner(run: CosmosRunLog) -> float:
    """Share of the reasoner tower's gradient that arrives from the diffusion loss.

    The reasoner never processes a generator slot, so this is non-zero only
    because the two towers share an attention operator -- and it is the only
    route by which the generator's objective can move the vision-language model
    at all.
    """
    attribution = np.array(run.final["tower_attribution"])
    column = attribution[:, run.tower_of(TEXT), :]
    active = [TEXT, IMAGE, GENERATOR] if run.config.get("lm_loss", True) \
        else [GENERATOR]
    total = np.nansum(column[:, active], axis=1)
    if not np.any(total > 0):
        # A frozen reasoner receives no gradient at all, so the *share* of it
        # coming from anywhere is undefined rather than zero.
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total > 0, column[:, GENERATOR] / total, np.nan)
    return float(np.nanmean(share[total > 0]))


def reasoner_drift(run: CosmosRunLog) -> float:
    """How far the reasoner has moved from the pretrained vision-language model."""
    return float(np.mean(np.array(run.final["drift"])[:, run.tower_of(TEXT)]))


def reasoner_cost(run: CosmosRunLog) -> dict[str, float]:
    """Change in the reasoner's own losses since before a generator was attached.

    Positive is degradation: the vision-language model got worse at the job it
    was pretrained on while a diffusion tower was bolted to its attention.
    """
    final = run.final["reasoner"]
    out: dict[str, float] = {}
    for key in ("text_loss", "image_loss"):
        # The probe writes "text"/"image"; the pretraining baseline writes
        # "text_loss"/"image_loss". Accept either so old logs stay readable.
        probe_key = key if key in final else key.removesuffix("_loss")
        if probe_key in final and key in run.baseline:
            out[key] = final[probe_key] - run.baseline[key]
    return out


def conditioning(run: CosmosRunLog) -> dict[str, float]:
    """How much worse the generator gets when a modality is cut from its view."""
    ablation = run.final["ablation"]
    full = max(ablation["full"], 1e-9)
    return {"no_text": ablation["no_text"] / full,
            "no_image": ablation["no_image"] / full}


def summarize(run: CosmosRunLog) -> dict:
    quality = run.quality
    cost = reasoner_cost(run)
    ratios = conditioning(run)
    return {
        "n_towers": run.n_towers,
        "flow_loss": quality["flow_loss"],
        "cell_accuracy": quality["cell_accuracy"],
        "exact_frame": quality["exact_frame"],
        "no_text": ratios["no_text"],
        "no_image": ratios["no_image"],
        "d_text_loss": cost.get("text_loss", float("nan")),
        "d_image_loss": cost.get("image_loss", float("nan")),
        "gen_grad_share": diffusion_into_reasoner(run),
        "reasoner_drift": reasoner_drift(run),
        "params": run.meta["param_summary"]["non_embedding"],
        "wall_seconds": run.meta.get("wall_seconds", float("nan")),
    }


def fusion_profile(run: CosmosRunLog) -> np.ndarray:
    """Where in the stack the generator reads the reasoner. (L, 3, 3)."""
    return np.array(run.final["attention_mass"])
