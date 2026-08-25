"""Load staged VLM logs, and derive the quantities the write-up argues from."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TEXT, IMAGE = 0, 1
CAPABILITIES = ("statement_acc", "assoc0_acc", "assoc4_acc")


@dataclass
class VLMRunLog:
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
    def n_experts(self) -> int:
        return int(self.meta["n_experts"])

    @property
    def is_dense(self) -> bool:
        return self.n_experts < 2

    def text_expert(self) -> int:
        return 0

    def image_expert(self) -> int:
        return 0 if self.is_dense else 1

    def stage_names(self) -> list[str]:
        seen: list[str] = []
        for p in self.probes:
            if p["stage"] not in seen:
                seen.append(p["stage"])
        return seen

    def at_end_of(self, stage: str) -> dict:
        """The boundary probe for a stage: the last one recorded inside it."""
        matching = [p for p in self.probes if p["stage"] == stage]
        if not matching:
            raise KeyError(stage)
        return matching[-1]

    @property
    def final(self) -> dict:
        return self.probes[-1]


def load_vlm_run(path: Path) -> VLMRunLog:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return VLMRunLog(
        name=raw["config"]["name"],
        config=raw["config"],
        meta=raw["meta"],
        baseline=raw["baseline"],
        steps=raw["steps"],
        probes=raw["probes"],
    )


def load_all_vlm(directory: Path) -> dict[str, VLMRunLog]:
    return {p.stem: load_vlm_run(p) for p in sorted(Path(directory).glob("*.json"))}


def forgetting(run: VLMRunLog, stage: str | None = None) -> dict[str, float]:
    """Change in each text capability since the language model was pretrained.

    Negative is forgetting, in accuracy points. Reported as a difference rather
    than the relative drop VILA quotes, because a capability sitting near chance
    has no meaningful relative drop.
    """
    probe = run.final if stage is None else run.at_end_of(stage)
    return {k: probe["text"][k] - run.baseline[k] for k in CAPABILITIES}


def icl_gain(run: VLMRunLog, stage: str | None = None) -> float:
    """Four-shot accuracy minus zero-shot: what the demonstrations are worth."""
    probe = run.final if stage is None else run.at_end_of(stage)
    return probe["text"]["assoc4_acc"] - probe["text"]["assoc0_acc"]


def baseline_icl_gain(run: VLMRunLog) -> float:
    return run.baseline["assoc4_acc"] - run.baseline["assoc0_acc"]


def image_grad_into_text_expert(run: VLMRunLog, stage: str | None = None) -> float:
    """Share of the text expert's gradient that arrives from the image loss.

    In a Mixture-of-Transformers the text expert never processes an image token,
    so this is non-zero only because attention is global -- and it is the only
    route by which the image objective can move the language model at all.
    """
    probe = run.final if stage is None else run.at_end_of(stage)
    attribution = np.array(probe["grad_attribution"])          # (L, E, M)
    column = attribution[:, run.text_expert(), :]
    total = column.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmean(np.where(total > 0, column[:, IMAGE] / total, np.nan)))


def text_expert_drift(run: VLMRunLog, stage: str | None = None) -> float:
    """How far the language model has been dragged from its pretrained weights."""
    probe = run.final if stage is None else run.at_end_of(stage)
    return float(np.mean(np.array(probe["drift"])[:, run.text_expert()]))


def summarize(run: VLMRunLog) -> dict:
    final, delta = run.final, forgetting(run)
    out = {
        "statement_acc": final["text"]["statement_acc"],
        "assoc0_acc": final["text"]["assoc0_acc"],
        "assoc4_acc": final["text"]["assoc4_acc"],
        "d_statement": delta["statement_acc"],
        "d_assoc4": delta["assoc4_acc"],
        "icl_gain": icl_gain(run),
        "icl_gain_baseline": baseline_icl_gain(run),
        "image_loss": final["visual"].get("image", float("nan")),
        "text_loss": final["visual"].get("text", float("nan")),
        "image_given_text": final["conditional"].get("image_given_text", float("nan")),
        "text_drift": text_expert_drift(run),
        "image_grad_share": image_grad_into_text_expert(run),
        "wall_seconds": run.meta.get("wall_seconds", float("nan")),
    }
    return out


def stage_ledger(run: VLMRunLog) -> list[dict]:
    """Capability at every stage boundary, so a cost can be attributed to a stage."""
    rows = [{"stage": "text-pretrain (baseline)", **run.baseline}]
    for stage in run.stage_names():
        probe = run.at_end_of(stage)
        rows.append({"stage": stage, **probe["text"],
                     "image_loss": probe["visual"].get("image", float("nan")),
                     "text_drift": text_expert_drift(run, stage)})
    return rows
