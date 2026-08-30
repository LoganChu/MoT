"""Paired statistics over the ablation runs.

Every arm is scored on the same questions in the same order, so the quantity of
interest is the per-question difference against baseline rather than the
difference of two means. That pairing removes the between-question variance,
which dominates: questions differ from each other far more than arms do.

Composite claims -- whether the three DeepStack depths are additive, whether
M-RoPE and DeepStack are redundant -- are formed per question and only then
averaged. Combining aggregate standard errors instead would require assuming the
arms are independent, and they are emphatically not: they share the questions,
the model and the images.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArmResult:
    arm: str
    metric: str
    scores: dict[str, float]        # qid -> score, so arms align by question
    seconds: float

    @property
    def mean(self) -> float:
        return sum(self.scores.values()) / len(self.scores)


def load_runs(directory: Path) -> dict[str, ArmResult]:
    out: dict[str, ArmResult] = {}
    for path in sorted(directory.glob("*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        out[blob["arm"]] = ArmResult(
            arm=blob["arm"], metric=blob["metric"],
            scores={r["qid"]: r["score"] for r in blob["per_question"]
                    if r["score"] is not None},
            seconds=blob.get("seconds", float("nan")))
    return out


def _summarise(diffs: list[float]) -> dict[str, float]:
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    se = sd / math.sqrt(n) if n else float("nan")
    moved = [d for d in diffs if d != 0]
    worse = sum(1 for d in moved if d < 0)
    return {"n": n, "delta": mean, "sd": sd, "se": se,
            "lo": mean - 1.96 * se, "hi": mean + 1.96 * se,
            "moved": len(moved), "worse": worse,
            "better": len(moved) - worse}


def paired(runs: dict[str, ArmResult], arm: str, baseline: str = "baseline"):
    """Per-question difference of ``arm`` against ``baseline``."""
    base, other = runs[baseline], runs[arm]
    qids = [q for q in base.scores if q in other.scores]
    return _summarise([other.scores[q] - base.scores[q] for q in qids])


def _diff_vector(runs, arm, baseline="baseline"):
    base, other = runs[baseline], runs[arm]
    return {q: other.scores[q] - base.scores[q]
            for q in base.scores if q in other.scores}


def combination(runs: dict[str, ArmResult], joint: str, parts: list[str]):
    """Per-question excess of a joint arm over the sum of its parts.

    Deltas are negative when an arm does damage, so a *negative* excess means the
    joint removal cost MORE than the individual removals predict. That is the
    signature of mutually redundant components: each one alone is dispensable
    because the others compensate for it, while removing all of them at once
    loses the capability outright. A positive excess is the reverse -- the parts
    overlap, so taking them together costs less than taking them separately.
    """
    missing = [a for a in [joint, *parts] if a not in runs]
    if missing:
        return None
    dj = _diff_vector(runs, joint)
    dp = [_diff_vector(runs, p) for p in parts]
    qids = [q for q in dj if all(q in d for d in dp)]
    excess = [dj[q] - sum(d[q] for d in dp) for q in qids]
    return _summarise(excess)


def required_n(sd: float, half_width: float = 0.02) -> int:
    """Questions needed for a 95% interval of the given half-width."""
    if sd <= 0:
        return 0
    return math.ceil((1.96 * sd / half_width) ** 2)


def contrast(runs: dict[str, ArmResult], a: str, b: str):
    """Per-question difference between two arms, neither of them the baseline.

    Formed per question for the same reason the composites are: the two arms saw
    identical inputs, so their scores are correlated and treating their standard
    errors as independent would overstate the uncertainty.
    """
    if a not in runs or b not in runs:
        return None
    xa, xb = runs[a].scores, runs[b].scores
    qids = [q for q in xa if q in xb]
    return _summarise([xa[q] - xb[q] for q in qids])
