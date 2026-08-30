"""Benchmark loading and official scoring for the ablation study.

Reads the parquet shards directly with pyarrow, so the study takes no
dependency on ``datasets``. Both metrics are the ones the benchmarks define,
because an ablation is only interesting against a number that others report:
ANLS for DocVQA and the ten-annotator agreement score for TextVQA.
"""

from __future__ import annotations

import io
import random
import re
from dataclasses import dataclass

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

DOCVQA = ("lmms-lab/DocVQA", ["DocVQA/validation-0000{}-of-00006.parquet".format(i)
                              for i in range(6)])
TEXTVQA = ("lmms-lab/textvqa", ["data/validation-0000{}-of-00003.parquet".format(i)
                                for i in range(3)])


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    answers: tuple[str, ...]
    image: Image.Image


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, golds, tau: float = 0.5) -> float:
    """DocVQA's metric: best normalised edit similarity, zeroed below tau."""
    p = " ".join(pred.lower().strip().split())
    best = 0.0
    for gold in golds:
        g = " ".join(str(gold).lower().strip().split())
        if not p and not g:
            best = 1.0
        elif p or g:
            best = max(best, 1 - _levenshtein(p, g) / max(len(p), len(g)))
    return best if best >= tau else 0.0


def vqa_accuracy(pred: str, golds) -> float:
    """TextVQA's metric: agreement with ten annotators, saturating at three."""
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", str(s).lower().strip())
    p = norm(pred)
    return min(sum(norm(g) == p for g in golds) / 3.0, 1.0)


BENCHMARKS = {
    "docvqa": dict(source=DOCVQA, metric=anls, metric_name="ANLS",
                   qid="questionId", answers="answers"),
    "textvqa": dict(source=TEXTVQA, metric=vqa_accuracy, metric_name="VQA-acc",
                    qid="question_id", answers="answers"),
}


def _shard_rows(repo: str, files: list[str]) -> list[tuple[str, int]]:
    """Every (file, row) pair in the split, without decoding any images."""
    pairs: list[tuple[str, int]] = []
    for fn in files:
        n = pq.ParquetFile(hf_hub_download(repo, fn, repo_type="dataset")).metadata.num_rows
        pairs.extend((fn, i) for i in range(n))
    return pairs


def load_questions(benchmark: str, n: int, seed: int = 0) -> list[Question]:
    """A deterministic subsample, nested across sizes.

    Two properties matter. Every arm must see the same questions in the same
    order, because the study compares arms per question and that pairing is what
    makes a few hundred questions enough. And the sample for a small ``n`` must
    be a prefix of the sample for a larger one, so a pilot run measures the
    variance of the very questions the full run will use and its scores stay
    comparable instead of being discarded.
    """
    spec = BENCHMARKS[benchmark]
    repo, files = spec["source"]
    pairs = _shard_rows(repo, files)
    order = random.Random(seed).sample(range(len(pairs)), k=len(pairs))
    chosen = [pairs[i] for i in order[:n]]

    wanted: dict[str, list[int]] = {}
    for fn, row in chosen:
        wanted.setdefault(fn, []).append(row)

    decoded: dict[tuple[str, int], Question] = {}
    for fn, rows in wanted.items():
        table = pq.ParquetFile(hf_hub_download(repo, fn, repo_type="dataset")).read()
        for row in rows:
            values = {k: table.column(k)[row].as_py()
                      for k in (spec["qid"], "question", spec["answers"], "image")}
            decoded[(fn, row)] = Question(
                qid=str(values[spec["qid"]]),
                question=values["question"],
                answers=tuple(values[spec["answers"]]),
                image=Image.open(io.BytesIO(values["image"]["bytes"])).convert("RGB"),
            )
    return [decoded[key] for key in chosen]     # permutation order, so nesting holds
