"""Run the ablation matrix over a VQA benchmark.

Per-question scores are written, not just the mean: every arm sees the same
questions in the same order, so the comparison that matters is per question, and
the paired difference has far less variance than either score alone.

Each (benchmark, arm) result is a separate file and completed pairs are skipped,
so an interrupted sweep resumes instead of restarting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mot.ablate import ablated
from mot.ablate_configs import ABLATIONS
from mot.vqa import BENCHMARKS, load_questions

PROMPT_SUFFIX = "\nAnswer briefly using the image."


def evaluate(model, proc, questions, spec, metric, max_new_tokens: int):
    """Score every question, surviving the occasional page that will not fit.

    DocVQA pages vary a lot in size and the largest ones sit close to the card's
    limit, so a single failure must not end a multi-hour sweep. A question that
    runs out of memory is recorded with a null score and kept in place: the
    resume logic compares question ids positionally, and the analysis pairs arms
    by id, so a question missing from one arm simply drops out of that arm's
    comparison rather than biasing it toward zero.
    """
    records = []
    for q in questions:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": q.question + PROMPT_SUFFIX}]}]
        text = proc.apply_chat_template(messages, add_generation_prompt=True)
        model.model.rope_deltas = None          # never carry a cache between questions
        try:
            batch = proc(text=[text], images=[q.image], return_tensors="pt").to("cuda")
            prompt_len = batch["input_ids"].shape[1]
            with torch.no_grad(), ablated(model, spec):
                out = model.generate(**batch, max_new_tokens=max_new_tokens,
                                     do_sample=False)
            pred = proc.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
            records.append({"qid": q.qid, "pred": pred, "score": metric(pred, q.answers)})
        except torch.OutOfMemoryError:
            print(f"    OOM on {q.qid}, recording null", flush=True)
            records.append({"qid": q.qid, "pred": None, "score": None})
        # Page sizes vary by more than an order of magnitude, so the caching
        # allocator accumulates fragmented blocks it cannot reuse. Left alone,
        # reserved memory climbs past the card and the driver silently pages
        # over PCIe, which costs a factor of two and makes identical work take
        # anywhere from 14 to 32 seconds. Releasing the cache each question
        # holds reserved memory at ~4.4 GB and changes nothing measured.
        torch.cuda.empty_cache()
    return [r["score"] for r in records if r["score"] is not None], records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--arms", nargs="*", default=sorted(ABLATIONS))
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", default="runs_ablate")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    args = ap.parse_args()

    out = Path(args.out) / args.benchmark
    out.mkdir(parents=True, exist_ok=True)
    def existing(arm: str) -> dict | None:
        path = out / f"{arm}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    todo = [a for a in args.arms
            if (prev := existing(a)) is None or prev["n"] < args.n]
    if not todo:
        print(f"{args.benchmark}: all {len(args.arms)} arms already cover n={args.n}")
        return 0
    print(f"{args.benchmark}: {len(todo)} arms to run "
          f"({len(args.arms) - len(todo)} already at n>={args.n})", flush=True)

    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16).cuda().eval()

    spec_bench = BENCHMARKS[args.benchmark]
    questions = load_questions(args.benchmark, args.n)
    print(f"  {len(questions)} questions, metric {spec_bench['metric_name']}", flush=True)

    for name in todo:
        spec = ABLATIONS[name]
        t0 = time.time()
        # Sampling is nested, so a smaller earlier run evaluated exactly this
        # prefix; reuse it and only score the questions it did not reach.
        prev, done = existing(name), []
        if prev is not None:
            done = prev["per_question"]
            assert [r["qid"] for r in done] == [q.qid for q in questions[:len(done)]], (
                f"{name}: stored questions do not prefix the current sample")
            print(f"  {name:<22} extending from n={len(done)}", flush=True)
        fresh_scores, fresh = evaluate(model, proc, questions[len(done):], spec,
                                       spec_bench["metric"], args.max_new_tokens)
        records = done + fresh
        scores = [r["score"] for r in records if r["score"] is not None]
        mean = sum(scores) / len(scores) if scores else float("nan")
        skipped = sum(1 for r in records if r["score"] is None)
        payload = {
            "benchmark": args.benchmark, "arm": name, "model": args.model,
            "reused": len(done),
            "description": spec.description, "n": len(scores),
            "metric": spec_bench["metric_name"], "mean": mean,
            "seconds": time.time() - t0, "per_question": records,
        }
        (out / f"{name}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"  {name:<22} {spec_bench['metric_name']} {mean:.4f}   "
              f"[{(time.time() - t0)/60:.1f} min]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
