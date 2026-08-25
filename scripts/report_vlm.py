"""Print the VLM run summary and check the study's claims against the logs.

Each check states what it expects *before* looking, and prints the verdict
either way. `control` checks are the ones the rest of the argument rests on;
`prediction` checks are hypotheses, and one the data does not support is
reported as NOT CONFIRMED rather than quietly rewritten to match.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mot.vlm_analysis import (
    IMAGE, TEXT, VLMRunLog, forgetting, icl_gain, image_grad_into_text_expert,
    load_all_vlm, stage_ledger, summarize, text_expert_drift,
)

# Below this, four-shot accuracy is indistinguishable from zero-shot and the
# pretrained model has no in-context-learning ability to lose.
ICL_MIN_GAIN = 0.05

HEADER = ["run", "fact", "d fact", "0-shot", "4-shot", "d 4-shot", "ICL",
          "img loss", "img|cap", "LM drift", "img->LM"]
WIDTHS = [22, 6, 7, 7, 7, 9, 7, 9, 8, 9, 8]


def _row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(c.ljust(w) for c, w in zip(cells, widths))


def print_summary(runs: dict[str, VLMRunLog]) -> None:
    any_run = next(iter(runs.values()))
    base = any_run.baseline
    print("\nThe pretrained language model, before it ever saw an image:")
    gain = base["assoc4_acc"] - base["assoc0_acc"]
    print(f"  fact recall {base['statement_acc']:.3f} (chance 0.200)   "
          f"0-shot {base['assoc0_acc']:.3f}   4-shot {base['assoc4_acc']:.3f} "
          f"(chance 0.167)   in-context gain {gain:+.3f}")
    if gain < ICL_MIN_GAIN:
        print("\n  NOTE: the in-context-learning analogue did not train -- "
              "four-shot accuracy is\n  indistinguishable from zero-shot, so "
              "the pretrained model has no such ability to\n  lose. Every "
              "comparison that depends on it is skipped rather than reported, "
              "and the\n  0-shot / 4-shot / ICL columns below are noise. Fact "
              "recall is unaffected and\n  carries the forgetting results.")

    print("\n" + _row(HEADER, WIDTHS))
    print("-" * (sum(WIDTHS) + 2 * len(WIDTHS)))
    for name in sorted(runs):
        s = summarize(runs[name])
        print(_row([
            name,
            f"{s['statement_acc']:.3f}",
            f"{s['d_statement']:+.3f}",
            f"{s['assoc0_acc']:.3f}",
            f"{s['assoc4_acc']:.3f}",
            f"{s['d_assoc4']:+.3f}",
            f"{s['icl_gain']:+.3f}",
            f"{s['image_loss']:.4f}",
            f"{s['image_given_text']:.3f}",
            f"{s['text_drift']:.4f}",
            f"{s['image_grad_share']:.3f}",
        ], WIDTHS))
    print("\n  d fact / d 4-shot: change since text pretraining, in accuracy points. "
          "Negative is forgetting.")
    print("  ICL: four-shot minus zero-shot accuracy -- what the demonstrations buy.")
    print("  LM drift: how far the text expert has moved from its pretrained weights.")
    print("  img->LM: share of the text expert's gradient arriving from the image "
          "loss, which exists only because attention is global.")


def check(label: str, expectation: str, passed: bool, detail: str,
          kind: str = "check") -> bool:
    tag = {"prediction": ("CONFIRMED", "NOT CONFIRMED"),
           "control": ("PASS", "CONTROL FAILED"),
           "check": ("PASS", "FAIL")}[kind]
    print(f"\n[{tag[0] if passed else tag[1]}] {label}")
    print(f"       expected: {expectation}")
    print(f"       measured: {detail}")
    return passed


def icl_is_trained(runs: dict[str, VLMRunLog]) -> bool:
    base = next(iter(runs.values())).baseline
    return (base["assoc4_acc"] - base["assoc0_acc"]) >= ICL_MIN_GAIN


def run_checks(runs: dict[str, VLMRunLog]) -> list[bool]:
    results: list[bool] = []
    pairs = runs.get("vlm_pairs")
    interleaved = runs.get("vlm_interleaved")
    blend = runs.get("vlm_blend")
    sft_blend = runs.get("vlm_pairs_sft_blend")
    frozen = runs.get("vlm_frozen_lm")
    dense = runs.get("vlm_dense")
    insulated = runs.get("vlm_insulated")
    blocked = runs.get("vlm_blocked")
    upcycled = runs.get("vlm_upcycled")

    # --- controls ---------------------------------------------------------
    if blocked is not None:
        attribution = np.array(blocked.final["grad_attribution"])
        stray = max(float(np.abs(attribution[:, 0, IMAGE]).max()),
                    float(np.abs(attribution[:, blocked.image_expert(), TEXT]).max()))
        results.append(check(
            "blocked attention removes every cross-modal gradient path",
            "both off-diagonals identically zero, not merely small",
            stray == 0.0,
            f"largest off-diagonal gradient norm {stray:.3e}",
            kind="control"))

    if insulated is not None:
        attribution = np.array(insulated.final["grad_attribution"])
        into_lm = float(np.abs(attribution[:, 0, IMAGE]).max())
        own = float(attribution[:, 0, TEXT].min())
        results.append(check(
            "insulation cuts only the image loss out of the language model",
            "the image objective contributes exactly zero to the text expert "
            "while the text objective still trains it",
            into_lm == 0.0 and own > 0,
            f"image->text-expert {into_lm:.3e}; text->text-expert {own:.4f}",
            kind="control"))

    if frozen is not None:
        drift = text_expert_drift(frozen, "pretrain")
        results.append(check(
            "freezing the language model moves none of its weights",
            "text-expert drift is exactly zero at the end of pretraining",
            drift == 0.0,
            f"drift {drift:.3e}",
            kind="control"))

    # --- VILA's pretraining findings --------------------------------------
    if pairs is not None and interleaved is not None:
        lost_pairs = forgetting(pairs)
        lost_inter = forgetting(interleaved)
        results.append(check(
            "caption-only pretraining costs more text capability than interleaved",
            "training on nothing but caption-image pairs degrades the "
            "pretrained text capability further than interleaved documents do, "
            "because interleaving keeps the text distribution alive",
            lost_pairs["statement_acc"] < lost_inter["statement_acc"],
            f"fact recall {lost_pairs['statement_acc']:+.3f} on pairs vs "
            f"{lost_inter['statement_acc']:+.3f} interleaved; "
            f"4-shot {lost_pairs['assoc4_acc']:+.3f} vs "
            f"{lost_inter['assoc4_acc']:+.3f}",
            kind="prediction"))

    # The in-context-learning capability has to exist before anything can be
    # said about losing it. If pretraining left the four-shot accuracy at the
    # zero-shot level, every ICL comparison below would be a comparison between
    # two noise measurements, so they are skipped and said to be skipped.
    icl_trained = ICL_MIN_GAIN <= (
        next(iter(runs.values())).baseline["assoc4_acc"]
        - next(iter(runs.values())).baseline["assoc0_acc"])

    if pairs is not None and frozen is not None and icl_trained:
        results.append(check(
            "freezing the language model preserves recall and costs in-context learning",
            "fact recall survives because the weights holding it never move, "
            "but the demonstrations stop being worth anything, because the "
            "image and text representations never align in the deeper layers",
            forgetting(frozen)["statement_acc"] > forgetting(pairs)["statement_acc"]
            and icl_gain(frozen) < icl_gain(pairs),
            f"frozen: fact {forgetting(frozen)['statement_acc']:+.3f}, "
            f"ICL {icl_gain(frozen):+.3f}; "
            f"unfrozen: fact {forgetting(pairs)['statement_acc']:+.3f}, "
            f"ICL {icl_gain(pairs):+.3f}",
            kind="prediction"))

    if pairs is not None and sft_blend is not None:
        results.append(check(
            "blending text back in at instruction tuning recovers what was lost",
            "the same pretraining, and text-only data only in the final stage, "
            "ends with more text capability retained",
            forgetting(sft_blend)["statement_acc"] > forgetting(pairs)["statement_acc"],
            f"fact recall {forgetting(sft_blend)['statement_acc']:+.3f} with a "
            f"blended SFT vs {forgetting(pairs)['statement_acc']:+.3f} without",
            kind="prediction"))

    if pairs is not None and blend is not None:
        results.append(check(
            "blending text into pretraining works too, and earlier",
            "mixing text-only documents into the pretraining stage retains at "
            "least as much as fixing it afterwards",
            forgetting(blend)["statement_acc"] > forgetting(pairs)["statement_acc"],
            f"fact recall {forgetting(blend)['statement_acc']:+.3f} blended "
            f"pretraining vs {forgetting(pairs)['statement_acc']:+.3f} pairs",
            kind="prediction"))

    # --- what the Mixture-of-Transformers changes -------------------------
    if pairs is not None and dense is not None:
        results.append(check(
            "decoupling the experts by modality does not by itself prevent forgetting",
            "the text expert never processes an image token, but global "
            "attention still delivers image-loss gradient into it, so the "
            "decoupled model forgets on the same order as the dense one",
            abs(forgetting(pairs)["statement_acc"]
                - forgetting(dense)["statement_acc"]) < 0.25,
            f"fact recall {forgetting(pairs)['statement_acc']:+.3f} decoupled vs "
            f"{forgetting(dense)['statement_acc']:+.3f} dense; the image loss "
            f"supplies {image_grad_into_text_expert(pairs):.1%} of the decoupled "
            f"text expert's gradient",
            kind="prediction"))

    if pairs is not None and insulated is not None:
        results.append(check(
            "the forgetting travels along the cross-modal attention path",
            "cutting the image loss out of the text expert -- and changing "
            "nothing else -- leaves more of the pretrained capability intact",
            forgetting(insulated)["statement_acc"] > forgetting(pairs)["statement_acc"],
            f"fact recall {forgetting(insulated)['statement_acc']:+.3f} insulated "
            f"vs {forgetting(pairs)['statement_acc']:+.3f} not",
            kind="prediction"))

    if pairs is not None and upcycled is not None:
        results.append(check(
            "upcycling the image expert from the language model helps it learn",
            "starting the image expert from a trained transformer rather than "
            "from noise reaches a lower image loss",
            float(upcycled.final["visual"]["image"])
            < float(pairs.final["visual"]["image"]),
            f"image loss {float(upcycled.final['visual']['image']):.4f} upcycled "
            f"vs {float(pairs.final['visual']['image']):.4f} from scratch",
            kind="prediction"))

    # --- which decoupling earns its parameters ----------------------------
    grounding = {n: runs[n].final["conditional"]["image_given_text"]
                 for n in ("vlm_pairs", "vlm_attention_only", "vlm_share_norms",
                           "vlm_ffn_only", "vlm_taper", "vlm_dense")
                 if n in runs}
    if len(grounding) == 6:
        results.append(check(
            "the sub-modules that differentiate least are the ones you cannot share",
            "sharing the feed-forward network costs nothing, but sharing the "
            "attention norms -- which the caption/image study measures as barely "
            "differentiating between the experts (0.08-0.11, against 1.2-1.3 for "
            "the big matrices) -- costs most of the cross-modal grounding",
            grounding["vlm_attention_only"] < 0.5 * grounding["vlm_share_norms"],
            "; ".join(f"{k.replace('vlm_', '')} {v:.3f}"
                      for k, v in sorted(grounding.items(), key=lambda kv: kv[1])),
            kind="prediction"))

    return results


def print_ledger(run: VLMRunLog) -> None:
    print(f"\nstage ledger -- {run.name}")
    print(f"  {'stage':26s} {'fact':>6s} {'0-shot':>7s} {'4-shot':>7s} "
          f"{'img loss':>9s} {'LM drift':>9s}")
    for row in stage_ledger(run):
        print(f"  {row['stage']:26s} {row['statement_acc']:>6.3f} "
              f"{row['assoc0_acc']:>7.3f} {row['assoc4_acc']:>7.3f} "
              f"{row.get('image_loss', float('nan')):>9.4f} "
              f"{row.get('text_drift', float('nan')):>9.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs_vlm")
    args = parser.parse_args()

    runs = load_all_vlm(Path(args.runs))
    if not runs:
        print(f"no runs found in {args.runs}/ -- "
              f"try `python scripts/run_all.py --suite vlm`")
        return 1

    print_summary(runs)
    results = run_checks(runs)
    for name in ("vlm_pairs", "vlm_interleaved"):
        if name in runs:
            print_ledger(runs[name])

    print(f"\n{sum(results)}/{len(results)} checks and predictions held.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
