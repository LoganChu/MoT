"""Print the two-tower run summary and check the study's claims against the logs.

Each check states what it expects *before* looking, and prints the verdict
either way. `control` checks are the ones the rest of the argument rests on; a
`prediction` the data does not support is reported as NOT CONFIRMED rather than
quietly rewritten to match.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mot.cosmos_analysis import (
    GENERATOR, IMAGE, TEXT, CosmosRunLog, conditioning, diffusion_into_reasoner,
    fusion_profile, load_all_cosmos, reasoner_cost, reasoner_drift, summarize,
)

HEADER = ["run", "tow", "flow", "cell", "exact", "no-txt", "no-img",
          "d text", "d image", "gen->rsn", "drift"]
WIDTHS = [24, 4, 8, 6, 7, 7, 7, 8, 8, 9, 7]


def _row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(c.ljust(w) for c, w in zip(cells, widths))


def print_summary(runs: dict[str, CosmosRunLog]) -> None:
    base = next(iter(runs.values())).baseline
    print("\nThe pretrained vision-language model, before a generator was attached:")
    print(f"  text loss {base['text_loss']:.4f}   image loss {base['image_loss']:.4f}")
    print("  (the caption determines the frame, so the image loss is a real "
          "cross-modal\n   capability and not a floor)")

    print("\n" + _row(HEADER, WIDTHS))
    print("-" * (sum(WIDTHS) + 2 * len(WIDTHS)))
    for name in sorted(runs):
        s = summarize(runs[name])
        print(_row([
            name.replace("cosmos_", ""),
            str(s["n_towers"]),
            f"{s['flow_loss']:.4f}",
            f"{s['cell_accuracy']:.3f}",
            f"{s['exact_frame']:.3f}",
            f"{s['no_text']:.2f}x",
            f"{s['no_image']:.2f}x",
            f"{s['d_text_loss']:+.3f}",
            f"{s['d_image_loss']:+.3f}",
            f"{s['gen_grad_share']:.3f}",
            f"{s['reasoner_drift']:.3f}",
        ], WIDTHS))
    print("\n  cell / exact: agreement with the true next frame, per cell and "
          "whole-frame.")
    print("  no-txt / no-img: diffusion loss with that modality cut from the "
          "generator's view,\n    relative to the full context. 1.00x means the "
          "generator was not using it.")
    print("  d text / d image: change in the reasoner's own losses since before "
          "a generator\n    existed. Positive is degradation.")
    print("  gen->rsn: share of the reasoner's gradient arriving from the "
          "diffusion objective,\n    which is non-zero only because the two "
          "towers share an attention operator.")


def check(label: str, expectation: str, passed: bool, detail: str,
          kind: str = "check") -> bool:
    tag = {"prediction": ("CONFIRMED", "NOT CONFIRMED"),
           "control": ("PASS", "CONTROL FAILED"),
           "check": ("PASS", "FAIL")}[kind]
    print(f"\n[{tag[0] if passed else tag[1]}] {label}")
    print(f"       expected: {expectation}")
    print(f"       measured: {detail}")
    return passed


def run_checks(runs: dict[str, CosmosRunLog]) -> list[bool]:
    results: list[bool] = []
    two = runs.get("cosmos_two_tower")
    dense = runs.get("cosmos_dense")
    three = runs.get("cosmos_three_tower")
    blocked = runs.get("cosmos_blocked")
    insulated = runs.get("cosmos_insulated")
    frozen = runs.get("cosmos_frozen_reasoner")
    gen_only = runs.get("cosmos_gen_only")
    no_upcycle = runs.get("cosmos_no_upcycle")

    # --- controls ---------------------------------------------------------
    if blocked is not None:
        attribution = np.array(blocked.final["tower_attribution"])
        stray = float(np.abs(attribution[:, blocked.tower_of(TEXT), GENERATOR]).max())
        results.append(check(
            "blocking the towers removes the whole cross-tower gradient path",
            "the diffusion objective contributes identically zero to the "
            "reasoner, not merely a little",
            stray == 0.0,
            f"largest cross-tower gradient norm {stray:.3e}",
            kind="control"))

        ratios = conditioning(blocked)
        results.append(check(
            "and makes the conditioning ablations no-ops",
            "cutting a modality costs nothing once no path to it exists",
            abs(ratios["no_text"] - 1.0) < 1e-4
            and abs(ratios["no_image"] - 1.0) < 1e-4,
            f"no-text {ratios['no_text']:.6f}x, no-image {ratios['no_image']:.6f}x",
            kind="control"))

    if insulated is not None:
        attribution = np.array(insulated.final["tower_attribution"])
        into = float(np.abs(attribution[:, insulated.tower_of(TEXT), GENERATOR]).max())
        own = float(attribution[:, insulated.tower_of(TEXT), TEXT].min())
        results.append(check(
            "insulation cuts only the diffusion gradient out of the reasoner",
            "the generator's objective contributes exactly zero to the reasoner "
            "while the reasoner still trains on its own",
            into == 0.0 and own > 0,
            f"diffusion->reasoner {into:.3e}; text->reasoner {own:.4f}",
            kind="control"))

    if frozen is not None:
        drift = reasoner_drift(frozen)
        results.append(check(
            "freezing the reasoner moves none of its weights",
            "reasoner drift is exactly zero at the end of joint training",
            drift == 0.0,
            f"drift {drift:.3e}",
            kind="control"))

    # --- what the towers do to each other ---------------------------------
    if two is not None and blocked is not None:
        results.append(check(
            "the generator cannot predict the future without the reasoner",
            "the next frame is determined by the rule and the board together, "
            "so a generator cut off from both must fall to guessing",
            blocked.quality["exact_frame"] < two.quality["exact_frame"],
            f"exact frames {blocked.quality['exact_frame']:.3f} blocked vs "
            f"{two.quality['exact_frame']:.3f} joined",
            kind="control"))

    if two is not None:
        ratios = conditioning(two)
        results.append(check(
            "the generator conditions on both halves of the reasoner",
            "cutting either the rule or the board from its view makes the "
            "diffusion loss worse, because the next frame needs both",
            ratios["no_text"] > 1.05 and ratios["no_image"] > 1.05,
            f"no-text {ratios['no_text']:.2f}x, no-image {ratios['no_image']:.2f}x",
            kind="prediction"))

        cost = reasoner_cost(two)
        results.append(check(
            "attaching a diffusion tower degrades the vision-language model",
            "the reasoner gets worse at the cross-modal job it was pretrained "
            "on, because a second objective now reaches it through the shared "
            "attention",
            cost["image_loss"] > 0,
            f"image loss {cost['image_loss']:+.4f}, text loss "
            f"{cost['text_loss']:+.4f}; the diffusion objective supplies "
            f"{diffusion_into_reasoner(two):.1%} of the reasoner's gradient",
            kind="prediction"))

    if two is not None and insulated is not None:
        results.append(check(
            "and insulating that path protects it",
            "cutting the diffusion gradient out of the reasoner -- and changing "
            "nothing else -- leaves the vision-language model less degraded",
            reasoner_cost(insulated)["image_loss"] < reasoner_cost(two)["image_loss"],
            f"image loss {reasoner_cost(insulated)['image_loss']:+.4f} insulated "
            f"vs {reasoner_cost(two)['image_loss']:+.4f} joined",
            kind="prediction"))

    if two is not None and gen_only is not None:
        share = diffusion_into_reasoner(gen_only)
        results.append(check(
            "with no objective of its own the reasoner is still trained, "
            "entirely across the attention",
            "the diffusion loss becomes the reasoner's only gradient, and the "
            "generator still learns something from a tower trained that way",
            share > 0.99 and gen_only.quality["cell_accuracy"] > 0.3,
            f"diffusion supplies {share:.1%} of the reasoner's gradient; "
            f"cell accuracy {gen_only.quality['cell_accuracy']:.3f}",
            kind="prediction"))

    if two is not None and frozen is not None:
        results.append(check(
            "the reasoner has to adapt, not merely be present",
            "a frozen reasoner conditions the generator worse than one allowed "
            "to move with it",
            frozen.quality["flow_loss"] > two.quality["flow_loss"],
            f"flow loss {frozen.quality['flow_loss']:.4f} frozen vs "
            f"{two.quality['flow_loss']:.4f} trainable",
            kind="prediction"))

    # --- what the tower split is worth ------------------------------------
    if two is not None and dense is not None:
        results.append(check(
            "decoupling the towers beats sharing one transformer",
            "an autoregressive objective and a diffusion objective want "
            "different weights, so giving them their own -- while keeping one "
            "attention -- should generate better",
            two.quality["flow_loss"] < dense.quality["flow_loss"],
            f"flow loss {two.quality['flow_loss']:.4f} two-tower vs "
            f"{dense.quality['flow_loss']:.4f} dense; exact frames "
            f"{two.quality['exact_frame']:.3f} vs {dense.quality['exact_frame']:.3f}",
            kind="prediction"))

    if two is not None and three is not None:
        results.append(check(
            "splitting the reasoner further does not pay for itself",
            "Cosmos 3 decouples per tower, not per modality inside a tower; a "
            "third tower for the image side should not beat two",
            three.quality["flow_loss"] >= two.quality["flow_loss"],
            f"flow loss {three.quality['flow_loss']:.4f} three-tower vs "
            f"{two.quality['flow_loss']:.4f} two-tower",
            kind="prediction"))

    if two is not None and no_upcycle is not None:
        results.append(check(
            "upcycling the generator from the vision-language model helps",
            "Cosmos 3 starts both towers from the same pretrained weights, so "
            "a generator that starts there should beat one starting from noise",
            two.quality["flow_loss"] < no_upcycle.quality["flow_loss"],
            f"flow loss {two.quality['flow_loss']:.4f} upcycled vs "
            f"{no_upcycle.quality['flow_loss']:.4f} from scratch",
            kind="prediction"))

    # --- which decoupling earns its parameters ----------------------------
    layouts = {n: runs[n] for n in (
        "cosmos_two_tower", "cosmos_attention_only", "cosmos_share_norms",
        "cosmos_ffn_only", "cosmos_taper", "cosmos_dense") if n in runs}
    if len(layouts) >= 4:
        ranked = sorted(layouts.items(), key=lambda kv: kv[1].quality["flow_loss"])
        best = ranked[0][0]
        results.append(check(
            "the attention path is what has to be decoupled between towers",
            "sharing the feed-forward network should cost little, and sharing "
            "any of the attention path should cost a lot -- the ordering the "
            "caption/image study's divergence numbers do *not* predict",
            best in ("cosmos_two_tower", "cosmos_attention_only"),
            "; ".join(f"{k.replace('cosmos_', '')} {v.quality['flow_loss']:.4f}"
                      for k, v in ranked),
            kind="prediction"))

    return results


def print_fusion(run: CosmosRunLog) -> None:
    print(f"\nwhere the generator reads the reasoner -- {run.name}")
    print("  layer   gen->text   gen->image   gen->gen")
    mass = fusion_profile(run)
    for layer in range(run.n_layers):
        print(f"  {layer:^5d}   {mass[layer, GENERATOR, TEXT]:>9.3f}   "
              f"{mass[layer, GENERATOR, IMAGE]:>10.3f}   "
              f"{mass[layer, GENERATOR, GENERATOR]:>8.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs_cosmos")
    args = parser.parse_args()

    runs = load_all_cosmos(Path(args.runs))
    if not runs:
        print(f"no runs found in {args.runs}/ -- "
              f"try `python scripts/run_all.py --suite cosmos`")
        return 1

    print_summary(runs)
    results = run_checks(runs)
    if "cosmos_two_tower" in runs:
        print_fusion(runs["cosmos_two_tower"])

    print(f"\n{sum(results)}/{len(results)} checks and predictions held.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
