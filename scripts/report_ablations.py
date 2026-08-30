"""Report the ablation study: paired deltas, composites, and the n each needs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mot.ablate_analysis import (combination, contrast, load_runs, paired,
                                 required_n)

GROUPS = [("M-RoPE", ["mrope_1d", "mrope_swap_hw", "mrope_shuffle"]),
          ("DeepStack", ["ds_off_0", "ds_off_1", "ds_off_2", "ds_off_all", "ds_shuffle"]),
          ("Both", ["mrope_1d_ds_off_all"])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_ablate")
    ap.add_argument("--benchmark", default="textvqa")
    ap.add_argument("--target", type=float, default=0.02,
                    help="95%% interval half-width you want on a delta")
    args = ap.parse_args()

    directory = Path(args.runs) / args.benchmark
    runs = load_runs(directory)
    if "baseline" not in runs:
        print(f"no baseline in {directory}/ -- nothing to pair against")
        return 1

    base = runs["baseline"]
    print(f"\n{args.benchmark}  |  metric {base.metric}  |  "
          f"baseline {base.mean:.4f} on {len(base.scores)} questions\n")
    print(f"  {'arm':<22} {'score':>7} {'delta':>8} {'95% CI':>18} "
          f"{'moved':>7} {'worse':>6} {'sd(d)':>7}")
    print("  " + "-" * 80)

    sds = []
    for title, arms in GROUPS:
        present = [a for a in arms if a in runs]
        if not present:
            continue
        print(f"  {title}")
        for arm in present:
            s = paired(runs, arm)
            sds.append(s["sd"])
            flag = "" if s["lo"] < 0 < s["hi"] else "  *"
            print(f"    {arm:<20} {runs[arm].mean:>7.4f} {s['delta']:>+8.4f} "
                  f"[{s['lo']:>+7.4f},{s['hi']:>+7.4f}] "
                  f"{s['moved']:>7} {s['worse']:>6} {s['sd']:>7.3f}{flag}")
    print("\n  * = 95% interval excludes zero")

    print("\n  Composite tests (formed per question, so arm correlations are kept)")
    for label, joint, parts in [
        ("DeepStack additivity: off_all vs sum of depths",
         "ds_off_all", ["ds_off_0", "ds_off_1", "ds_off_2"]),
        ("Redundancy: both vs M-RoPE + DeepStack separately",
         "mrope_1d_ds_off_all", ["mrope_1d", "ds_off_all"])]:
        c = combination(runs, joint, parts)
        if c is None:
            print(f"    {label}: arms missing")
            continue
        verdict = ("joint costs MORE than parts -> mutually redundant" if c["hi"] < 0
                   else "joint costs LESS than parts -> overlapping" if c["lo"] > 0
                   else "consistent with additive")
        print(f"    {label}")
        print(f"      excess {c['delta']:>+.4f}  "
              f"[{c['lo']:>+.4f},{c['hi']:>+.4f}]   {verdict}")

    print("\n  Direct contrasts between arms")
    for label, a, b in [
            ("wrong axes vs no axes      swap_hw - 1d", "mrope_swap_hw", "mrope_1d"),
            ("lost layout vs range shift shuffle - 1d", "mrope_shuffle", "mrope_1d"),
            ("alignment vs presence      shuffle - off_all", "ds_shuffle", "ds_off_all")]:
        c = contrast(runs, a, b)
        if c is None:
            print(f"    {label}: arms missing")
            continue
        verdict = "resolved" if not (c["lo"] < 0 < c["hi"]) else "not resolved at this n"
        print(f"    {label:<44} {c['delta']:>+.4f} "
              f"[{c['lo']:>+.4f},{c['hi']:>+.4f}]  {verdict}")

    if sds:
        worst = max(sds)
        print(f"\n  Sizing: largest observed sd(d) = {worst:.3f}")
        for hw in (0.05, args.target, 0.01):
            print(f"    +/-{hw:<5} on a single delta        -> n = "
                  f"{required_n(worst, hw):>6}")
        print(f"    +/-{args.target:<5} on a composite (~2x sd)   -> n = "
              f"{required_n(2 * worst, args.target):>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
