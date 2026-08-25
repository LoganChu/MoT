"""Train one run: `python scripts/run.py balanced` or `... vlm_pairs`.

The name selects the suite: caption/image runs land in `runs/`, staged VLM runs
in `runs_vlm/`, because their log schemas differ.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mot.configs import RUNS
from mot.train import train
from mot.vlm_configs import VLM_RUNS
from mot.vlm_train import train as train_vlm

ALL_RUNS = {**RUNS, **VLM_RUNS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", choices=sorted(ALL_RUNS))
    parser.add_argument("--out", default=None,
                        help="default: runs/ for the caption study, runs_vlm/ for VLM")
    parser.add_argument("--threads", type=int, default=0, help="0 = torch default")
    parser.add_argument("--steps", type=int, default=0, help="0 = config default")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    is_vlm = args.run in VLM_RUNS
    cfg = ALL_RUNS[args.run]
    out = Path(args.out or ("runs_vlm" if is_vlm else "runs"))
    if args.steps > 0 and not is_vlm:
        cfg = type(cfg)(**{**cfg.__dict__, "steps": args.steps})

    print(f"[{cfg.name}] {cfg.description}", flush=True)
    if is_vlm:
        train_vlm(cfg, out, cache_dir=Path("runs_vlm/cache"))
    else:
        train(cfg, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
