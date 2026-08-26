"""Train one run: `python scripts/run.py balanced` or `... vlm_pairs`.

The name selects the suite: caption/image runs land in `runs/`, two-tower runs
in `runs_cosmos/`, because their log schemas differ.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mot.configs import RUNS
from mot.train import train
from mot.cosmos_configs import COSMOS_RUNS
from mot.cosmos_train import train as train_cosmos

ALL_RUNS = {**RUNS, **COSMOS_RUNS}


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

    is_cosmos = args.run in COSMOS_RUNS
    cfg = ALL_RUNS[args.run]
    out = Path(args.out or ("runs_cosmos" if is_cosmos else "runs"))
    if args.steps > 0:
        cfg = type(cfg)(**{**cfg.__dict__, "steps": args.steps})

    print(f"[{cfg.name}] {cfg.description}", flush=True)
    if is_cosmos:
        train_cosmos(cfg, out, cache_dir=Path("runs_cosmos/cache"))
    else:
        train(cfg, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
