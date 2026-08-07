"""Train one run: `python scripts/run.py balanced`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mot.configs import RUNS
from mot.train import train


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", choices=sorted(RUNS))
    parser.add_argument("--out", default="runs")
    parser.add_argument("--threads", type=int, default=0, help="0 = torch default")
    parser.add_argument("--steps", type=int, default=0, help="0 = config default")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    cfg = RUNS[args.run]
    if args.steps > 0:
        cfg = type(cfg)(**{**cfg.__dict__, "steps": args.steps})

    print(f"[{cfg.name}] {cfg.description}", flush=True)
    train(cfg, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
