"""Train every run concurrently.

Torch scales poorly past a few threads on this workload -- the model is small
enough that per-op overhead dominates -- so six processes on a few threads each
finish far sooner than six sequential runs on all cores.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mot.configs import RUNS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs")
    parser.add_argument("--threads", type=int, default=1, help="threads per run")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    names = args.only or sorted(RUNS)
    log_dir = ROOT / "runs_logs"
    log_dir.mkdir(exist_ok=True)

    started = time.time()
    procs = []
    for name in names:
        env = {**os.environ,
               "OMP_NUM_THREADS": str(args.threads),
               "MKL_NUM_THREADS": str(args.threads),
               # Without this the per-process thread pools spin-wait against each
               # other and the machine sits mostly idle while every run crawls.
               "OMP_WAIT_POLICY": "PASSIVE",
               "PYTHONPATH": str(ROOT)}
        cmd = [sys.executable, str(ROOT / "scripts" / "run.py"), name,
               "--out", args.out, "--threads", str(args.threads)]
        if args.steps:
            cmd += ["--steps", str(args.steps)]
        handle = (log_dir / f"{name}.log").open("w", encoding="utf-8")
        procs.append((name, subprocess.Popen(cmd, cwd=ROOT, env=env,
                                             stdout=handle, stderr=subprocess.STDOUT),
                      handle))
        print(f"launched {name}", flush=True)

    failed = []
    for name, proc, handle in procs:
        code = proc.wait()
        handle.close()
        status = "ok" if code == 0 else f"FAILED ({code})"
        print(f"{name}: {status}  [{time.time()-started:.0f}s]", flush=True)
        if code != 0:
            failed.append(name)

    print(f"\nall runs finished in {(time.time()-started)/60:.1f} min", flush=True)
    if failed:
        print(f"failed runs: {failed}; see runs_logs/", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
