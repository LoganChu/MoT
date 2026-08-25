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
from mot.vlm_configs import VLM_RUNS

SUITES = {"base": sorted(RUNS), "vlm": sorted(VLM_RUNS),
          "all": sorted(RUNS) + sorted(VLM_RUNS)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=sorted(SUITES), default="all")
    parser.add_argument("--out", default=None,
                        help="default: runs/ for the caption study, runs_vlm/ for VLM")
    parser.add_argument("--threads", type=int, default=1, help="threads per run")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    names = args.only or SUITES[args.suite]

    # Every VLM arm starts from the same pretrained language model. Build it
    # once here rather than letting nine processes race to train the same
    # checkpoint -- and so that the arms are exactly comparable.
    vlm_names = [n for n in names if n in VLM_RUNS]
    if vlm_names:
        from mot.vlm_train import ensure_text_base

        print("building the shared text base (cached after the first time)",
              flush=True)
        ensure_text_base(VLM_RUNS[vlm_names[0]], ROOT / "runs_vlm" / "cache")
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
        out = args.out or ("runs_vlm" if name in VLM_RUNS else "runs")
        cmd = [sys.executable, str(ROOT / "scripts" / "run.py"), name,
               "--out", out, "--threads", str(args.threads)]
        if args.steps and name not in VLM_RUNS:
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
