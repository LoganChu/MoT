"""Bundle measured run logs into the self-contained interactive page.

The published artifact runs under a CSP that blocks every external request, so
the data is inlined rather than fetched. Series are downsampled to roughly 120
points and rounded, which keeps the payload small without visibly changing any
curve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mot.analysis import RunLog, load_all, summarize
from mot.data import render_scene, sample_scene
from mot.vocab import COLORS, describe

TARGET_POINTS = 120

# Content colours for the toy images -- these render "a red square", so they are
# labels rather than chart series, and must actually look like the colour named.
SWATCHES = {
    "red": "#d1495b", "green": "#3f8f5c", "blue": "#3d6bb3", "yellow": "#d8a125",
    "purple": "#7b5ea7", "orange": "#d97a34", "cyan": "#2f97a8", "magenta": "#b8558f",
}


def _thin(array: np.ndarray, target: int = TARGET_POINTS) -> np.ndarray:
    if len(array) <= target:
        return array
    idx = np.unique(np.linspace(0, len(array) - 1, target).round().astype(int))
    return array[idx]


def _thin_index(length: int, target: int = TARGET_POINTS) -> np.ndarray:
    if length <= target:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, target).round().astype(int))


def _round(values, digits: int = 5):
    """Round for payload size; NaN becomes null so the page can skip those points."""
    return _nan_to_none(np.round(np.asarray(values, dtype=np.float64), digits).tolist())


def _nan_to_none(value):
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def pack_run(run: RunLog) -> dict:
    keep = _thin_index(len(run.step))
    probe_keep = _thin_index(len(run.probe_step))

    return {
        "name": run.name,
        "description": run.config.get("description", ""),
        "optimizer": run.config["optimizer"],
        "lossWeighting": run.config["loss_weighting"],
        "blocked": bool(run.config["blocked_attention"]),
        "dense": bool(run.config["dense"]),
        "textFrac": round(run.text_frac, 4),
        "nLayers": run.n_layers,
        "nExperts": run.n_experts,
        "steps": int(run.config["steps"]),
        "wallSeconds": run.meta.get("wall_seconds"),
        "params": run.meta["param_summary"],
        "step": run.step[keep].astype(int).tolist(),
        "lossText": _round(run.loss_text[keep], 4),
        "lossImage": _round(run.loss_image[keep], 4),
        "displacement": _round(run.displacement[keep], 5),
        "updateNorm": _round(run.update_norm[keep], 6),
        "gradNorm": _round(run.grad_norm[keep], 6),
        "probeStep": run.probe_step[probe_keep].astype(int).tolist(),
        "attribution": _round(run.attribution[probe_keep], 7),
        "attentionMass": _round(run.attention_mass[probe_keep], 5),
        "divergence": {k: _round(v[probe_keep], 5) for k, v in run.divergence.items()},
        "cka": _round(run.cka[probe_keep], 4),
        "valLossText": _round(run.val_loss_text[probe_keep], 4),
        "valLossImage": _round(run.val_loss_image[probe_keep], 4),
        "conditional": {k: _round(v[probe_keep], 4) for k, v in run.conditional.items()},
        "gradSnr": _round(run.grad_snr[probe_keep], 5),
        "summary": summarize(run),
    }


def build_examples(n: int = 3) -> list[dict]:
    """A few real scenes, so the page can show what the model is actually fitting."""
    rng = np.random.default_rng(7)
    out = []
    for _ in range(n):
        scene = sample_scene(rng)
        grid = render_scene(scene)
        out.append({
            "caption": [describe(t) for t in scene.caption_tokens()],
            "imageTokens": [describe(t) for t in scene.image_tokens()],
            "grid": [[None if c < 0 else COLORS[c] for c in row] for row in grid.tolist()],
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--template", default="viz/artifact.html")
    parser.add_argument("--out", default="viz/mot_dynamics.html")
    args = parser.parse_args()

    runs = load_all(ROOT / args.runs)
    if not runs:
        print(f"no run logs found in {args.runs}/", file=sys.stderr)
        return 1

    payload = {
        "runs": {name: pack_run(run) for name, run in runs.items()},
        "examples": build_examples(),
        "swatches": SWATCHES,
    }
    blob = json.dumps(payload, separators=(",", ":"))

    template = (ROOT / args.template).read_text(encoding="utf-8")
    if "__MOT_DATA__" not in template:
        print("template is missing the __MOT_DATA__ placeholder", file=sys.stderr)
        return 1
    out_path = ROOT / args.out
    out_path.write_text(template.replace("__MOT_DATA__", blob), encoding="utf-8")

    print(f"bundled {len(runs)} runs -> {out_path} "
          f"({out_path.stat().st_size/1024:.0f} KB, data {len(blob)/1024:.0f} KB)")
    for name, run in sorted(runs.items()):
        s = summarize(run)
        print(f"  {name:24s} text_frac={s['text_frac']:.3f}  "
              f"val text={s['val_loss_text']:.3f} image={s['val_loss_image']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
