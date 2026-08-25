"""Render the VLM figure set from measured run logs.

Colour carries one meaning throughout and never carries it alone: the text
expert blue, the image expert vermillion -- the Okabe-Ito pair, which stays
distinguishable under the common colour-vision deficiencies. Every multi-series
panel is also labelled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mot.vlm_analysis import (
    IMAGE, TEXT, VLMRunLog, forgetting, icl_gain, load_all_vlm,
    text_expert_drift,
)

TEXT_C, IMAGE_C = "#2a78d6", "#eb6834"
INK, SECOND, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list(
    "blues", ["#e8f1fd", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"])

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9, "axes.labelcolor": SECOND, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 130, "lines.linewidth": 2.0,
})

CHANCE = {"statement_acc": 0.20, "assoc0_acc": 1 / 6, "assoc4_acc": 1 / 6}


def _title(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=6)


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    print(f"  wrote {name}")


def _series(run: VLMRunLog, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Capability against a running step index across all stages."""
    values, marks, offset, last_stage = [], [], 0, None
    step = 0
    for probe in run.probes:
        if last_stage is not None and probe["stage"] != last_stage:
            offset = step
            marks.append(offset)
        last_stage = probe["stage"]
        step = offset + probe["step"]
        values.append((step, probe["text"][key]))
    xs, ys = zip(*values)
    return np.array(xs), np.array(ys)


def fig_forgetting(runs: dict[str, VLMRunLog], out: Path) -> None:
    """The headline: what each data regime costs the language model."""
    names = [n for n in ("vlm_pairs", "vlm_interleaved", "vlm_blend",
                         "vlm_pairs_sft_blend") if n in runs]
    if not names:
        return
    keys = ("statement_acc", "assoc4_acc")
    labels = ("fact recall", "four-shot accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))

    for ax, key, label in zip(axes, keys, labels):
        for colour, name in zip(palette, names):
            xs, ys = _series(runs[name], key)
            ax.plot(xs, ys, color=colour, label=name.replace("vlm_", ""))
        baseline = runs[names[0]].baseline[key]
        ax.axhline(baseline, color=INK, lw=1.2, ls="-",
                   label="after text pretraining")
        ax.axhline(CHANCE[key], color=MUTED, lw=1, ls="--", label="chance")
        ax.set_xlabel("step (align, then pretrain, then instruction tuning)")
        ax.set_ylabel("accuracy")
        _title(ax, label)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("What multimodal training costs the language model underneath",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.text(0.02, -0.06,
             "Same images, same captions, same cross-modal dependency in every "
             "arm. The only difference is how much of the text is prose rather "
             "than a caption.", fontsize=8.5, color=MUTED)
    _save(fig, out, "vlm_fig1_forgetting")


def fig_who_trains_whom(runs: dict[str, VLMRunLog], out: Path) -> None:
    """The image loss reaches the text expert only because attention is global."""
    names = [n for n in ("vlm_pairs", "vlm_insulated", "vlm_blocked")
             if n in runs]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.0))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        run = runs[name]
        grid = np.array(run.final["grad_attribution"]).sum(axis=0)     # (E, M)
        ax.imshow(grid, cmap=SEQ, aspect="auto", vmin=0, vmax=max(grid.max(), 1e-9))
        for e in range(grid.shape[0]):
            for m in range(grid.shape[1]):
                ax.text(m, e, f"{grid[e, m]:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if grid[e, m] > grid.max() * 0.55 else INK)
        ax.set_xticks([TEXT, IMAGE], ["text\nloss", "image\nloss"])
        ax.set_yticks(range(grid.shape[0]),
                      ["text\nexpert", "image\nexpert"][:grid.shape[0]])
        ax.grid(False)
        _title(ax, name.replace("vlm_", ""))
    fig.suptitle("Which loss trains which expert",
                 x=0.02, ha="left", fontsize=12, color=INK)
    _save(fig, out, "vlm_fig2_who_trains_whom")


def fig_drift(runs: dict[str, VLMRunLog], out: Path) -> None:
    """How far the language model gets dragged, and by what."""
    names = [n for n in ("vlm_pairs", "vlm_dense", "vlm_insulated",
                         "vlm_interleaved", "vlm_frozen_lm") if n in runs]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    palette = plt.cm.plasma(np.linspace(0.1, 0.8, len(names)))
    for colour, name in zip(palette, names):
        run = runs[name]
        xs, offset, last, step = [], 0, None, 0
        ys = []
        for probe in run.probes:
            if last is not None and probe["stage"] != last:
                offset = step
            last = probe["stage"]
            step = offset + probe["step"]
            xs.append(step)
            ys.append(float(np.mean(np.array(probe["drift"])[:, 0])))
        ax.plot(xs, ys, color=colour, label=name.replace("vlm_", ""))
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\|\theta_t-\theta_{\rm pretrained}\| \, / \, \|\theta_{\rm pretrained}\|$")
    ax.legend(loc="best", fontsize=8)
    _title(ax, "Drift of the text expert from the pretrained language model")
    _save(fig, out, "vlm_fig3_drift")


def fig_decoupling(runs: dict[str, VLMRunLog], out: Path) -> None:
    """Which decoupling earns its parameters."""
    names = [n for n in ("vlm_pairs", "vlm_share_norms", "vlm_attention_only",
                         "vlm_ffn_only", "vlm_taper", "vlm_dense", "vlm_muon",
                         "vlm_upcycled") if n in runs]
    if len(names) < 2:
        return
    image = [runs[n].final["visual"].get("image", np.nan) for n in names]
    kept = [forgetting(runs[n])["statement_acc"] for n in names]
    params = [runs[n].meta["param_summary"]["non_embedding"] / 1e6 for n in names]

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.4, 3.6))
    order = np.argsort(image)
    labels = [names[i].replace("vlm_", "") for i in order]
    left.barh(range(len(order)), [image[i] for i in order], color=IMAGE_C)
    left.set_yticks(range(len(order)), labels, fontsize=8)
    left.set_xlabel("held-out image loss")
    left.invert_yaxis()
    _title(left, "What the image side achieves")

    right.scatter([params[i] for i in order], [kept[i] for i in order],
                  color=TEXT_C, s=44, zorder=3)
    for i in order:
        right.annotate(names[i].replace("vlm_", ""),
                       (params[i], kept[i]), fontsize=7, color=SECOND,
                       xytext=(4, 4), textcoords="offset points")
    right.axhline(0, color=MUTED, lw=1, ls="--")
    right.set_xlabel("non-embedding parameters (millions)")
    right.set_ylabel("change in fact recall")
    _title(right, "What it costs the language model")

    fig.suptitle("Which decoupling earns its parameters",
                 x=0.02, ha="left", fontsize=12, color=INK)
    _save(fig, out, "vlm_fig4_decoupling")


def fig_in_context(runs: dict[str, VLMRunLog], out: Path) -> None:
    """Zero-shot survives what in-context learning does not."""
    names = [n for n in ("vlm_pairs", "vlm_frozen_lm", "vlm_interleaved")
             if n in runs]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.35
    positions = np.arange(len(names) + 1)
    zero = [runs[names[0]].baseline["assoc0_acc"]] + \
           [runs[n].final["text"]["assoc0_acc"] for n in names]
    four = [runs[names[0]].baseline["assoc4_acc"]] + \
           [runs[n].final["text"]["assoc4_acc"] for n in names]
    ax.bar(positions - width / 2, zero, width, color=MUTED, label="zero-shot")
    ax.bar(positions + width / 2, four, width, color=TEXT_C, label="four-shot")
    ax.axhline(CHANCE["assoc4_acc"], color=INK, lw=1, ls="--", label="chance")
    ax.set_xticks(positions,
                  ["pretrained\nLM"] + [n.replace("vlm_", "") for n in names],
                  fontsize=8)
    ax.set_ylabel("accuracy")
    ax.legend(loc="best", fontsize=8)
    _title(ax, "The demonstrations are only worth anything if the gap is positive")
    fig.suptitle("In-context learning is the capability that goes first",
                 x=0.02, ha="left", fontsize=12, color=INK)
    _save(fig, out, "vlm_fig5_in_context")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs_vlm")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    runs = load_all_vlm(Path(args.runs))
    if not runs:
        print(f"no runs found in {args.runs}/ -- "
              f"try `python scripts/run_all.py --suite vlm`")
        return 1

    out = Path(args.out)
    fig_forgetting(runs, out)
    fig_who_trains_whom(runs, out)
    fig_drift(runs, out)
    fig_decoupling(runs, out)
    fig_in_context(runs, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
