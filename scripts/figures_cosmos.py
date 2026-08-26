"""Render the two-tower figure set from measured run logs.

Colour carries one meaning throughout and never carries it alone: the reasoner
tower blue, the generator tower vermillion -- the Okabe-Ito pair, which stays
distinguishable under the common colour-vision deficiencies.
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

from mot.cosmos_analysis import (
    GENERATOR, IMAGE, TEXT, CosmosRunLog, conditioning, diffusion_into_reasoner,
    fusion_profile, load_all_cosmos, reasoner_cost, summarize,
)

REASONER_C, GENERATOR_C = "#2a78d6", "#eb6834"
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

OBJECTIVE_LABELS = ("text\nCE", "image\nCE", "diffusion")


def _title(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=6)


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    print(f"  wrote {name}")


def fig_who_trains_whom(runs: dict[str, CosmosRunLog], out: Path) -> None:
    """Nearly half the reasoner's gradient comes from the other tower's loss."""
    names = [n for n in ("cosmos_two_tower", "cosmos_insulated", "cosmos_blocked")
             if n in runs]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.1),
                             sharey=True)
    fig.subplots_adjust(wspace=0.25, top=0.78)
    axes = np.atleast_1d(axes)
    for panel, (ax, name) in enumerate(zip(axes, names)):
        grid = np.nansum(np.array(runs[name].final["tower_attribution"]), axis=0)
        shown = np.nan_to_num(grid, nan=0.0)
        ax.imshow(shown, cmap=SEQ, aspect="auto", vmin=0,
                  vmax=max(shown.max(), 1e-9))
        for t in range(shown.shape[0]):
            for o in range(shown.shape[1]):
                ax.text(o, t, f"{shown[t, o]:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if shown[t, o] > shown.max() * 0.55 else INK)
        ax.set_xticks(range(3), OBJECTIVE_LABELS)
        if panel == 0:
            ax.set_yticks(range(shown.shape[0]),
                          ["reasoner", "generator"][:shown.shape[0]])
        ax.grid(False)
        _title(ax, name.replace("cosmos_", ""))
    fig.suptitle("Which objective trains which tower",
                 x=0.02, y=1.0, ha="left", fontsize=12, color=INK)
    fig.text(0.02, -0.03,
             "The reasoner never processes a generator slot, so its diffusion "
             "column exists only because the two towers share an attention "
             "operator.", fontsize=8.5, color=MUTED)
    _save(fig, out, "cosmos_fig1_who_trains_whom")


def fig_generation_quality(runs: dict[str, CosmosRunLog], out: Path) -> None:
    """The loss, and the thing the loss is a proxy for."""
    order = sorted(runs, key=lambda n: runs[n].quality["flow_loss"])
    labels = [n.replace("cosmos_", "") for n in order]
    flow = [runs[n].quality["flow_loss"] for n in order]
    exact = [runs[n].quality["exact_frame"] for n in order]

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    y = np.arange(len(order))
    left.barh(y, flow, color=GENERATOR_C)
    left.set_yticks(y, labels, fontsize=8)
    left.invert_yaxis()
    left.set_xlabel("held-out diffusion loss")
    _title(left, "Objective")

    right.barh(y, exact, color=REASONER_C)
    right.set_xlabel("frames reconstructed exactly")
    _title(right, "Task")

    fig.suptitle("A small diffusion loss and the right next frame are not the "
                 "same claim", x=0.02, ha="left", fontsize=12, color=INK)
    _save(fig, out, "cosmos_fig2_generation_quality")


def fig_conditioning(runs: dict[str, CosmosRunLog], out: Path) -> None:
    """What the generator is actually reading."""
    names = [n for n in sorted(runs) if n != "cosmos_frozen_reasoner"]
    ratios = [conditioning(runs[n]) for n in names]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    y = np.arange(len(names))
    ax.barh(y - 0.2, [r["no_text"] for r in ratios], 0.4,
            color=REASONER_C, label="rule cut from its view")
    ax.barh(y + 0.2, [r["no_image"] for r in ratios], 0.4,
            color=GENERATOR_C, label="board cut from its view")
    ax.axvline(1.0, color=INK, lw=1.2, ls="--", label="not using it at all")
    ax.set_yticks(y, [n.replace("cosmos_", "") for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("diffusion loss relative to the full context (log scale)")
    ax.legend(loc="lower right", fontsize=8)
    _title(ax, "The next frame needs the rule and the board, and the generator "
               "reads both")
    _save(fig, out, "cosmos_fig3_conditioning")


def fig_reasoner_cost(runs: dict[str, CosmosRunLog], out: Path) -> None:
    """What bolting a diffusion tower onto a vision-language model costs it."""
    names = [n for n in ("cosmos_two_tower", "cosmos_insulated",
                         "cosmos_frozen_reasoner", "cosmos_gen_only",
                         "cosmos_blocked", "cosmos_dense") if n in runs]
    if not names:
        return
    cost = [reasoner_cost(runs[n]).get("image_loss", np.nan) for n in names]
    share = [diffusion_into_reasoner(runs[n]) for n in names]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    colours = [GENERATOR_C if c > 0 else REASONER_C for c in cost]
    y = np.arange(len(names))
    ax.barh(y, cost, color=colours)
    for i, s in enumerate(share):
        if s == s:
            ax.annotate(f"{s:.0%} of its gradient is diffusion",
                        (cost[i], i), fontsize=7.5, color=SECOND,
                        xytext=(6 if cost[i] >= 0 else -6, 0),
                        textcoords="offset points",
                        ha="left" if cost[i] >= 0 else "right", va="center")
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(y, [n.replace("cosmos_", "") for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("change in the reasoner's image loss since pretraining "
                  "(negative is better)")
    _title(ax, "What the second objective costs the vision-language model")
    _save(fig, out, "cosmos_fig4_reasoner_cost")


def fig_fusion_depth(runs: dict[str, CosmosRunLog], out: Path) -> None:
    """Where in the stack the generator reads its conditioning."""
    run = runs.get("cosmos_two_tower")
    if run is None:
        return
    mass = fusion_profile(run)
    layers = np.arange(run.n_layers)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(layers, mass[:, GENERATOR, TEXT], color=REASONER_C, marker="s",
            label="generator reads the rule")
    ax.plot(layers, mass[:, GENERATOR, IMAGE], color=GENERATOR_C, marker="o",
            label="generator reads the board")
    ax.plot(layers, mass[:, GENERATOR, GENERATOR], color=MUTED, marker="^",
            label="generator reads itself")
    ax.set_xlabel("layer")
    ax.set_ylabel("attention probability mass")
    ax.legend(loc="best", fontsize=8)
    _title(ax, "Where the conditioning happens")
    _save(fig, out, "cosmos_fig5_fusion_depth")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs_cosmos")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    runs = load_all_cosmos(Path(args.runs))
    if not runs:
        print(f"no runs found in {args.runs}/ -- "
              f"try `python scripts/run_all.py --suite cosmos`")
        return 1

    out = Path(args.out)
    fig_who_trains_whom(runs, out)
    fig_generation_quality(runs, out)
    fig_conditioning(runs, out)
    fig_reasoner_cost(runs, out)
    fig_fusion_depth(runs, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
