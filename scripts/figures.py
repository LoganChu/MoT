"""Render the static figure set from measured run logs.

Colours follow one rule throughout: text expert = blue, image expert = orange.
The pair is validated for colour-vision deficiency on both surfaces, and every
multi-series panel carries a legend, so identity is never colour alone.
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

from mot.analysis import (
    IMAGE, TEXT, RunLog, cross_modal_attention_fraction, cross_modal_grad_fraction,
    expert_balance_ratio, load_all,
)

TEXT_C, IMAGE_C = "#2a78d6", "#eb6834"
INK, SECOND, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list(
    "blues", ["#e8f1fd", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"])

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


def _title(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="600",
                 pad=24 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5,
                color=SECOND, va="bottom")


def _smooth(values: np.ndarray, window: int = 15) -> np.ndarray:
    """Centred rolling mean, NaN-safe. Raw data is always drawn behind it."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full_like(values, np.nan)
    half = window // 2
    for i in range(len(values)):
        chunk = values[max(0, i - half):i + half + 1]
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk):
            out[i] = chunk.mean()
    return out


def _noisy_line(ax, x, y, color, label) -> None:
    """Per-step ratios bounce hard; show the trend without hiding the variance."""
    ax.plot(x, y, color=color, lw=0.8, alpha=0.25)
    ax.plot(x, _smooth(y), color=color, lw=2.0, label=label)


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


def fig_how_trained(runs: dict[str, RunLog], out: Path) -> None:
    """The headline thermometer: cumulative weight movement per expert."""
    names = [n for n in ("balanced", "text_heavy") if n in runs]
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(4.1 * (len(names) + 1), 3.3))

    for ax, name in zip(axes, names):
        run = runs[name]
        disp = run.displacement.mean(axis=1)
        ax.plot(run.step, disp[:, TEXT], color=TEXT_C, label="text expert")
        ax.plot(run.step, disp[:, IMAGE], color=IMAGE_C, label="image expert")
        share = 100 * (1 - run.text_frac)
        _title(ax, name.replace("_", " "),
               f"image modality = {share:.0f}% of tokens")
        ax.set_xlabel("training step")
        ax.set_ylabel(r"$\|\theta_t-\theta_0\|\,/\,\|\theta_0\|$")
        ax.legend(loc="lower right")

    ax = axes[-1]
    for name in names:
        run = runs[name]
        _noisy_line(ax, run.step, expert_balance_ratio(run, "displacement"),
                    TEXT_C if name == "balanced" else IMAGE_C, name.replace("_", " "))
    ax.axhline(1.0, color=MUTED, lw=0.9)
    ax.text(runs[names[0]].step[-1], 1.0, " equal", color=MUTED, fontsize=8, va="center")
    _title(ax, "balance ratio", "image displacement / text displacement")
    ax.set_xlabel("training step")
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, out, "fig1_how_trained")


def fig_who_trains_whom(runs: dict[str, RunLog], out: Path) -> None:
    """Row-normalised gradient attribution: where each expert's gradient comes from."""
    panels = [n for n in ("balanced", "text_heavy", "blocked_attention") if n in runs]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(3.5 * (len(panels) + 1), 3.4))

    for ax, name in zip(axes, panels):
        run = runs[name]
        # Normalise within each layer, then average the shares, so every layer
        # counts equally. This matches `cross_modal_grad_fraction` exactly.
        per_layer = run.attribution[-1]                     # (L, E, M)
        totals = per_layer.sum(axis=2, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            shares = np.where(totals > 0, per_layer / totals, np.nan)
        rows = np.nanmean(shares, axis=0)                   # (E, M)
        im = ax.imshow(rows, cmap=SEQ, vmin=0, vmax=1)
        for i in range(rows.shape[0]):
            for j in range(rows.shape[1]):
                ax.text(j, i, f"{rows[i, j]*100:.0f}%", ha="center", va="center",
                        fontsize=10, color="#ffffff" if rows[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ["text loss", "image loss"])
        ax.set_yticks([0, 1], ["text\nexpert", "image\nexpert"])
        ax.grid(False)
        _title(ax, name.replace("_", " "), "share of each expert's gradient")

    ax = axes[-1]
    for name in panels:
        run = runs[name]
        frac = cross_modal_grad_fraction(run)
        ax.plot(run.probe_step, 100 * frac[:, IMAGE],
                label=name.replace("_", " "),
                color={"balanced": TEXT_C, "text_heavy": IMAGE_C,
                       "blocked_attention": MUTED}.get(name, MUTED))
    _title(ax, "image expert's cross-modal share",
           "% of its gradient arriving from the text loss")
    ax.set_xlabel("training step")
    ax.set_ylabel("%")
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, out, "fig2_who_trains_whom")


def fig_where_fusion(runs: dict[str, RunLog], out: Path) -> None:
    """Cross-modal attention mass by layer -- where the two streams actually meet."""
    run = runs.get("balanced")
    if run is None:
        return
    frac = cross_modal_attention_fraction(run)              # (P, L, M)
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.3))

    for ax, m, colour, label in ((axes[0], TEXT, TEXT_C, "text queries"),
                                 (axes[1], IMAGE, IMAGE_C, "image queries")):
        im = ax.imshow(frac[:, :, m].T, aspect="auto", origin="lower", cmap=SEQ,
                       extent=[0, run.probe_step[-1], -0.5, run.n_layers - 0.5],
                       vmin=0, vmax=float(np.nanmax(frac)))
        ax.set_yticks(range(run.n_layers))
        ax.set_xlabel("training step")
        ax.set_ylabel("layer")
        ax.grid(False)
        _title(ax, label, "fraction of attention spent on the other modality")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[2]
    final = frac[-1]                                        # (L, M)
    layers = np.arange(run.n_layers)
    ax.plot(layers, 100 * final[:, TEXT], color=TEXT_C, marker="o",
            markersize=5, label="text -> image")
    ax.plot(layers, 100 * final[:, IMAGE], color=IMAGE_C, marker="o",
            markersize=5, label="image -> text")
    _title(ax, "at end of training", "cross-modal attention by depth")
    ax.set_xlabel("layer")
    ax.set_ylabel("%")
    ax.set_xticks(layers)
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, out, "fig3_where_fusion")


def fig_specialization(runs: dict[str, RunLog], out: Path) -> None:
    """Divergence between the two experts, which starts at exactly zero."""
    run = runs.get("balanced")
    if run is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.3))

    ax = axes[0]
    shades = SEQ(np.linspace(0.25, 0.95, run.n_layers))
    for layer in range(run.n_layers):
        ax.plot(run.probe_step, run.divergence["overall"][:, layer],
                color=shades[layer], label=f"layer {layer}")
    _title(ax, "experts diverge from a shared start",
           "identical init, so every difference is training")
    ax.set_xlabel("training step")
    ax.set_ylabel("relative L2 distance")
    ax.legend(loc="best", ncol=2, fontsize=7.5)

    ax = axes[1]
    ax.plot(np.arange(run.n_layers), run.divergence["overall"][-1],
            color=TEXT_C, marker="o", markersize=5)
    _title(ax, "specialisation by depth", "final divergence per layer")
    ax.set_xlabel("layer")
    ax.set_ylabel("relative L2 distance")
    ax.set_xticks(range(run.n_layers))

    ax = axes[2]
    parts = [k for k in run.divergence if k != "overall"]
    values = [float(run.divergence[k][-1].mean()) for k in parts]
    order = np.argsort(values)[::-1]
    ax.barh([parts[i] for i in order], [values[i] for i in order],
            color=TEXT_C, height=0.62)
    ax.invert_yaxis()
    ax.grid(False, axis="y")
    _title(ax, "which components specialise", "mean over layers, end of training")
    ax.set_xlabel("relative L2 distance")
    fig.tight_layout()
    _save(fig, out, "fig4_specialization")


def fig_balance(runs: dict[str, RunLog], out: Path) -> None:
    """Adam hides token imbalance in step size; it resurfaces as gradient SNR."""
    pairs = [(n, runs[n]) for n in ("text_heavy", "text_heavy_sgd") if n in runs]
    if not pairs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.3))

    for ax, field, title, subtitle in (
        (axes[0], "grad_norm", "gradient norm ratio", "image expert / text expert"),
        (axes[1], "update_norm", "update norm ratio", "the step actually taken"),
    ):
        for name, run in pairs:
            _noisy_line(ax, run.step, expert_balance_ratio(run, field),
                        TEXT_C if "sgd" not in name else IMAGE_C,
                        run.config["optimizer"])
        ax.axhline(1.0, color=MUTED, lw=0.9)
        _title(ax, title, subtitle)
        ax.set_xlabel("training step")
        ax.legend(loc="best")

    ax = axes[2]
    for name, run in pairs:
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = run.grad_snr
            ratio = np.nanmean(snr[:, :, IMAGE] / snr[:, :, TEXT], axis=1)
        _noisy_line(ax, run.probe_step, ratio,
                    TEXT_C if "sgd" not in name else IMAGE_C, run.config["optimizer"])
    ax.axhline(1.0, color=MUTED, lw=0.9)
    _title(ax, "gradient SNR ratio", "signal-to-noise, image / text")
    ax.set_xlabel("training step")
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, out, "fig5_balance")


def fig_losses(runs: dict[str, RunLog], out: Path) -> None:
    """What the interventions actually cost or buy, on held-out data."""
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.3))
    order = [n for n in ("balanced", "text_heavy", "text_heavy_normalized",
                         "text_heavy_sgd", "blocked_attention", "dense") if n in runs]

    for ax, (field, label) in zip(axes[:2], (("val_loss_text", "text"),
                                             ("val_loss_image", "image"))):
        shades = SEQ(np.linspace(0.25, 0.95, len(order)))
        for i, name in enumerate(order):
            run = runs[name]
            ax.plot(run.probe_step, getattr(run, field), color=shades[i],
                    label=name.replace("_", " "))
        _title(ax, f"held-out {label} loss", "lower is better")
        ax.set_xlabel("training step")
        ax.set_ylabel("cross-entropy (nats)")
        ax.legend(loc="best", fontsize=7.5)

    ax = axes[2]
    labels = ["image | caption", "caption | image"]
    keys = ["image_given_text", "text_given_image"]
    compare = [n for n in ("balanced", "blocked_attention") if n in runs]
    width = 0.36
    for i, name in enumerate(compare):
        run = runs[name]
        values = [float(run.conditional[k][-1]) for k in keys]
        ax.bar(np.arange(2) + (i - 0.5) * width, values, width * 0.94,
               color=TEXT_C if name == "balanced" else MUTED,
               label=name.replace("_", " "))
    ax.set_xticks(range(2), labels)
    ax.grid(False, axis="x")
    _title(ax, "the cross-modal test",
           "loss on tokens only the other modality explains")
    ax.set_ylabel("cross-entropy (nats)")
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, out, "fig6_losses")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    runs = load_all(Path(args.runs))
    if not runs:
        print(f"no run logs in {args.runs}/", file=sys.stderr)
        return 1
    print(f"loaded {len(runs)} runs: {', '.join(sorted(runs))}")

    out = Path(args.out)
    fig_how_trained(runs, out)
    fig_who_trains_whom(runs, out)
    fig_where_fusion(runs, out)
    fig_specialization(runs, out)
    fig_balance(runs, out)
    fig_losses(runs, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
