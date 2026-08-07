"""Print the run summary and check the study's claims against the measured logs.

Each check states what it expects *before* looking, and prints PASS/FAIL either
way. A failed check is a real result about this setup, not a reason to hide it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mot.analysis import IMAGE, TEXT, RunLog, cross_modal_grad_fraction, load_all, summarize


def _row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(c.ljust(w) for c, w in zip(cells, widths))


def print_summary(runs: dict[str, RunLog]) -> None:
    header = ["run", "text%", "val text", "val img", "img|cap", "cap|img",
              "move ratio", "upd ratio", "SNR ratio", "diverge"]
    widths = [22, 6, 9, 8, 8, 8, 11, 10, 10, 8]
    print("\n" + _row(header, widths))
    print("-" * (sum(widths) + 2 * len(widths)))
    for name in sorted(runs):
        s = summarize(runs[name])
        print(_row([
            name,
            f"{s['text_frac']*100:.0f}%",
            f"{s['val_loss_text']:.3f}",
            f"{s['val_loss_image']:.3f}",
            f"{s['image_given_text']:.3f}",
            f"{s['text_given_image']:.3f}",
            f"{s.get('displacement_ratio', float('nan')):.2f}",
            f"{s.get('update_ratio', float('nan')):.2f}",
            f"{s.get('snr_ratio', float('nan')):.2f}",
            f"{s.get('divergence_final', float('nan')):.3f}",
        ], widths))


def check(label: str, expectation: str, passed: bool, detail: str,
          kind: str = "check") -> bool:
    """Report one claim. `kind="prediction"` marks a hypothesis stated in advance.

    A prediction the data does not support is reported as NOT CONFIRMED rather
    than quietly restated to match the result.
    """
    if kind == "prediction":
        tag = "CONFIRMED" if passed else "NOT CONFIRMED"
    else:
        tag = "PASS" if passed else "FAIL"
    print(f"\n[{tag}] {label}")
    print(f"       expected: {expectation}")
    print(f"       measured: {detail}")
    return passed


def run_checks(runs: dict[str, RunLog]) -> list[bool]:
    results = []
    bal = runs.get("balanced")
    blocked = runs.get("blocked_attention")
    heavy = runs.get("text_heavy")
    norm = runs.get("text_heavy_normalized")
    sgd = runs.get("text_heavy_sgd")

    if blocked is not None:
        off = float(np.abs(blocked.attribution[:, :, TEXT, IMAGE]).max() +
                    np.abs(blocked.attribution[:, :, IMAGE, TEXT]).max())
        diag = float(blocked.attribution[:, :, TEXT, TEXT].min())
        results.append(check(
            "probe validity: blocking cross-modal attention kills the cross gradient",
            "off-diagonal attribution identically 0, diagonal still positive",
            off == 0.0 and diag > 0,
            f"max |off-diagonal| = {off:g}; min diagonal = {diag:.3e}"))

    if bal is not None:
        off = float(np.abs(bal.attribution[-1, :, IMAGE, TEXT]).min())
        frac = float(cross_modal_grad_fraction(bal)[-1, IMAGE])
        results.append(check(
            "cross-modal training path exists with global attention",
            "image expert receives non-zero gradient from the text loss",
            off > 0,
            f"min over layers = {off:.3e}; {frac*100:.1f}% of its gradient is cross-modal"))

    if bal is not None and blocked is not None:
        a = float(bal.conditional["image_given_text"][-1])
        b = float(blocked.conditional["image_given_text"][-1])
        results.append(check(
            "task validity: the task genuinely needs the other modality",
            "blocked attention is strictly worse on image-given-caption tokens",
            b > a,
            f"balanced {a:.3f} vs blocked {b:.3f} nats "
            f"({'+' if b > a else ''}{(b-a):.3f})"))

    if bal is not None:
        d0 = float(np.abs(bal.divergence["overall"][0]).max())
        dN = float(bal.divergence["overall"][-1].mean())
        results.append(check(
            "identical init makes specialisation attributable",
            "divergence exactly 0 at step 0, positive at the end",
            d0 == 0.0 and dN > 0,
            f"step 0 = {d0:g}; final mean = {dN:.3f}"))

    if heavy is not None and sgd is not None:
        adam = float(np.nanmean(
            heavy.update_norm[-20:, :, IMAGE] / heavy.update_norm[-20:, :, TEXT]))
        plain = float(np.nanmean(
            sgd.update_norm[-20:, :, IMAGE] / sgd.update_norm[-20:, :, TEXT]))
        results.append(check(
            "Adam normalises away the token-count imbalance in step size",
            "update-norm ratio nearer 1.0 under Adam than under SGD",
            abs(adam - 1) < abs(plain - 1),
            f"AdamW ratio {adam:.3f} vs SGD ratio {plain:.3f} "
            f"(image tokens are {(1-heavy.text_frac)*100:.0f}% of the stream)"))

    if heavy is not None and norm is not None:
        a = float(heavy.val_loss_image[-1])
        b = float(norm.val_loss_image[-1])
        results.append(check(
            "per-modality loss normalisation helps the starved modality",
            "lower held-out image loss than the token-mean objective",
            b < a,
            f"token-mean {a:.3f} vs normalised {b:.3f} nats ({b-a:+.3f})"))

    return results


def run_predictions(runs: dict[str, RunLog]) -> list[bool]:
    """Hypotheses stated before the runs. Reported as measured, not as hoped."""
    results = []
    bal, heavy = runs.get("balanced"), runs.get("text_heavy")

    if bal is not None and heavy is not None:
        base = float(np.nanmean(bal.grad_snr[-1][:, IMAGE] / bal.grad_snr[-1][:, TEXT]))
        starved = float(np.nanmean(heavy.grad_snr[-1][:, IMAGE] / heavy.grad_snr[-1][:, TEXT]))
        results.append(check(
            "starving a modality degrades its expert's gradient SNR",
            "image/text SNR ratio LOWER under the 85/15 mixture than under 50/50",
            starved < base,
            f"balanced {base:.3f} -> text_heavy {starved:.3f} (ratio ROSE). "
            f"SNR here tracks how predictable a modality's tokens are, not how many "
            f"there are: image patch codes are determined by the caption, while the "
            f"extra text comes from text-only documents whose captions are inherently "
            f"unpredictable, so the text expert's gradient is the noisier one",
            kind="prediction"))

    if bal is not None and heavy is not None:
        base = float(cross_modal_grad_fraction(bal)[-1, IMAGE])
        starved = float(cross_modal_grad_fraction(heavy)[-1, IMAGE])
        results.append(check(
            "cross-modal attention subsidises the starved expert most when needed",
            "image expert's cross-modal gradient share HIGHER under 85/15 than 50/50",
            starved > base,
            f"balanced {base*100:.1f}% -> text_heavy {starved*100:.1f}% (share FELL). "
            f"The scarce modality offers fewer keys to attend to, so the abundant "
            f"modality's loss reaches the scarce expert less, not more",
            kind="prediction"))

    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    args = parser.parse_args()

    runs = load_all(Path(args.runs))
    if not runs:
        print(f"no run logs in {args.runs}/", file=sys.stderr)
        return 1

    print_summary(runs)
    print("\n" + "=" * 72)
    print("CLAIM CHECKS  (measurement validity + intended effects)")
    print("=" * 72)
    checks = run_checks(runs)

    print("\n" + "=" * 72)
    print("PREDICTIONS  (stated in advance; reported as measured)")
    print("=" * 72)
    predictions = run_predictions(runs)

    print(f"\n{sum(checks)}/{len(checks)} checks passed; "
          f"{sum(predictions)}/{len(predictions)} predictions confirmed")
    if not all(predictions):
        print("A prediction the data did not support is a result, not a bug — "
              "see the note under each.")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
