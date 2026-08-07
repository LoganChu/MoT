# Measuring what actually trains in a Mixture-of-Transformers

[Mixture-of-Transformers](https://arxiv.org/abs/2411.04996) (Liang et al., TMLR 2025)
decouples every *non-embedding* parameter by modality — feed-forward networks, the
Q/K/V/O projections, QK-norms, and layer norms — while keeping a single global
self-attention over the full interleaved sequence. Tokens are routed to their
modality's expert deterministically; there is no learned router.

That design raises a question the architecture diagram does not answer: **if each
expert only ever processes its own modality's tokens, what stops the modality with
more data from being trained far harder than the other?**

This repo answers it by instrumenting a small MoT and measuring, rather than by
reasoning about it.

## What the measurements say

All numbers below are from `scripts/report.py` on the committed run logs.

**The starved expert is not under-trained in weight movement — under Adam.**
With image tokens at just **14%** of the stream, the image expert's weights end up
**1.04×** as far from initialisation as the text expert's (in the balanced run,
0.94×). Adam divides by a running second moment, so the size of the step it takes
is close to scale-free in the gradient. Swap in plain SGD and the same mixture
gives **0.69×** — there the starvation is plainly visible. *Token share and
training progress are simply not the same quantity under an adaptive optimiser.*

**But equal movement is not equal quality.** Held-out image loss still degrades
from **0.936** (balanced) to **1.134** (85/15). Weighting each modality's loss
equally rather than by token count recovers much of it, to **0.998**. So the
imbalance shows up in the objective, not in how far the weights travelled.

**The experts really are cross-trained by each other's loss.** In the balanced run
**19.6%** of the image expert's gradient arrives from the *text* loss (and 4.8% of
the text expert's from the image loss). Mask cross-modal attention and this becomes
*identically* zero, which is what proves the path is real.

**Fusion concentrates near the top of the stack.** Cross-modal attention rises with
depth — image queries spend 45% of their attention on text at layer 0 and 66% at
layer 5. Linear CKA between the two modalities' pooled representations tells a
matching story: the middle layers de-align (0.78 → 0.16) while the top two stay
aligned (0.93 → 0.86). Encode separately in the middle; share at the top.

**Specialisation is concentrated in the big matrices.** Final relative distance
between the two experts: `wq` 1.31, `wk` 1.28, `wo` 1.26, `wv` 1.22, `ffn` 1.17 —
but the layer norms barely differentiate at all (0.08–0.11). Divergence also grows
with depth (0.78 at layer 0 → 1.04 at layer 5).

### Two predictions the data did not support

Both were stated before the runs and are reported by `report.py` as NOT CONFIRMED:

- **"Starving a modality degrades its expert's gradient SNR."** It rose instead
  (1.08 → 1.28). SNR here tracks how *predictable* a modality's tokens are, not how
  many there are: image patch codes are determined by the caption, while the added
  text comes from text-only documents whose captions are inherently unpredictable.
- **"Cross-modal attention subsidises the starved expert most when it's needed
  most."** The subsidy *shrank*, 19.6% → 2.7%. A scarce modality offers fewer keys
  to attend to, so the abundant modality's loss reaches the scarce expert *less*.
  Global attention is not a self-correcting mechanism for token imbalance.

## Quick start

```bash
python -m pytest tests/ -q          # 12 correctness tests, ~5s
python scripts/run_all.py           # all six runs in parallel  (~25 min, CPU)
python scripts/figures.py           # static figure set -> figures/
python viz/build.py                 # interactive page -> viz/mot_dynamics.html
```

Single run: `python scripts/run.py balanced --steps 400`.

`run_all.py` launches one process per run with `OMP_NUM_THREADS=1` and
`OMP_WAIT_POLICY=PASSIVE`. This model is small enough to be dominated by per-op
overhead rather than FLOPs, so torch scales poorly across threads and the
per-process thread pools otherwise spin-wait against each other — six
single-threaded processes beat six sequential runs on all cores by a wide margin.

## The task

Each scene is a 3×3 board holding one to three coloured shapes, written two ways:

- **caption** — `<bot> red square top-left blue circle center <eot>`
- **patch codes** — one VQ-style code per board cell, `<boi> … <eoi>`

The two views are in **bijection** (asserted in `tests/`), so in a `caption → image`
document the image tokens are predictable *only* by attending back to the caption,
and vice versa. This is what makes cross-modal attention measurable rather than
assumed: mask it out and the loss on exactly those tokens must get worse.

The text/image token ratio is controlled by mixing in text-only documents —
the same lever real multimodal corpora pull. `solve_text_only_prob` inverts the
mixture analytically and the achieved ratio is measured and logged.

## Runs

| Run | What it isolates |
|---|---|
| `balanced` | 50/50 tokens, AdamW, token-mean loss — the reference |
| `text_heavy` | 85/15 mixture — does the starved image expert under-train? |
| `text_heavy_normalized` | 85/15 with each modality's loss weighted equally |
| `text_heavy_sgd` | 85/15 under SGD — isolates Adam's update normalisation |
| `blocked_attention` | cross-modal attention masked — the control |
| `dense` | both modalities share one transformer |

## What is measured

Every step: per-modality loss, and per expert per layer the gradient norm, the
update norm, and cumulative displacement `‖θ_t − θ_0‖ / ‖θ_0‖`.

Every 25 steps, on a **fixed** held-out probe batch (so variation over time is the
model changing, not the data):

| Probe | Question it answers |
|---|---|
| `grad_attribution` | `‖∂L_m′/∂θ_m‖` — who trains whom |
| `attention_mass` | where in the stack the modalities read each other |
| `expert_divergence` | how far apart the two experts have pulled, by sub-module |
| `layerwise_cka` | do the experts map their modality into a shared space |
| `GradientSNR` | is a starved expert taking smaller steps, or noisier ones |
| `conditional_losses` | loss on tokens only the *other* modality explains |

Probes run before the optimiser step and clear their gradients afterwards, so a
probe never contributes to training (`test_probe_leaves_no_gradient_behind`).

## Why the blocked-attention control matters

The headline claim — that a starved expert is subsidised through cross-modal
attention — rests entirely on the off-diagonal of `grad_attribution` being real.
It is validated two ways:

- **By construction.** `grad_attribution` scores each modality's loss only on
  positions where the input *and* the target are that modality. Boundary positions
  (a caption's `<eot>` predicting `<boi>`) are routed to one expert while their loss
  belongs to the other, and would leak gradient across the diagonal for reasons that
  have nothing to do with attention. They are excluded.
- **By test.** With cross-modal attention masked, the off-diagonal must be
  *identically* zero — not merely small — while the diagonal stays healthy.
  `test_blocked_attention_zeroes_off_diagonal` asserts exactly that.

## Layout

```
mot/vocab.py     union vocabulary; modality is a function of token id
mot/data.py      scene generation, bijection, document packing, ratio control
mot/model.py     MoT blocks, deterministic routing, global attention
mot/probes.py    the measurements above
mot/train.py     instrumented training loop
mot/analysis.py  load logs into arrays, derive reported quantities
mot/configs.py   the six runs
scripts/         run.py, run_all.py, figures.py
viz/             artifact.html template + build.py bundler
tests/           12 tests, correctness of routing, model, and probes
```

## Reading the results honestly

This is a ~1.3M-parameter model on synthetic data with a nine-token image.
**Magnitudes do not transfer to 7B.** What transfers is mechanism: that gradient
exposure scales with token count while Adam's step size does not; that global
attention creates a gradient path into every expert regardless of its token share;
that specialisation is layer-dependent. Those follow from the architecture and the
optimiser, not from this task.

Identical expert initialisation is a measurement choice, not standard practice — it
pins divergence to zero at step 0 so later divergence is attributable to training.
