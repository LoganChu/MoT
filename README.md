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
reasoning about it. **Part I** does that for a text-and-image model trained from
scratch. **Part II** does it for the way a VLM is actually built — starting from
a pretrained language model and putting it at risk, following the recipe behind
NVIDIA's Cosmos-Nemotron/VILA family — and tests which parts of MoT's decoupling
earn their parameters.

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
python -m pytest tests/ -q            # 31 correctness tests, ~10s
python scripts/run_all.py             # every run in parallel (~70 min, CPU)
python scripts/report.py              # the caption/image numbers
python scripts/report_vlm.py          # the VLM-recipe numbers
python scripts/figures.py             # caption/image figures -> figures/
python scripts/figures_vlm.py         # VLM figures          -> figures/
python viz/build.py                   # interactive page -> viz/mot_dynamics.html
```

Single run: `python scripts/run.py balanced --steps 400`. One suite at a time:
`python scripts/run_all.py --suite base` or `--suite vlm`. Caption/image logs
land in `runs/`, VLM-recipe logs in `runs_vlm/`. The VLM suite trains its shared
language model once and caches it in `runs_vlm/cache/`; `run_all.py` builds it
before fanning out so the arms all start from the same weights.

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

---

# Part II — training a VLM the way NVIDIA's are trained

Part I asks what a Mixture-of-Transformers does with two modalities it learns
from scratch. No real VLM is built that way. A real one starts from a language
model and then risks it: NVIDIA's Cosmos-Nemotron family (formerly VILA) trains
in three stages on top of a pretrained LLM, and its central pre-training results
are all statements about what the multimodal stages *cost* the language model
underneath.

Three of them, from [VILA](https://arxiv.org/abs/2312.07533):

- Pretraining on caption-image pairs degrades text-only accuracy (MMLU) by
  **17.2%**. Interleaved image-text documents cost about **5%** — same images,
  same captions, but the text distribution survives.
- Freezing the language model during pretraining preserves zero-shot accuracy
  and destroys in-context learning: **72.1% → 58.1%** at four shots.
- Blending text-only data into instruction tuning recovers it, **40.7% → 51.4%**,
  and improves the *visual* scores at the same time.

Every one of those is a comparison across a stage boundary, and every one is a
capability the model had before and has less of afterwards. So this part needs
two things Part I does not have: a language model with something to lose, and a
training loop built out of stages.

It also asks a question that only arises for a Mixture-of-Transformers. The text
expert never processes an image token, so the naive expectation is that
decoupling protects the language model. It does not have to. Global attention
still delivers image-loss gradient into the text expert — and that path is
exactly the off-diagonal of `grad_attribution` that Part I measures. Whether it
is enough to do the damage is the point of `vlm_dense` and `vlm_insulated`.

## What the measurements say

All numbers from `scripts/report_vlm.py` on the committed logs. The pretrained
language model recalls its facts perfectly — **1.000** against a chance of
0.200 — so every figure below is a subtraction from a clean starting point.

**Interleaving preserves the language model; caption pairs destroy it.** Fact
recall after the full recipe: **0.941** interleaved against **0.219** on pairs,
a loss of 0.059 against 0.781. VILA reports 5% against 17.2% on MMLU; the
direction reproduces and the toy magnifies it. Same images, same captions, same
cross-modal dependency — the only difference is whether the model keeps seeing
prose.

**The stage ledger says exactly where it goes.** On pairs, fact recall is still
**1.000** after alignment, with text-expert drift of exactly 0.000 because the
language model is frozen — and **0.223** after pretraining. Every point is lost
in the one stage where an unfrozen language model meets caption-only data. On
interleaved it survives pretraining at **1.000** and loses its 0.059 only in the
caption-only instruction-tuning stage at the end.

**Both mitigations work, and earlier is better.** Blending text into pretraining
costs **0.039**; blending it only into instruction tuning recovers 0.219 → 0.359
after the damage is already done.

**Decoupling by modality does not protect the language model.** This is the
question the architecture invites, and the answer is no: **−0.781** for the
Mixture-of-Transformers against **−0.777** dense. The text expert never
processes a single image token and forgets just as much.

### Two predictions the data did not support

- **"The forgetting travels along the cross-modal attention path."** It does
  not. `vlm_insulated` cuts the image objective out of the text expert
  *exactly* — the probe reads 0.000 while the text objective still trains it at
  0.0735 — and forgetting got marginally **worse**, −0.828 against −0.781. The
  image loss only ever supplied 9.2% of that expert's gradient. What overwrites
  the language model is the text expert being retrained on caption text, not
  the image objective reaching it through attention. The path Part I measured is
  real, and it is not the culprit here.
- **"Upcycling the image expert from the language model helps it learn."**
  It hurt, and substantially: image loss **0.911** against **0.726** from
  scratch. Identical *random* initialisation is fine — Part I relies on it — but
  starting the image expert as a copy of a *trained* text expert drops it into a
  committed solution it never leaves.

### Which decoupling earns its parameters

Scored by how well the caption explains the image (`image_given_text`, lower is
better), the arms fall into two tight clusters with nothing in between:

| | grounding | image loss | what it decouples |
|---|---|---|---|
| full MoT | **0.343** | 0.726 | everything |
| attention only | **0.345** | 0.723 | attention, shares the whole FFN |
| share the norms | 0.727 | 0.911 | the big matrices and the FFN |
| FFN only | 0.727 | 0.914 | what a mixture-of-experts decouples |
| depth taper | 0.729 | 0.912 | bottom four layers |
| dense | 0.725 | 0.909 | nothing |

Sharing the feed-forward network — the largest block of per-expert parameters —
costs **nothing**. Sharing any part of the attention path costs half the
grounding, and the two arms are separated by the attention *norms* alone:
`attention_only` decouples them and succeeds, `share_norms` decouples the big
matrices instead and fails.

That is the reverse of what Part I's divergence numbers predict. There the layer
norms barely differentiate at all (0.08–0.11) while `wq` and `wk` pull far apart
(1.31, 1.28) — which reads as "the norms don't need decoupling". They are the
ones you cannot share. **How far two experts drift apart does not tell you which
of them had to be separate.** A cheap measurement that looked like it answered
the parameter-budget question turns out to answer a different one.

Muon is the outlier worth noting: the best image loss of any arm (**0.593**
against 0.726) with the largest drift away from the pretrained language model
(0.383 against 0.233) — it learns the new modality hardest and holds the old one
least.

## The language model

Stage 0 trains on text alone, and the corpus is built so that the model comes
out with two capabilities that degrade differently and can each be scored as an
accuracy rather than only as a loss:

- **`<bot> <stmt> c1 c2 REL <eot>`** — the relation is a fixed function of the
  two colours, drawn once from a hidden 8×8 table. Pure memorisation from
  weights: the MMLU analogue, and the thing that can be forgotten. Chance is
  0.20.
- **`<bot> <assoc> c s  c s  <query> c s <eot>`** — the colour-to-shape mapping
  is resampled for *every document*, so no amount of memorisation helps and the
  only way to answer is to attend to the demonstrations. Accuracy at zero shots
  against four is the in-context-learning analogue. Chance is 0.167.

The two are deliberately distinguishable from their first tokens: a statement
puts two colours next to each other, an association always puts a shape after a
colour. That is not cosmetic. An earlier version wrote statements as
`c1 s1 c2 s2 REL`, which is the same local pattern an association has, so the
only way to tell the tasks apart was to retrieve the document marker — and
across 2500 steps of pretraining, fact recall never rose above 0.27 against a
chance of 0.20. With the tasks locally distinct it reaches 0.90 in 300 steps.
`test_statements_are_locally_distinguishable_from_associations` pins it.

The checkpoint is trained once and shared: every arm loads the *same* language
model, into whatever architecture it is testing, so a difference between two
arms is caused by the stages and not by initialisation.

### The in-context-learning half did not train, and is reported as such

Fact recall reaches **1.000** — a perfect capability, and the cleanest possible
thing to watch degrade. In-context learning did not form at all. After 4000
steps the pretrained model scores **0.133** at four shots against **0.207** at
zero, an in-context gain of **−0.074** on a task where chance is 0.167.

It is not that the circuit cannot form here. On association data *alone* it
reaches 0.344 in 900 steps and is still climbing. It fails in the mixed corpus,
and raising the association share from 60% to 85% and the budget from 2500 to
4000 steps did not rescue it: what the extra steps bought was the fact table,
which jumps from 0.27 to 1.000 between steps 2500 and 3500 while the four-shot
number sits still. The most likely reading is that memorising an 8×8 table and
building an induction head compete for the same few attention heads at this
width, and the table — which gets gradient from a token in every statement,
against induction's one token per association — wins.

The consequence is stated rather than papered over. VILA's first finding, that
freezing the language model costs in-context learning specifically, **cannot be
reproduced here**: there is no such ability in the pretrained model to lose.
`scripts/report_vlm.py` detects this from the baseline, prints a note, marks the
0-shot / 4-shot / ICL columns as noise, and *skips* every check that depends on
them rather than printing a verdict computed from two chance-level numbers. The
other two findings, and everything about the decoupling, rest on fact recall and
are unaffected.

## The three corpora

The whole comparison is a difference in the text, not in the images.

| Regime | Structure | Prose share of text |
|---|---|---|
| `pairs` | caption + image, nothing else | 0.11 |
| `blend` | pairs, with text-only documents mixed back in | 0.40 |
| `interleaved` | prose, captioned image, prose, captioned image, prose | 0.67 |

Captions still ground their images in the interleaved regime — the cross-modal
dependency Part I measures is intact and `conditional_losses` still scores it.
What changes is that the model keeps seeing the text distribution it was
pretrained on.

## The stages

```
0  text pretraining     the language model.  Shared, cached, identical for every arm.
1  align                freeze the text expert, train the image side.
2  multimodal pretrain  the arm under test: pairs, interleaved, or blended.
3  instruction tuning   with or without text-only data mixed back in.
```

In a Mixture-of-Transformers the image expert *is* the projector — it is the
whole of what an image token passes through — so stage 1 freezing the text
expert and training the image side is the closest analogue of VILA's projector
initialisation. The embedding table stays trainable because it is global here;
`vlm_frozen_lm` runs the stricter version that freezes it too.

## Runs

| Run | What it isolates |
|---|---|
| `vlm_pairs` | caption-image pairs only — the COYO analogue, the reference |
| `vlm_interleaved` | interleaved documents — the MMC4 analogue |
| `vlm_blend` | text-only data mixed into pretraining |
| `vlm_pairs_sft_blend` | text-only data mixed into instruction tuning instead |
| `vlm_frozen_lm` | language model frozen through pretraining (its ICL claim is not measurable here — see above) |
| `vlm_dense` | one transformer for both modalities — does decoupling protect it? |
| `vlm_insulated` | the image loss cannot reach the text expert; nothing is frozen |
| `vlm_blocked` | cross-modal attention masked — the control |
| `vlm_upcycled` | image expert initialised from the language model |
| `vlm_share_norms` | decouple the projections and FFN, share every norm |
| `vlm_attention_only` | decouple attention, share the FFN |
| `vlm_ffn_only` | decouple the FFN, share attention — what an MoE would do |
| `vlm_taper` | decoupled below layer 4, shared above it |
| `vlm_muon` | Muon instead of AdamW |

The last five are the *optimisations of the decoupling itself*, and their
hypotheses come from Part I's own measurements: the layer norms barely
differentiate (0.08–0.11) while the big matrices pull far apart (1.2–1.3), and
the modalities de-align in the middle of the stack and re-align at the top.
[TECHNIQUES.md](TECHNIQUES.md) collects these and the ones not implemented.

## What is measured

Everything from Part I, plus three that only make sense once there is a
pretrained capability at stake:

| Probe | Question it answers |
|---|---|
| `statement_accuracy` | can it still recall a fact it learned before it saw pixels |
| `association_accuracy` | can it still answer from its context, at zero shots and four |
| `ExpertDrift` | how far multimodal training has dragged the text expert from the pretrained weights |

Capabilities are evaluated on fixed held-out batches at every probe interval
*and* at every stage boundary, so "what did stage 2 cost" is a subtraction
rather than an inference.

## Layout

```
mot/vocab.py        union vocabulary; modality is a function of token id
mot/model.py        MoT blocks, routing, global attention; partial and tapered
                    decoupling; selective stop-gradient
mot/probes.py       the measurements shared by both parts
mot/optim.py        Muon

Part I -- caption and image
mot/data.py         scene generation, bijection, document packing, ratio control
mot/train.py        instrumented training loop
mot/analysis.py     load logs into arrays, derive reported quantities
mot/configs.py      the six runs

Part II -- the VLM recipe
mot/text.py         the text corpus the language model is pretrained on
mot/vlm_data.py     pairs / interleaved / blend, and the fixed eval batches
mot/vlm_probes.py   fact recall, in-context learning, drift from pretrained
mot/vlm_train.py    staged training with a freeze schedule, shared LM checkpoint
mot/vlm_analysis.py load logs into arrays, derive reported quantities
mot/vlm_configs.py  the fourteen runs

scripts/            run.py, run_all.py, report*.py, figures*.py
viz/                artifact.html template + build.py bundler
tests/              31 tests; the controls both parts rest on are in here
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
