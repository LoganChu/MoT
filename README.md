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
scratch. **Part II** does it for [NVIDIA Cosmos 3](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf),
whose two-tower architecture *is* a Mixture-of-Transformers — an autoregressive
reasoner and a diffusion generator sharing one attention operator — and tests
which parts of the decoupling earn their parameters.

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
python -m pytest tests/ -q            # 31 correctness tests, ~6s
python scripts/run_all.py             # every run in parallel (~50 min, CPU)
python scripts/report.py              # the caption/image numbers
python scripts/report_cosmos.py       # the two-tower numbers
python scripts/figures.py             # caption/image figures -> figures/
python scripts/figures_cosmos.py      # two-tower figures    -> figures/
python viz/build.py                   # interactive page -> viz/mot_dynamics.html
```

Single run: `python scripts/run.py balanced --steps 400`. One suite at a time:
`python scripts/run_all.py --suite base` or `--suite cosmos`. Caption/image logs
land in `runs/`, two-tower logs in `runs_cosmos/`. The two-tower suite trains its
shared vision-language model once and caches it in `runs_cosmos/cache/`;
`run_all.py` builds it before fanning out so the arms all start from the same
weights.

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

---

# Part II — the Cosmos 3 two-tower model

[Cosmos 3](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)
(NVIDIA, June 2026) is a Mixture-of-Transformers, and the decoupling is per
*tower* rather than per modality:

- a **reasoner** tower — a vision-language model over images, video and text,
  autoregressive, causal self-attention
- a **generator** tower — diffusion, producing future observations and action
  trajectories conditioned on the reasoner, its noisy tokens attending in full
- the two "share one transformer architecture and a joint attention operator",
  and both are initialised from pretrained Qwen3-VL weights

That is the architecture this repo was built to instrument, with the question
put more sharply than Part I could: the reasoner is a working vision-language
model, the generator's objective is not even the same *kind* of function, and
the two share an attention. So what does the diffusion loss do to the language
model it is bolted to?

## The world

A 3×3 board and a rule the text names — `shift_right`, `transpose`, `recolour`,
`mirror`, `shift_down`. The next frame is a deterministic function of the rule
*and* the board and of neither alone, which is what makes a conditioning
ablation a measurement. The current frame is captioned, so the reasoner has a
real cross-modal capability the generator's objective can be caught damaging.

```
<doc> | <bot> rule colour caption <eot> | <boi> current frame <eoi> | <gen> x 9
```

The nine generator slots carry no token identity. Their embeddings are noised
latents of the next frame, injected through a separate continuous path, and they
are scored by flow matching rather than cross-entropy. Causal throughout, except
that the generator's slots see each other in full and the reasoner never reads
them at all — so the reasoner runs standalone, as Cosmos 3's does, and
`test_the_reasoner_forward_pass_ignores_the_generator` proves it rather than
asserting it.

## What the measurements say

All numbers from `scripts/report_cosmos.py` on the committed logs. Every arm
starts from the same cached vision-language model and runs at the same learning
rate.

**Nearly half the vision-language model's gradient comes from the other tower.**
In the reference run the diffusion objective supplies **44.6%** of the reasoner's
gradient — computed entirely on positions the reasoner never processes, and
non-zero only because the attention is joint. Block the towers and it is
*identically* zero.

**And it is crowding out the reasoner's own learning.** Insulating that path —
cutting the diffusion gradient without freezing anything, so the reasoner still
trains on its own loss — leaves the reasoner's image loss **0.835 better** than
joint training does (−0.835 against −0.008). Summed over layers, the reasoner's
own image-cross-entropy gradient is 0.22 insulated against 0.03 joined: seven
times more of its own objective actually reaches it once the diffusion tower
stops competing for the same weights.

**The generator genuinely conditions on both halves.** Cutting the rule from its
view costs **32×** the diffusion loss; cutting the board costs **27×**. Blocked,
both are exactly 1.00× and the generator reconstructs **0.0%** of frames.

**A reasoner trained by nothing but the other tower still works.** With no
cross-entropy at all, the diffusion loss becomes 100% of the reasoner's gradient
— and the generator still reaches 0.787 cell accuracy. What Part I measured as a
side channel is here sufficient, on its own, to train half the model.

**The reasoner has to adapt, not merely be present.** Frozen, it conditions the
generator three times worse (0.403 against 0.125).

### Four predictions the data did not support

- **"Attaching a diffusion tower degrades the vision-language model."** It did
  not: the reasoner's losses barely moved (−0.008). The damage is real but it is
  an *opportunity cost*, visible only against the insulated arm above — the
  reasoner would have improved a great deal more had the second objective not
  been there.
- **"Decoupling the towers beats sharing one transformer."** One transformer for
  both objectives generated better (0.0997 against 0.1250).
- **"Splitting the reasoner further does not pay for itself."** It paid
  handsomely — three towers, with the reasoner itself decoupled over image and
  text, reached **0.0606** and reconstructed **34.6%** of frames exactly against
  8.2% for two towers. Cosmos 3 splits per tower; on this task splitting per
  modality inside the reasoner was worth more than the tower split itself.
- **"Upcycling the generator from the vision-language model helps."** It hurt
  (0.0835 from random against 0.1250 upcycled) — the same result the caption
  study's own upcycling arm gave. A *trained* initialisation drops the second
  tower into a basin it does not leave.

### Which decoupling earns its parameters

Ranked by held-out diffusion loss: **attention-only 0.0597**, share-norms 0.0781,
dense 0.0997, taper 0.1013, two-tower 0.1250, **FFN-only 0.1277**.

Decoupling only the attention path and sharing the entire feed-forward network —
the largest block of per-tower parameters — was the best layout tested. Sharing
attention and decoupling only the FFN, which is what a mixture-of-experts does,
was the worst. The ordering is the reverse of what Part I's divergence numbers
predict, where the layer norms differentiate least (0.08–0.11) and the big
matrices most (1.31). **How far two experts drift apart does not tell you which
had to be separate.**

### The optimiser is the largest single effect

Muon reached a diffusion loss of **0.0169** and reconstructed **97.5%** of frames
exactly, against 0.1250 and 8.2% for AdamW at the same learning rate — while
drifting *least* from the pretrained reasoner (1.28 against 3.44). That gap was
checked against the obvious confound: at 3e-3 the two optimisers are
indistinguishable (0.035 against 0.039 exact frames) and only separate at 1e-2,
so it is the optimiser and not the step size. The learning rate itself matters
enormously and was chosen by measurement — at 1e-3 the generator reconstructs
0.8% of frames, at 1e-2 it reaches 43.8%, and every architecture comparison made
down at 1e-3 is a comparison between two models that have not finished learning.

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

Part II -- the Cosmos 3 two-tower model
mot/world.py           a board, a rule, and the frame the pair determines
mot/cosmos_data.py     the reasoner prefix and the generator's noised slots
mot/cosmos_probes.py   cross-tower attribution, conditioning ablations, decoding
mot/cosmos_train.py    two-tower training on a shared vision-language checkpoint
mot/cosmos_analysis.py load logs into arrays, derive reported quantities
mot/cosmos_configs.py  the thirteen runs
mot/optim.py           Muon

scripts/            run.py, run_all.py, report*.py, figures*.py
viz/                artifact.html template + build.py bundler
tests/              31 tests; the controls both parts rest on are in here
```

## Reading the results honestly

These are ~2.4M-parameter models on synthetic data with a nine-cell board.
**Magnitudes do not transfer to Cosmos 3's 16B, and neither do the architecture
rankings.** Three towers beating two here says something about a six-layer model
on a five-rule world; it does not say NVIDIA factorised theirs wrongly, and the
same caveat applies to every architecture ordering in Part II.

What does transfer is mechanism, because it follows from the architecture and
the optimiser rather than from the task:

- gradient exposure scales with token count while Adam's step size does not
- one joint attention operator creates a gradient path into every expert
  regardless of what that expert processes, and the path can be cut *exactly* --
  by blocking the attention, or by detaching one modality's reads
- a second objective on a shared attention competes for the same weights, so its
  cost to the first shows up as forgone improvement rather than as damage, and
  is invisible unless you run the insulated arm to compare against
- how far two experts drift apart does not predict which of them had to be
  separate

Two measurement choices are worth naming because they are choices, not
discoveries. Identical expert initialisation pins divergence to zero at step 0,
so later divergence is attributable to training. And the probe batches fix their
diffusion time and noise draw, so a change in the generator's loss over training
is the model moving rather than a different sample.

One number in Part II is redundant by construction rather than by coincidence:
every image token in the two-tower layout sits in a caption-then-frame segment,
so `image_given_text` and the unconditional image loss cover identical positions
and are always equal. The two-tower report uses the plain image loss for that
reason.
