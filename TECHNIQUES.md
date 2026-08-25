# Making a Mixture-of-Transformers better

A working menu, ordered by how cheaply each one can be *measured* with the
probes in this repo. Entries marked **implemented** have a run arm; the rest
name the hook they would attach to, so the list stays actionable rather than
bibliographic.

MoT as published decouples every non-embedding parameter by modality and shares
exactly one thing: the attention. That is a strong, simple choice, and most of
what follows is a way of relaxing it in one direction or another.

---

## Decoupling less than everything

**Partial decoupling — share the sub-modules that never differentiate.**
*Implemented:* `decoupled_submodules`, arms `vlm_share_norms`,
`vlm_attention_only`, `vlm_ffn_only`.

The caption/image study measures, per sub-module, how far the two experts
actually pull apart: `wq` 1.31, `wk` 1.28, `wo` 1.26, `wv` 1.22, `ffn` 1.17 —
and the layer norms at 0.08–0.11. A parameter that ends up in the same place for
both modalities did not need two copies. The three arms split the difference
three ways, and `vlm_ffn_only` doubles as the question of whether MoT buys
anything a modality-aware mixture of experts would not, since an MoE decouples
the feed-forward network and nothing else.

**Depth-tapered decoupling.** *Implemented:* `shared_from_layer`, arm
`vlm_taper`.

Linear CKA between the modalities collapses through the middle of the stack
(0.78 → 0.16) and recovers at the top (0.93 → 0.86), while cross-modal attention
rises with depth. Both say the same thing: encode separately, then share. A
depth schedule on decoupling follows directly, and nothing in the architecture
requires the decision to be global.

**Low-rank residual experts — `W = W_shared + Δ_m`.** *Not implemented.*
The continuous version of partial decoupling, and the one that actually saves
parameters at scale. `MoTBlock` would grow a shared base weight plus a per
modality low-rank term; `expert_divergence` already reports exactly the quantity
that would tell you what rank each sub-module deserves.

**MoT × MoE (MoMa).** *Not implemented.* Modality-aware routing on the outside,
learned sparse-FFN routing inside each modality expert. Adds capacity without
the load-balancing pathology, because the modality routing stays deterministic
and only the inner routing is learned.

**Mixture-of-Depths for the image stream.** *Not implemented.* Image tokens are
the redundant majority — over half the stream in the `pairs` regime here. Letting
them exit early compounds with the FLOP saving MoT already gets from skipping the
other modality's feed-forward network.

---

## Optimisation

**Muon instead of AdamW.** *Implemented:* `optimizer="muon"`, `mot/optim.py`,
arm `vlm_muon`.

The caption/image study's headline result is that weight movement is close to
scale-free in the gradient *because Adam divides by a running second moment* — an
85/15 token split still leaves the starved expert at 1.04× the movement of the
fed one, and plain SGD drops that to 0.69×. Muon is a third answer to the same
question and a structurally different one: it orthogonalises the update rather
than rescaling it element by element, so every direction of a weight matrix moves
by about the same amount however the gradient was distributed across them.

**Per-expert learning rate, and μP.** *Not implemented.* Once experts differ in
width — which partial decoupling and upcycling both invite — the learning rate
that is right for one is wrong for the other, and μP gives a principled scaling
rather than a tuned guess.

**Gradient surgery and multi-objective weighting.** *Not implemented.* PCGrad,
GradNorm, uncertainty weighting. The caption/image study's `modality_normalized`
arm is the crudest member of this family, and every one of them is directly
visible in `grad_attribution`.

**Selective stop-gradient.** *Implemented:* `insulate_modality`, arm
`vlm_insulated`.

Cut one modality's loss out of another modality's expert without freezing
anything. Here it isolates whether multimodal forgetting travels along the
cross-modal attention path; the same mechanism is what knowledge-insulation
schemes use to protect a backbone from a second objective.

---

## Initialisation and data

**Sparse upcycling.** *Implemented:* `upcycle`, arm `vlm_upcycled`.

Start the image expert from the pretrained language model rather than from
noise, which is what every real VLM does when it initialises from something
already trained. Free here, because stage 0 produces exactly such a checkpoint.

**Interleaved rather than paired multimodal data.** *Implemented:* the
`interleaved` regime, arm `vlm_interleaved`.

VILA's central pre-training result, and the reason it is in this repo at all:
training on caption-image pairs degrades text-only accuracy by 17.2% while
interleaved documents cost about 5%, because a caption corpus quietly replaces
the text distribution the language model was trained on.

**Text replay.** *Implemented:* the `blend` regime, arms `vlm_blend` and
`vlm_pairs_sft_blend`.

Mixing text-only data back in, either during multimodal pretraining or only at
instruction tuning. VILA reports the latter recovering MMLU from 40.7% to 51.4%
*and* improving the visual scores at the same time.

**Staged training with a freeze schedule.** *Implemented:* `Stage.freeze`, arm
`vlm_frozen_lm`. Align, pretrain, instruction-tune — and the finding that
freezing the language model through pretraining preserves zero-shot accuracy
while costing in-context learning (72.1% → 58.1% at four shots).

---

## Systems

MoT needs no load-balancing loss, because routing is deterministic — but it does
have *device* imbalance, since the modality mix of a batch decides which
expert's weights are hot. Modality-grouped sequence packing and expert-parallel
placement are where the wall-clock win actually lands, and neither shows up in a
loss curve. *Not implemented; out of scope for a CPU-sized study.*

---

## What each hook is measured by

| Technique | Hook | Probe that shows it |
|---|---|---|
| Partial decoupling | `decoupled_submodules` | `expert_divergence`, held-out losses |
| Depth taper | `shared_from_layer` | `layerwise_cka`, `attention_mass` |
| Muon | `optimizer` | `displacement`, `GradientSNR` |
| Selective stop-gradient | `insulate_modality` | `grad_attribution` |
| Sparse upcycling | `upcycle` | image loss, `ExpertDrift` |
| Interleaved data | regime | `statement_acc`, `assoc4_acc` |
| Text replay | regime | the same, after the stage that adds it |
| Freeze schedule | `Stage.freeze` | `ExpertDrift`, in-context gain |
