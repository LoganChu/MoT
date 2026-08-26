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
*Implemented:* `decoupled_submodules`, arms `cosmos_share_norms`,
`cosmos_attention_only`, `cosmos_ffn_only`.

The caption/image study measures, per sub-module, how far the two experts
actually pull apart: `wq` 1.31, `wk` 1.28, `wo` 1.26, `wv` 1.22, `ffn` 1.17 —
and the layer norms at 0.08–0.11. A parameter that ends up in the same place for
both modalities did not need two copies. The three arms split the difference
three ways, and `cosmos_ffn_only` doubles as the question of whether MoT buys
anything a modality-aware mixture of experts would not, since an MoE decouples
the feed-forward network and nothing else.

**Depth-tapered decoupling.** *Implemented:* `shared_from_layer`, arm
`cosmos_taper`.

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

**Mixture-of-Depths for the observation stream.** *Not implemented.* Observation
tokens are the majority of a two-tower sequence. Letting them exit early
compounds with the FLOP saving MoT already gets from skipping the other tower's
feed-forward network.

---

## Optimisation

**Muon instead of AdamW.** *Implemented:* `optimizer="muon"`, `mot/optim.py`,
arm `cosmos_muon`.

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
`cosmos_insulated`.

Cut one modality's loss out of another modality's expert without freezing
anything. Here it isolates whether a diffusion objective damages the vision-language
model it is bolted to; the same mechanism is what knowledge-insulation schemes
use to protect a backbone from a second objective.

---

## Initialisation and data

**Sparse upcycling.** *Implemented:* `upcycle`, arm `cosmos_no_upcycle`.

Start the second tower from the pretrained vision-language model rather than
from noise. Cosmos 3 initialises *both* of its towers from pretrained Qwen3-VL
weights, so upcycling is the faithful default here and the arm is its ablation.

**Cross-objective insulation.** *Implemented:* `insulate_modality`, arm
`cosmos_insulated`.

A two-tower model puts an autoregressive objective and a diffusion objective on
one attention operator, and the second reaches the first whether or not you
wanted it to. Cutting that path -- without freezing anything, so the reasoner
still trains on its own loss -- is the only way to attribute a change to that
gradient specifically.

**Freezing the conditioning tower.** *Implemented:* `freeze_reasoner`, arm
`cosmos_frozen_reasoner`. The blunt version of the above, and the pair says
whether the reasoner needs to adapt to the generator or merely be left alone.

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
| Sparse upcycling | `upcycle` | generator quality, `ExpertDrift` |
| Cross-objective insulation | `insulate_modality` | `tower_attribution` |
| Freezing a tower | `freeze_reasoner` | `ExpertDrift`, generator quality |
