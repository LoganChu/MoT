"""The VLM run matrix.

The first four arms reproduce VILA's three pre-training findings; the last five
ask what a Mixture-of-Transformers changes about them. Every arm starts from the
*same* cached language model, so a difference between two of them is caused by
the stages, not by initialisation.
"""

from __future__ import annotations

from mot.vlm_train import Stage, VLMRunConfig

ALIGN_STEPS = 150
PRETRAIN_STEPS = 800
SFT_STEPS = 300

SHARED = dict(batch_size=16, seq_len=192, d_model=128, n_layers=6, n_heads=4,
              probe_every=50, log_every=10, probe_batch=8, eval_batch=256,
              text_steps=3000, seed=0)


def recipe(pretrain_regime: str, sft_regime: str = "pairs",
           pretrain_freeze: str = "none") -> tuple[Stage, ...]:
    """The three stages on top of the shared language model.

    `align` freezes the language model *including its embeddings* and trains
    only the image side, which in a Mixture-of-Transformers is the whole of what
    an image token passes through. Freezing the embedding is not fussiness: the
    table is tied to the output head and shared across modalities, and leaving
    it trainable for 150 align steps drove fact recall from 1.000 to 0.297 all
    on its own, with the text expert already frozen. Constrained and
    unconstrained accuracy fell together, so that was the knowledge going, not
    the head merely unlearning to emit relation tokens.

    The multimodal stages also run below the rate the language model was
    pretrained at, as every real recipe does, and the rate was bracketed by
    measurement rather than guessed: at the full 3e-3 the language model is
    obliterated whatever the data is (fact recall -1.000 in every arm), and at
    3e-4 nothing happens at all -- neither forgetting nor image learning. 1e-3
    is between them.
    """
    return (
        Stage("align", "pairs", ALIGN_STEPS,
              freeze="text_expert_and_embed", lr=2e-3),
        Stage("pretrain", pretrain_regime, PRETRAIN_STEPS,
              freeze=pretrain_freeze, lr=1e-3),
        Stage("sft", sft_regime, SFT_STEPS, lr=3e-4),
    )


VLM_RUNS: dict[str, VLMRunConfig] = {
    # --- VILA's pre-training findings ------------------------------------
    "vlm_pairs": VLMRunConfig(
        name="vlm_pairs",
        description="Multimodal pretraining on caption-image pairs only, so the "
                    "only text the model sees is a caption. The COYO analogue, "
                    "and the arm VILA reports losing 17.2% of text accuracy.",
        stages=recipe("pairs"), **SHARED),

    "vlm_interleaved": VLMRunConfig(
        name="vlm_interleaved",
        description="Pretraining on interleaved documents -- prose, a captioned "
                    "image, more prose. Same images, same captions, same "
                    "cross-modal dependency; the difference is that the text "
                    "distribution survives. The MMC4 analogue.",
        stages=recipe("interleaved"), **SHARED),

    "vlm_blend": VLMRunConfig(
        name="vlm_blend",
        description="Pairs, with text-only documents mixed back into "
                    "pretraining. The mitigation applied early rather than at "
                    "instruction tuning.",
        stages=recipe("blend"), **SHARED),

    "vlm_pairs_sft_blend": VLMRunConfig(
        name="vlm_pairs_sft_blend",
        description="Pairs throughout pretraining, then text-only data blended "
                    "into instruction tuning. VILA's third finding: joint SFT "
                    "recovers the text capability the pretraining cost.",
        stages=recipe("pairs", sft_regime="blend"), **SHARED),

    "vlm_frozen_lm": VLMRunConfig(
        name="vlm_frozen_lm",
        description="The language model frozen through multimodal pretraining. "
                    "VILA's first finding: zero-shot survives and in-context "
                    "learning does not, because the visual and text embeddings "
                    "never align in the deeper layers.",
        stages=recipe("pairs", pretrain_freeze="text_expert_and_embed"), **SHARED),

    # --- what the Mixture-of-Transformers changes ------------------------
    "vlm_dense": VLMRunConfig(
        name="vlm_dense",
        description="One transformer for both modalities. The comparison that "
                    "asks the question this repo exists for: does decoupling "
                    "the experts by modality protect the language model from "
                    "multimodal forgetting, or does global attention deliver "
                    "the damage anyway?",
        # A dense model has no image expert, so there is nothing
        # modality-specific for an alignment stage to train: freezing the
        # language model here freezes the entire network. Its align stage
        # therefore trains the embedding -- the only parameters an image token
        # touches that a text token does not entirely share -- and that is not a
        # workaround but the point. A Mixture-of-Transformers can fit a vision
        # side without disturbing the language model at all; a dense model
        # cannot, and pays for it in the drift and forgetting columns.
        stages=(Stage("align", "pairs", ALIGN_STEPS, freeze="text_expert",
                      lr=2e-3),
                Stage("pretrain", "pairs", PRETRAIN_STEPS, lr=1e-3),
                Stage("sft", "pairs", SFT_STEPS, lr=3e-4)),
        dense=True, **SHARED),

    "vlm_insulated": VLMRunConfig(
        name="vlm_insulated",
        description="The text expert still trains on text, but the image loss "
                    "can no longer reach it through the attention. Freezing "
                    "stops all learning; this stops one specific gradient path, "
                    "so whatever forgetting survives it was not caused by the "
                    "image objective reaching the language model.",
        stages=recipe("pairs"), insulate_image_gradient=True, **SHARED),

    "vlm_blocked": VLMRunConfig(
        name="vlm_blocked",
        description="Cross-modal attention masked out entirely. The control: no "
                    "cross-modal gradient exists at all, and the caption can no "
                    "longer explain the image.",
        stages=recipe("pairs"), blocked_attention=True, **SHARED),

    "vlm_upcycled": VLMRunConfig(
        name="vlm_upcycled",
        description="Sparse upcycling: the image expert starts as a copy of the "
                    "pretrained text expert rather than at random. What every "
                    "real VLM does when it initialises from something already "
                    "trained, and free here because stage 0 produces exactly "
                    "such a checkpoint.",
        stages=recipe("pairs"), upcycle=True, **SHARED),

    # --- optimisations of the decoupling itself --------------------------
    # MoT decouples every non-embedding parameter. These ask which of them
    # actually earns it, using the caption/image study's own divergence
    # measurements as the hypothesis: the big projections pull far apart while
    # the layer norms barely differentiate, and the modalities re-align in the
    # top layers after separating in the middle.
    "vlm_share_norms": VLMRunConfig(
        name="vlm_share_norms",
        description="Decouple the projections and the feed-forward network; "
                    "share every normalisation. The caption/image study found "
                    "the layer norms diverging by 0.08-0.11 against 1.2-1.3 for "
                    "the big matrices, which predicts that decoupling them buys "
                    "nothing.",
        stages=recipe("pairs"),
        decoupled_submodules=("wq", "wk", "wv", "wo", "ffn"), **SHARED),

    "vlm_attention_only": VLMRunConfig(
        name="vlm_attention_only",
        description="Decouple only the attention projections and share the "
                    "feed-forward network, which is the largest block of "
                    "per-expert parameters. The parameter-saving version of the "
                    "same question.",
        stages=recipe("pairs"),
        decoupled_submodules=("attn_norm", "wq", "wk", "wv", "q_norm", "k_norm",
                              "wo"), **SHARED),

    "vlm_ffn_only": VLMRunConfig(
        name="vlm_ffn_only",
        description="The opposite split: decouple only the feed-forward network "
                    "and share all of attention. This is what a mixture of "
                    "experts decouples, so the pair of arms says whether MoT is "
                    "buying anything a modality-aware MoE would not.",
        stages=recipe("pairs"),
        decoupled_submodules=("ffn_norm", "ffn"), **SHARED),

    "vlm_taper": VLMRunConfig(
        name="vlm_taper",
        description="Decoupled in the bottom four layers, fully shared in the "
                    "top two. The caption/image study found the modalities "
                    "de-aligning in the middle of the stack and re-aligning at "
                    "the top -- encode separately, then share -- and this is "
                    "that schedule applied.",
        stages=recipe("pairs"), shared_from_layer=4, **SHARED),

    "vlm_muon": VLMRunConfig(
        name="vlm_muon",
        description="Muon instead of AdamW. The caption/image study showed "
                    "weight movement is near scale-free in the gradient because "
                    "Adam divides by a second moment, and plainly not scale-free "
                    "under SGD. Muon is a third answer: it orthogonalises the "
                    "update instead of rescaling it element by element.",
        stages=recipe("pairs"), optimizer="muon", lr=0.02, **SHARED),
}
