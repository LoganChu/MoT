"""The two-tower run matrix.

Every arm starts from the *same* cached vision-language model, so a difference
between two of them is caused by the variable under test and not by
initialisation. The first group asks what the tower split is worth, the second
what the two towers do to each other, and the third which parts of the
decoupling earn their parameters.
"""

from __future__ import annotations

from mot.cosmos_train import CosmosRunConfig

SHARED = dict(steps=2000, batch_size=32, d_model=128, n_layers=6, n_heads=4,
              probe_every=100, log_every=20, probe_batch=16, eval_batch=128,
              pretrain_steps=1500, seed=0)

COSMOS_RUNS: dict[str, CosmosRunConfig] = {
    # --- what the tower split is worth ------------------------------------
    "cosmos_two_tower": CosmosRunConfig(
        name="cosmos_two_tower",
        description="Cosmos 3's factorisation: an autoregressive reasoner tower "
                    "over text and image, a diffusion generator tower over the "
                    "next frame, decoupled weights and one joint attention "
                    "operator, both towers upcycled from the same pretrained "
                    "vision-language model. The reference.",
        towers="split", **SHARED),

    "cosmos_dense": CosmosRunConfig(
        name="cosmos_dense",
        description="One transformer for both objectives. Reference for what "
                    "the tower split costs or buys -- and the arm that says "
                    "whether a diffusion objective and a language-model "
                    "objective can share weights as well as attention.",
        towers="dense", **SHARED),

    "cosmos_three_tower": CosmosRunConfig(
        name="cosmos_three_tower",
        description="The reasoner itself decoupled over image and text, so "
                    "three towers rather than two. Cosmos 3 splits per tower, "
                    "not per modality within a tower; this measures whether "
                    "splitting the vision-language side further is worth it.",
        towers="three", **SHARED),

    "cosmos_no_upcycle": CosmosRunConfig(
        name="cosmos_no_upcycle",
        description="The generator tower starts from random initialisation "
                    "rather than from the pretrained vision-language model. "
                    "Cosmos 3 starts both towers from Qwen3-VL weights, so this "
                    "is the ablation of that choice.",
        towers="split", upcycle=False, **SHARED),

    # --- what the two towers do to each other ------------------------------
    "cosmos_insulated": CosmosRunConfig(
        name="cosmos_insulated",
        description="The generator reads the reasoner but cannot backpropagate "
                    "into it. Nothing is frozen -- the reasoner still trains on "
                    "its own objective -- so whatever changes is caused by the "
                    "diffusion gradient specifically, and not by anything else "
                    "joint training does.",
        towers="split", insulate_generator=True, **SHARED),

    "cosmos_frozen_reasoner": CosmosRunConfig(
        name="cosmos_frozen_reasoner",
        description="The reasoner frozen outright during joint training, so the "
                    "generator must condition on a vision-language model that "
                    "never adapts to it. The blunt version of insulation, and "
                    "the pair says whether the reasoner needs to move or merely "
                    "needs to be left alone.",
        towers="split", freeze_reasoner=True, **SHARED),

    "cosmos_gen_only": CosmosRunConfig(
        name="cosmos_gen_only",
        description="No cross-entropy at all: the reasoner tower has no "
                    "objective of its own and is trained solely by gradient "
                    "arriving from the diffusion loss across the attention. "
                    "What the caption/image study measured as a side channel is "
                    "here the entire training signal for half the model.",
        towers="split", lm_loss=False, **SHARED),

    "cosmos_blocked": CosmosRunConfig(
        name="cosmos_blocked",
        description="The generator cannot attend to the reasoner at all. The "
                    "control: the cross-tower gradient must be identically "
                    "zero, the conditioning ablations must become no-ops, and "
                    "the generator must fail, because the next frame is not "
                    "predictable from noise alone.",
        towers="split", blocked_attention=True, **SHARED),

    # --- which decoupling earns its parameters -----------------------------
    "cosmos_share_norms": CosmosRunConfig(
        name="cosmos_share_norms",
        description="Decouple the projections and the feed-forward network "
                    "between the towers; share every normalisation.",
        towers="split",
        decoupled_submodules=("wq", "wk", "wv", "wo", "ffn"), **SHARED),

    "cosmos_attention_only": CosmosRunConfig(
        name="cosmos_attention_only",
        description="Decouple only the attention path between the towers and "
                    "share the feed-forward network, which is the largest block "
                    "of per-tower parameters.",
        towers="split",
        decoupled_submodules=("attn_norm", "wq", "wk", "wv", "q_norm", "k_norm",
                              "wo"), **SHARED),

    "cosmos_ffn_only": CosmosRunConfig(
        name="cosmos_ffn_only",
        description="Decouple only the feed-forward network and share all of "
                    "attention -- what a mixture of experts would do. The pair "
                    "with the arm above says whether a two-tower MoT is buying "
                    "anything an MoE would not.",
        towers="split", decoupled_submodules=("ffn_norm", "ffn"), **SHARED),

    "cosmos_taper": CosmosRunConfig(
        name="cosmos_taper",
        description="Towers decoupled in the bottom four layers and fully "
                    "shared in the top two -- separate early, joint late.",
        towers="split", shared_from_layer=4, **SHARED),

    "cosmos_muon": CosmosRunConfig(
        name="cosmos_muon",
        description="Muon instead of AdamW. The two towers optimise objectives "
                    "whose gradients differ in scale by construction, which is "
                    "exactly the situation where how an optimiser normalises "
                    "its update decides which tower moves. Run at the same "
                    "learning rate as every other arm, so the comparison is "
                    "between optimisers and not between step sizes.",
        towers="split", optimizer="muon", **SHARED),
}
