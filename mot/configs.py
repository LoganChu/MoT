"""The six runs. Each isolates one variable against the `balanced` reference."""

from __future__ import annotations

from mot.train import RunConfig

STEPS = 1200
SHARED = dict(steps=STEPS, batch_size=12, seq_len=192, d_model=128,
              n_layers=6, n_heads=4, probe_every=25, log_every=5, seed=0)

RUNS: dict[str, RunConfig] = {
    "balanced": RunConfig(
        name="balanced",
        description="50/50 token mixture, AdamW, token-mean loss. The reference.",
        target_text_frac=0.5, **SHARED),

    "text_heavy": RunConfig(
        name="text_heavy",
        description="85/15 text-dominated mixture. Does the starved image expert "
                    "actually end up less trained?",
        target_text_frac=0.85, **SHARED),

    "text_heavy_normalized": RunConfig(
        name="text_heavy_normalized",
        description="85/15 mixture with each modality's loss weighted equally, "
                    "removing the token-count advantage from the objective.",
        target_text_frac=0.85, loss_weighting="modality_normalized", **SHARED),

    "text_heavy_sgd": RunConfig(
        name="text_heavy_sgd",
        description="85/15 under SGD. Adam normalises updates by the second "
                    "moment; SGD does not, so imbalance should show up directly "
                    "in step size here and not under Adam.",
        target_text_frac=0.85, optimizer="sgd", lr=0.3, **SHARED),

    "blocked_attention": RunConfig(
        name="blocked_attention",
        description="50/50 but cross-modal attention is masked out. The control: "
                    "cross-modal gradient must vanish and the conditional losses "
                    "must get worse.",
        target_text_frac=0.5, blocked_attention=True, **SHARED),

    "dense": RunConfig(
        name="dense",
        description="Both modalities share one transformer. Reference for what "
                    "the modality split costs or buys in loss.",
        target_text_frac=0.5, dense=True, **SHARED),
}
