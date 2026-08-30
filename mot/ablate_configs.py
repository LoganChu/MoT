"""The ablation matrix.

Three groups. The first asks what the trained model's reliance on M-RoPE looks
like, the second attributes DeepStack's contribution to individual depths, and
the third asks the question neither source paper asks: whether the two
techniques are supplying the same information.
"""

from __future__ import annotations

from mot.ablate import AblationSpec

ABLATIONS: dict[str, AblationSpec] = {
    "baseline": AblationSpec(
        name="baseline",
        description="Unmodified Qwen3-VL. The reference every arm is paired against."),

    # --- what the model's 2-D positions are worth -------------------------
    "mrope_1d": AblationSpec(
        name="mrope_1d", mrope="1d",
        description="Every token takes its sequential index on all three axes, "
                    "which is ordinary 1-D RoPE. Visual tokens keep their order "
                    "but lose all grid structure."),

    "mrope_swap_hw": AblationSpec(
        name="mrope_swap_hw", mrope="swap_hw",
        description="Height and width coordinates exchanged. The 2-D structure "
                    "survives intact and only its orientation changes, so this "
                    "separates 'uses a grid' from 'uses the right axes'."),

    "mrope_shuffle": AblationSpec(
        name="mrope_shuffle", mrope="shuffle",
        description="Visual coordinates permuted among visual tokens. The set of "
                    "positions is unchanged and only their assignment is "
                    "destroyed, so a drop cannot be blamed on unseen position "
                    "ranges. The floor for the M-RoPE group."),

    # --- what DeepStack contributes, and at which depth --------------------
    **{f"ds_off_{i}": AblationSpec(
        name=f"ds_off_{i}", deepstack_off=(i,),
        description=f"Injection {i} zeroed: vision layer {(5, 11, 17)[i]} no "
                    f"longer reaches decoder layer {i}. Attributes DeepStack's "
                    f"contribution to a single depth.")
       for i in range(3)},

    "ds_off_all": AblationSpec(
        name="ds_off_all", deepstack_off=(0, 1, 2),
        description="All three injections zeroed, leaving the ordinary "
                    "single-entry-point VLM. The published DeepStack gain is the "
                    "prediction for this arm, so it doubles as a check on the "
                    "harness."),

    "ds_shuffle": AblationSpec(
        name="ds_shuffle", deepstack_shuffle=True,
        description="Injected features permuted across positions. Signal "
                    "magnitude and statistics are preserved and only their "
                    "alignment to tokens is broken, separating 'needs these "
                    "features' from 'needs them in the right place'."),

    # --- are the two techniques redundant? ---------------------------------
    "mrope_1d_ds_off_all": AblationSpec(
        name="mrope_1d_ds_off_all", mrope="1d", deepstack_off=(0, 1, 2),
        description="Both removed. If the joint damage is smaller than the sum "
                    "of the separate damages, DeepStack was re-supplying spatial "
                    "information that M-RoPE also carries; if larger, the two are "
                    "complementary. Neither source paper measures this."),
}
