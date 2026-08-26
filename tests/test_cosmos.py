"""Correctness tests for the two-tower (Cosmos 3) study.

Three properties carry it, and each has a test whose job is to earn it:

  * the next frame is determined by the rule *and* the board and by neither
    alone, and the caption determines the current frame. Without the first, a
    conditioning ablation would prove nothing; without the second, the reasoner
    has no cross-modal capability for the generator's objective to damage.

  * the attention is the architecture: the generator reads the reasoner and its
    own slots, and the reasoner reads neither -- which is what makes "the
    reasoner can be called independently" true of this model.

  * the interventions are exact. Blocking zeroes the whole cross-tower path;
    insulation zeroes only the diffusion gradient into the reasoner and changes
    no forward pass.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mot.cosmos_configs import COSMOS_RUNS
from mot.cosmos_data import (
    N_CAPTION_OBJECTS, SEQ_LEN, caption_tokens, decode_batch, make_batch,
)
from mot.cosmos_probes import (
    flow_loss, generation_quality, reasoner_ablation, tower_attention_mass,
    tower_attribution,
)
from mot.cosmos_train import TOWER_LAYOUTS, load_pretrained, tower_parameters
from mot.model import GeneratorSpec, MoTConfig, MoTModel
from mot.world import (
    LATENT_DIM, MAX_OBJECTS, N_RULES, N_SLOTS, apply_rule, decode_frame,
    encode_frame, sample_scene, sample_transition,
)
from mot.vocab import COSMOS_VOCAB_SIZE, GENERATOR, IMAGE, N_COSMOS_MODALITIES, TEXT


def _config(**overrides) -> MoTConfig:
    base = dict(vocab_size=COSMOS_VOCAB_SIZE, n_layers=3, d_model=64, n_heads=4,
                n_modalities=N_COSMOS_MODALITIES, expert_of_modality=(0, 0, 1),
                seq_len=SEQ_LEN, generator=GeneratorSpec(LATENT_DIM, N_SLOTS),
                generator_modality=GENERATOR, tower_attention=True)
    base.update(overrides)
    return MoTConfig(**base)


def _model_and_batch(seed: int = 0, batch_size: int = 6, **overrides):
    torch.manual_seed(seed)
    model = MoTModel(_config(**overrides))
    batch = make_batch(np.random.default_rng(seed), batch_size)
    return model, batch


# --- the world --------------------------------------------------------------

def test_the_future_needs_both_the_rule_and_the_board():
    """Neither input alone determines the next frame, so ablating either must
    cost the generator something."""
    rng = np.random.default_rng(0)
    by_rule: dict[tuple, set] = {}
    by_board: dict[tuple, set] = {}
    for _ in range(4000):
        t = sample_transition(rng)
        by_rule.setdefault((t.rule, t.argument), set()).add(t.future.cells)
        by_board.setdefault(t.scene.cells, set()).add(t.future.cells)
    assert max(len(v) for v in by_rule.values()) > 1
    assert max(len(v) for v in by_board.values()) > 1


def test_the_caption_determines_the_current_frame():
    """The reasoner's cross-modal objective has to be learnable.

    A scene may hold no more objects than the caption can name. It once held
    four against a caption of three, and `image_given_text` then sat at exactly
    the unconditional image loss -- the objective was not hard, it was
    impossible.
    """
    assert MAX_OBJECTS <= N_CAPTION_OBJECTS
    rng = np.random.default_rng(1)
    by_caption: dict[tuple, set] = {}
    for _ in range(4000):
        t = sample_transition(rng)
        by_caption.setdefault(tuple(caption_tokens(t)), set()).add(t.scene.cells)
    assert max(len(v) for v in by_caption.values()) == 1


def test_every_rule_moves_the_board_somewhere_different():
    scene = sample_scene(np.random.default_rng(3))
    futures = {apply_rule(scene, rule, 2).cells for rule in range(N_RULES)}
    assert len(futures) == N_RULES


def test_the_latent_encoding_round_trips():
    rng = np.random.default_rng(4)
    for _ in range(300):
        scene = sample_scene(rng)
        assert decode_frame(encode_frame(scene)).cells == scene.cells


def test_ground_truth_latents_decode_perfectly():
    _, batch = _model_and_batch()
    cell, exact = decode_batch(batch.future, batch)
    assert cell == 1.0 and exact == 1.0


def test_generator_slots_carry_no_cross_entropy():
    """A generator slot has no token identity, so nothing may be scored on it."""
    _, batch = _model_and_batch()
    assert int(batch.is_generator.sum(1).unique()) == N_SLOTS
    assert not bool((batch.lm_mask & (batch.mod_y == GENERATOR)).any())
    assert not bool((batch.lm_mask & (batch.mod_x == GENERATOR)).any())


def test_flow_interpolant_matches_the_definition():
    _, batch = _model_and_batch()
    tau = batch.flow_t[:, None, None]
    assert torch.allclose(batch.noisy(), tau * batch.noise + (1 - tau) * batch.future)
    assert torch.allclose(batch.velocity_target(), batch.noise - batch.future)


# --- the attention is the architecture --------------------------------------

def test_the_reasoner_never_reads_the_generator():
    """Which is what lets the reasoner tower be called on its own."""
    model, batch = _model_and_batch()
    mask = model.tower_mask(batch.is_generator)[0, 0]
    slots = torch.nonzero(batch.is_generator[0]).flatten()
    assert not bool(mask[0, slots[0]])
    assert not bool(mask[:slots[0], slots].any())
    assert bool(mask[slots[0], 0]), "the generator cannot read the reasoner"
    assert bool(mask[slots[0], slots[-1]]), "the chunk is not denoised jointly"


def test_the_reasoner_forward_pass_ignores_the_generator():
    """Stronger than the mask: changing the noise must not move the reasoner."""
    model, batch = _model_and_batch()
    with torch.no_grad():
        a, _ = model(batch.x, batch.mod_x, generator=batch.generator_inputs())
        shifted = batch.generator_inputs()
        shifted["noisy"] = shifted["noisy"] + 5.0
        b, _ = model(batch.x, batch.mod_x, generator=shifted)
    reasoner = ~batch.is_generator
    assert torch.allclose(a[reasoner], b[reasoner], atol=1e-5)


def test_attention_mass_rows_are_distributions():
    model, batch = _model_and_batch()
    mass = tower_attention_mass(model, batch)
    present = ~np.all(np.isnan(mass), axis=2)
    assert np.allclose(np.nansum(mass, axis=2)[present], 1.0, atol=1e-4)


# --- who trains whom --------------------------------------------------------

def test_the_diffusion_loss_reaches_the_reasoner():
    """The cross-tower path, non-zero only because the attention is joint."""
    model, batch = _model_and_batch()
    attribution = tower_attribution(model, batch)
    assert attribution[:, 0, GENERATOR].min() > 0
    # ...and nothing flows the other way: the generator tower never processes a
    # reasoner token, so no cross-entropy can reach it.
    assert np.all(attribution[:, 1, TEXT] == 0.0)
    assert np.all(attribution[:, 1, IMAGE] == 0.0)


def test_blocked_attention_zeroes_the_cross_tower_path():
    model, batch = _model_and_batch(blocked_attention=True)
    attribution = tower_attribution(model, batch)
    assert np.all(attribution[:, 0, GENERATOR] == 0.0)
    assert attribution[:, 0, TEXT].min() > 0
    assert attribution[:, 1, GENERATOR].min() > 0


def test_blocked_attention_makes_the_ablations_no_ops():
    model, batch = _model_and_batch(blocked_attention=True)
    ablation = reasoner_ablation(model, batch)
    assert ablation["no_text"] == pytest.approx(ablation["full"], abs=1e-6)
    assert ablation["no_image"] == pytest.approx(ablation["full"], abs=1e-6)


def test_insulation_zeroes_only_the_diffusion_gradient():
    model, batch = _model_and_batch(insulate_modality=GENERATOR)
    attribution = tower_attribution(model, batch)
    assert np.all(attribution[:, 0, GENERATOR] == 0.0)
    assert attribution[:, 0, TEXT].min() > 0, "the reasoner stopped training"
    assert attribution[:, 1, GENERATOR].min() > 0, "the generator stopped training"


def test_insulation_leaves_the_forward_pass_unchanged():
    batch = make_batch(np.random.default_rng(0), 4)
    torch.manual_seed(0)
    plain = MoTModel(_config())
    torch.manual_seed(0)
    cut = MoTModel(_config(insulate_modality=GENERATOR))
    with torch.no_grad():
        a, aux_a = plain(batch.x, batch.mod_x, generator=batch.generator_inputs())
        b, aux_b = cut(batch.x, batch.mod_x, generator=batch.generator_inputs())
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(aux_a["velocity"], aux_b["velocity"], atol=1e-6)


def test_probes_leave_no_gradient_behind():
    model, batch = _model_and_batch()
    tower_attribution(model, batch)
    assert all(p.grad is None for p in model.parameters())


# --- the layouts ------------------------------------------------------------

def test_every_tower_layout_builds_and_trains():
    batch = make_batch(np.random.default_rng(0), 4)
    for name, layout in TOWER_LAYOUTS.items():
        torch.manual_seed(0)
        model = MoTModel(_config(expert_of_modality=layout))
        logits, aux = model(batch.x, batch.mod_x,
                            generator=batch.generator_inputs())
        loss = flow_loss(aux["velocity"], batch)
        loss.backward()
        trainable = [p for p in model.parameters() if p.grad is not None]
        assert trainable, name


def test_the_pretrained_model_transfers_into_every_layout():
    from mot.cosmos_train import CosmosRunConfig

    cfg = CosmosRunConfig(name="t", d_model=64, n_layers=2)
    torch.manual_seed(0)
    pretrained = MoTModel(cfg.pretrain_config())
    for towers in TOWER_LAYOUTS:
        arm = CosmosRunConfig(name="t", d_model=64, n_layers=2, towers=towers)
        model = MoTModel(arm.model_config())
        load_pretrained(model, pretrained.state_dict(), upcycle=True)
        assert torch.equal(model.layers[0].wq[0].weight,
                           pretrained.layers[0].wq[0].weight), towers
        assert torch.equal(model.layers[0].wq[-1].weight,
                           pretrained.layers[0].wq[0].weight), towers


def test_every_arm_has_parameters_to_train():
    for name, cfg in COSMOS_RUNS.items():
        config = cfg.model_config()
        config.n_layers, config.d_model = 2, 32
        model = MoTModel(config)
        assert tower_parameters(model, 0), name
