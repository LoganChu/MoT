"""Correctness tests for the staged VLM study.

Three claims carry it, and each has a test whose job is to earn it:

  * the text capabilities are real capabilities -- one only learnable from
    weights, one only answerable from context. If the association task could be
    memorised, "in-context learning degraded" would be unfalsifiable.

  * the three regimes differ in the *text* and nowhere else that matters.

  * the interventions are exact: freezing moves nothing, insulation zeroes one
    specific gradient and changes no forward pass, blocking zeroes all of them.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mot.data import PAIRED_IT, PAIRED_TI, TEXT_ONLY
from mot.model import MoTConfig, MoTModel
from mot.probes import grad_attribution
from mot.text import (
    RELATION_TABLE, association_answer_index, association_tokens,
    statement_relation_index, statement_tokens,
)
from mot.vlm_data import (
    REGIMES, association_eval_batch, make_batch, statement_eval_batch, token_shares,
)
from mot.vlm_probes import ExpertDrift, forgetting
from mot.vlm_train import (
    Stage, VLMRunConfig, expert_parameters, set_freeze, upcycle_image_expert,
)
from mot.vocab import (
    COLOR0, IMAGE, N_COLORS, N_RELATIONS, N_SHAPES, QUERY, REL0, SHAPE0, TEXT,
    VLM_VOCAB_SIZE,
)

SEQ_LEN = 128


def _config(**overrides) -> MoTConfig:
    base = dict(vocab_size=VLM_VOCAB_SIZE, n_layers=3, d_model=64, n_heads=4,
                seq_len=SEQ_LEN)
    base.update(overrides)
    return MoTConfig(**base)


def _model_and_batch(seed: int = 0, regime: str = "pairs", **overrides):
    torch.manual_seed(seed)
    model = MoTModel(_config(**overrides))
    batch = make_batch(np.random.default_rng(seed), 4, SEQ_LEN, REGIMES[regime])
    return model, batch


# --- the text capabilities --------------------------------------------------

def test_relation_is_a_deterministic_function_of_the_two_colours():
    """Otherwise the statement task is noise and cannot be 'forgotten'."""
    rng = np.random.default_rng(0)
    seen: dict[tuple[int, int], int] = {}
    for _ in range(3000):
        t = statement_tokens(rng)
        key = (t[2] - COLOR0, t[3] - COLOR0)
        relation = t[statement_relation_index()] - REL0
        assert seen.setdefault(key, relation) == relation
        assert relation == RELATION_TABLE[key[0], key[1]]


def test_statements_are_locally_distinguishable_from_associations():
    """A statement puts two colours next to each other; an association always
    puts a shape after a colour. Without that difference the model can only
    tell which task it is in by retrieving the document marker, and measured
    over 2500 steps of pretraining fact recall stalled just above chance."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        statement = statement_tokens(rng)
        assert COLOR0 <= statement[2] < COLOR0 + N_COLORS
        assert COLOR0 <= statement[3] < COLOR0 + N_COLORS
        assoc = association_tokens(rng, 2)
        assert COLOR0 <= assoc[2] < COLOR0 + N_COLORS
        assert SHAPE0 <= assoc[3] < SHAPE0 + N_SHAPES


def test_association_answer_is_present_in_the_demonstrations():
    """The k-shot task must be answerable by copying, or ICL is not measurable."""
    rng = np.random.default_rng(2)
    for n_shots in (1, 2, 4):
        for _ in range(200):
            tokens = association_tokens(rng, n_shots)
            answer_at = association_answer_index(n_shots)
            asked = tokens[answer_at - 1]
            demos = {tokens[2 + 2 * i]: tokens[3 + 2 * i] for i in range(n_shots)}
            assert asked in demos
            assert demos[asked] == tokens[answer_at]


def test_zero_shot_association_shows_no_demonstrations():
    """With no demonstrations the queried colour was never shown, so the best a
    model can do is the shape prior -- which is what makes the 0-shot number a
    floor rather than a second measurement of memorisation."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        tokens = association_tokens(rng, 0)
        assert tokens.index(QUERY) == 2, "something preceded the query"


def test_the_answer_immediately_follows_the_matched_colour():
    """Prefix-match then copy the *next* token: the canonical induction circuit.

    An earlier version put an arrow between a colour and its shape, which moved
    the copy to offset +2, and over 1200 steps of pretraining the circuit never
    formed at all -- four-shot accuracy stayed at chance. The offset is the
    whole difference, so it is pinned by a test.
    """
    rng = np.random.default_rng(11)
    for n_shots in (1, 2, 4):
        tokens = association_tokens(rng, n_shots)
        answer_at = association_answer_index(n_shots)
        asked = tokens[answer_at - 1]
        first = tokens.index(asked)
        assert tokens[first + 1] == tokens[answer_at]


def test_association_mapping_is_resampled_every_document():
    """If it were fixed, the model could memorise it and ICL would be confounded."""
    rng = np.random.default_rng(4)
    mappings: dict[int, set[int]] = {}
    for _ in range(500):
        tokens = association_tokens(rng, 4)
        for i in range(4):
            mappings.setdefault(tokens[2 + 2 * i], set()).add(tokens[3 + 2 * i])
    assert max(len(v) for v in mappings.values()) > 1


# --- the three regimes ------------------------------------------------------

def test_regimes_differ_in_how_much_of_the_text_is_prose():
    """The entire experiment is this difference, so it had better be large."""
    shares = {name: token_shares(np.random.default_rng(5), REGIMES[name],
                                 n_seq=16, seq_len=256)["prose_share_of_text"]
              for name in ("pairs", "blend", "interleaved")}
    assert shares["pairs"] < 0.2
    assert shares["pairs"] < shares["blend"] < shares["interleaved"]
    assert shares["interleaved"] > 0.5


def test_text_only_regime_contains_no_image_tokens():
    batch = make_batch(np.random.default_rng(6), 4, SEQ_LEN, REGIMES["text_only"])
    assert not bool((batch.mod_x == IMAGE).any())


def test_interleaved_documents_still_contain_captioned_images():
    """Prose must be added alongside the pairs, not instead of them, or the
    regime would differ in cross-modal structure too and prove nothing."""
    batch = make_batch(np.random.default_rng(7), 4, 256, REGIMES["interleaved"])
    kinds = batch.doc_type_x
    assert bool(((kinds == PAIRED_TI) | (kinds == PAIRED_IT)).any())
    assert bool((kinds == TEXT_ONLY).any())
    assert bool((batch.mod_x == IMAGE).any())


def test_eval_batches_score_the_intended_token():
    """The scored position is asserted, not assumed: an off-by-one here would
    silently measure the accuracy of predicting `<eot>`."""
    rng = np.random.default_rng(8)
    batch = statement_eval_batch(rng, 16)
    target = batch.y[:, statement_relation_index()]
    assert bool(((target >= REL0) & (target < REL0 + N_RELATIONS)).all())

    for n_shots in (0, 4):
        batch = association_eval_batch(rng, 16, n_shots)
        target = batch.y[:, association_answer_index(n_shots)]
        assert bool(((target >= SHAPE0) & (target < SHAPE0 + N_SHAPES)).all())


# --- the interventions ------------------------------------------------------

def test_freezing_stops_the_text_expert_and_nothing_else():
    model, batch = _model_and_batch()
    set_freeze(model, "text_expert")
    frozen = expert_parameters(model, 0)
    assert all(not p.requires_grad for p in frozen)
    assert all(p.requires_grad for p in expert_parameters(model, 1))
    assert model.embed.weight.requires_grad

    set_freeze(model, "text_expert_and_embed")
    assert not model.embed.weight.requires_grad

    set_freeze(model, "none")
    assert all(p.requires_grad for p in model.parameters())


def test_frozen_text_expert_does_not_move():
    model, batch = _model_and_batch()
    set_freeze(model, "text_expert")
    drift = ExpertDrift(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        logits, _ = model(batch.x, batch.mod_x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), batch.y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    moved = drift.drift()
    assert all(v == 0.0 for (layer, expert), v in moved.items() if expert == 0)
    assert any(v > 0.0 for (layer, expert), v in moved.items() if expert == 1)


def test_insulation_leaves_the_forward_pass_unchanged():
    torch.manual_seed(0)
    plain = MoTModel(_config())
    torch.manual_seed(0)
    cut = MoTModel(_config(insulate_modality=IMAGE))
    batch = make_batch(np.random.default_rng(0), 4, SEQ_LEN, REGIMES["pairs"])
    with torch.no_grad():
        a, _ = plain(batch.x, batch.mod_x)
        b, _ = cut(batch.x, batch.mod_x)
    assert torch.allclose(a, b, atol=1e-6)


def test_insulation_zeroes_the_image_gradient_into_the_text_expert():
    """The targeted version of freezing: the text expert still trains on text."""
    model, batch = _model_and_batch(insulate_modality=IMAGE)
    attribution = grad_attribution(model, batch)
    assert np.all(attribution[:, 0, IMAGE] == 0.0)
    assert attribution[:, 0, TEXT].min() > 0
    assert attribution[:, 1, IMAGE].min() > 0
    # The other direction is untouched: text loss still reaches the image expert.
    assert attribution[:, 1, TEXT].min() > 0


def test_blocked_attention_zeroes_both_off_diagonals():
    model, batch = _model_and_batch(blocked_attention=True)
    attribution = grad_attribution(model, batch)
    assert np.all(attribution[:, 0, IMAGE] == 0.0)
    assert np.all(attribution[:, 1, TEXT] == 0.0)
    assert attribution[:, 0, TEXT].min() > 0
    assert attribution[:, 1, IMAGE].min() > 0


def test_upcycling_makes_the_image_expert_a_copy_of_the_text_expert():
    torch.manual_seed(0)
    model = MoTModel(_config(identical_expert_init=False))
    before = [p.detach().clone() for p in expert_parameters(model, 1)]
    upcycle_image_expert(model)
    after = expert_parameters(model, 1)
    text = expert_parameters(model, 0)
    assert all(torch.equal(a, t) for a, t in zip(after, text))
    assert any(not torch.equal(a, b) for a, b in zip(after, before))


def test_drift_is_zero_at_the_reference_and_rebases():
    model, _ = _model_and_batch()
    drift = ExpertDrift(model)
    assert all(v == 0.0 for v in drift.drift().values())
    with torch.no_grad():
        for p in expert_parameters(model, 0):
            p.add_(torch.randn_like(p) * 0.01)
    assert any(v > 0.0 for v in drift.drift().values())
    drift.rebase(model)
    assert all(v == 0.0 for v in drift.drift().values())


def test_forgetting_reports_accuracy_deltas():
    before = {"statement_acc": 0.9, "assoc4_acc": 0.8, "statement_loss": 1.0}
    after = {"statement_acc": 0.7, "assoc4_acc": 0.3, "statement_loss": 2.0}
    delta = forgetting(before, after)
    assert delta == {"statement_acc": pytest.approx(-0.2),
                     "assoc4_acc": pytest.approx(-0.5)}


def test_recipe_stages_are_ordered_and_named():
    from mot.vlm_configs import VLM_RUNS

    for name, cfg in VLM_RUNS.items():
        assert [s.name for s in cfg.stages] == ["align", "pretrain", "sft"]
        # Align freezes the language model outright, embeddings included: the
        # table is tied to the output head, and leaving it trainable erased
        # fact recall on its own. The dense arm is the exception -- it has no
        # image expert, so that freeze would leave nothing to train at all.
        expected = "text_expert" if cfg.dense else "text_expert_and_embed"
        assert cfg.stages[0].freeze == expected, name
        # And the multimodal stages run below the language model's own rate.
        assert all(s.lr is not None and s.lr < cfg.text_lr for s in cfg.stages)
        assert all(s.steps > 0 for s in cfg.stages)
        cfg.model_config()


def test_every_arm_has_something_to_train_in_every_stage():
    """A freeze that leaves no trainable parameters is a silent no-op at best."""
    from mot.vlm_configs import VLM_RUNS
    from mot.vlm_train import build_optimizer, set_freeze

    for name, cfg in VLM_RUNS.items():
        config = cfg.model_config()
        config.n_layers, config.d_model = 2, 32
        model = MoTModel(config)
        for stage in cfg.stages:
            set_freeze(model, stage.freeze)
            trainable = [p for p in model.parameters() if p.requires_grad]
            assert trainable, f"{name}/{stage.name} has nothing to train"
            build_optimizer(cfg, model, stage.lr or cfg.lr)


def test_every_layout_can_be_tracked():
    """Partial and tapered decoupling leave some layers owning no expert weights.

    An expert owns only its *decoupled* sub-modules, so that a shared parameter
    -- which every modality trains -- never lands on `grad_attribution`'s
    off-diagonal. Two layouts fall out of that: a dense model, where nothing is
    decoupled and the single transformer is the expert, and a depth taper, whose
    top layers are shared outright and own nothing. Both used to divide by a
    zero norm the first time movement was measured.
    """
    from mot.train import ExpertTracker
    from mot.vlm_configs import VLM_RUNS

    for name in ("vlm_dense", "vlm_taper", "vlm_ffn_only", "vlm_share_norms",
                 "vlm_attention_only", "vlm_pairs"):
        config = VLM_RUNS[name].model_config()
        config.n_layers, config.d_model = 4, 64
        model = MoTModel(config)
        tracker = ExpertTracker(model)
        _, displacement = tracker.movement()
        assert displacement, name
        assert all(v == v and v >= 0.0 for v in displacement.values()), name

    # A dense model owns all nine sub-modules; a tapered one owns none above
    # the sharing point.
    dense = MoTModel(VLM_RUNS["vlm_dense"].model_config())
    assert len(dense.layers[0].submodules(0)) == 9
    taper = MoTModel(VLM_RUNS["vlm_taper"].model_config())
    assert len(taper.layers[0].submodules(1)) == 9
    assert taper.layers[5].submodules(1) == {}
