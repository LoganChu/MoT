"""Correctness tests for the routing, the model, and above all the probes.

The study's headline claim -- that an under-fed expert still gets trained through
cross-modal attention -- rests entirely on `grad_attribution` measuring a real
path. `test_blocked_attention_zeroes_off_diagonal` is the test that earns that.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mot.data import make_batch, solve_text_only_prob
from mot.model import (
    SUBMODULE_NAMES, MoTConfig, MoTModel, RMSNorm, expert_indices, route, route_rmsnorm,
)
from mot.probes import attention_mass, expert_divergence, grad_attribution
from mot.vocab import VOCAB_SIZE

SEQ_LEN = 128


def _batch(batch_size: int = 4, seed: int = 0):
    return make_batch(np.random.default_rng(seed), batch_size, SEQ_LEN,
                      solve_text_only_prob(0.5))


def _config(**overrides) -> MoTConfig:
    base = dict(vocab_size=VOCAB_SIZE, n_layers=3, d_model=64, n_heads=4, seq_len=SEQ_LEN)
    base.update(overrides)
    return MoTConfig(**base)


def test_route_is_identity_for_identity_experts():
    x = torch.randn(64, 5, 3)
    idx = torch.randint(0, 3, (64,))
    experts = nn.ModuleList([nn.Identity() for _ in range(3)])
    assert torch.equal(route(x, experts, expert_indices(idx, 3)), x)


def test_route_applies_the_expert_selected_by_index():
    x = torch.randn(32, 4)
    idx = torch.randint(0, 2, (32,))
    experts = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)])
    out = route(x, experts, expert_indices(idx, 2))
    for e in range(2):
        rows = idx == e
        assert torch.allclose(out[rows], experts[e](x[rows]), atol=1e-6)


def test_rmsnorm_fast_path_matches_gather_scatter_routing():
    """The norm fast path is an optimisation, so it must be numerically equal."""
    for shape in ((48, 16), (48, 4, 8)):
        x = torch.randn(*shape)
        idx = torch.randint(0, 2, (shape[0],))
        norms = nn.ModuleList([RMSNorm(shape[-1]) for _ in range(2)])
        for norm in norms:
            nn.init.normal_(norm.weight, mean=1.0, std=0.2)
        expected = route(x, norms, expert_indices(idx, 2))
        assert torch.allclose(route_rmsnorm(x, norms, idx), expected, atol=1e-6)


def test_identical_experts_match_a_single_expert_model():
    """Routing to two identical experts must equal routing to one.

    This is the equivalence check that the gather/scatter preserves order: any
    mis-scatter would permute rows and break the match.
    """
    torch.manual_seed(0)
    sparse = MoTModel(_config(expert_of_modality=(0, 1), identical_expert_init=True))
    dense = MoTModel(_config(expert_of_modality=(0, 0)))

    dense.embed.load_state_dict(sparse.embed.state_dict())
    dense.final_norm[0].load_state_dict(sparse.final_norm[0].state_dict())
    if not dense.cfg.tie_embeddings:
        dense.lm_head.load_state_dict(sparse.lm_head.state_dict())
    for layer_d, layer_s in zip(dense.layers, sparse.layers):
        for name in SUBMODULE_NAMES:
            getattr(layer_d, name)[0].load_state_dict(
                getattr(layer_s, name)[0].state_dict())

    batch = _batch()
    with torch.no_grad():
        a, _ = sparse(batch.x, batch.mod_x)
        b, _ = dense(batch.x, batch.mod_x)
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_identical_init_gives_exactly_zero_divergence():
    model = MoTModel(_config(identical_expert_init=True))
    divergence = expert_divergence(model)
    assert np.all(divergence["overall"] == 0.0)
    for name in SUBMODULE_NAMES:
        assert np.all(divergence[name] == 0.0), name


def test_independent_init_gives_nonzero_divergence():
    model = MoTModel(_config(identical_expert_init=False))
    assert np.all(expert_divergence(model)["overall"] > 0.0)


def test_attention_mass_rows_are_distributions():
    model = MoTModel(_config())
    mass = attention_mass(model, _batch())
    assert np.allclose(mass.sum(axis=2), 1.0, atol=1e-5)
    assert (mass >= 0).all()


def test_joint_attention_gives_nonzero_cross_modal_gradient():
    """With global attention, each modality's loss must reach the other's expert."""
    torch.manual_seed(0)
    model = MoTModel(_config(blocked_attention=False))
    g = grad_attribution(model, _batch())
    off_diagonal = g[:, 0, 1], g[:, 1, 0]
    assert all((x > 0).all() for x in off_diagonal), g


def test_blocked_attention_zeroes_off_diagonal():
    """The control: remove cross-modal attention and the cross path vanishes exactly.

    Not merely small -- identically zero. Any non-zero value here would mean the
    probe is picking up some other coupling and the headline result would be
    unreliable.
    """
    torch.manual_seed(0)
    model = MoTModel(_config(blocked_attention=True))
    g = grad_attribution(model, _batch())
    assert (g[:, 0, 1] == 0.0).all(), g[:, 0, 1]
    assert (g[:, 1, 0] == 0.0).all(), g[:, 1, 0]
    # The diagonal must survive, otherwise the test passes for the wrong reason.
    assert (g[:, 0, 0] > 0).all() and (g[:, 1, 1] > 0).all(), g


def test_blocked_attention_isolates_modalities_in_attention_mass():
    model = MoTModel(_config(blocked_attention=True))
    mass = attention_mass(model, _batch())
    assert np.allclose(mass[:, 0, 1], 0.0) and np.allclose(mass[:, 1, 0], 0.0)


def test_probe_leaves_no_gradient_behind():
    """A probe must not contribute to the optimiser step that follows it."""
    model = MoTModel(_config())
    grad_attribution(model, _batch())
    assert all(p.grad is None for p in model.parameters())


def test_caption_and_image_determine_each_other():
    """The task is only cross-modal if the two views are in bijection."""
    from mot.data import sample_scene

    rng = np.random.default_rng(3)
    seen: dict[tuple, tuple] = {}
    for _ in range(400):
        scene = sample_scene(rng)
        caption = tuple(scene.caption_tokens())
        image = tuple(scene.image_tokens())
        if caption in seen:
            assert seen[caption] == image, "caption maps to two different images"
        seen[caption] = image
    assert len({v for v in seen.values()}) == len(seen), "image maps to two captions"
