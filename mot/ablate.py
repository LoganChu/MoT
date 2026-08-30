"""Reversible architecture ablations for Qwen3-VL.

Every intervention here is a runtime patch: the 4.26 GB weight file is never
rewritten and never reloaded, so each arm runs against bit-identical weights and
a difference between two arms is caused by the intervention alone.

Two mechanisms are exposed.

M-RoPE.  ``position_ids`` has shape ``(3, B, T)`` and its rows are the temporal,
height and width coordinates.  Both interventions are applied inside
``Qwen3VLTextModel.forward`` because that is the one point every caller reaches:
``generate`` supplies ``position_ids`` from ``prepare_inputs_for_generation``, so
a patch on ``compute_3d_position_ids`` is silently skipped during generation
while still appearing to work under a plain forward pass.  Text tokens already carry
``t == h == w``, and ``Qwen3VLTextRotaryEmbedding.forward`` expands a 2-D
``position_ids`` into three identical rows for the no-image path -- so forcing
``t == h == w`` *is* ordinary 1-D RoPE, by the library's own definition rather
than by our assertion.  Text-only behaviour must therefore be untouched by any
M-RoPE arm, which is the correctness check the study leans on.

DeepStack.  ``_deepstack_process`` is ``hidden_states[mask] += visual_embeds``,
so zeroing an entry is exactly equivalent to skipping it.  The entries must be
zeroed rather than dropped: the caller guards on
``layer_idx in range(len(deepstack_visual_embeds))``, so a shorter list would
silently renumber the remaining injections instead of removing one.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch


@dataclass(frozen=True)
class AblationSpec:
    """One arm of the study."""

    name: str
    description: str
    mrope: str | None = None
    mrope_section: tuple[int, int, int] | None = None
    deepstack_off: tuple[int, ...] = ()
    deepstack_shuffle: bool = False
    seed: int = 0

    def touches_text_only_path(self) -> bool:
        """True when the arm can alter a prompt that contains no image.

        Every M-RoPE arm must return False: with no image every token already
        has t == h == w, so re-deriving those rows cannot change anything.
        """
        return bool(self.deepstack_off or self.deepstack_shuffle)


def _image_token_mask(pos: torch.Tensor) -> torch.Tensor:
    """Positions whose three coordinate rows disagree, i.e. the visual tokens."""
    return (pos[1] != pos[0]) | (pos[2] != pos[0])


def _rewrite_positions(pos: torch.Tensor, mode: str, seed: int) -> torch.Tensor:
    """Return new ``(3, B, T)`` position ids under the named intervention."""
    if mode == "swap_hw":
        return pos[[0, 2, 1]].contiguous()
    if mode == "shuffle":
        # Keep the multiset of visual coordinates and destroy only their
        # assignment to tokens, so any drop is about layout and not about range.
        out = pos.clone()
        visual = _image_token_mask(pos)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for b in range(pos.shape[1]):
            idx = visual[b].nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue
            perm = idx[torch.randperm(idx.numel(), generator=gen).to(idx.device)]
            out[1, b, idx], out[2, b, idx] = pos[1, b, perm], pos[2, b, perm]
        return out
    raise ValueError(f"unknown mrope mode {mode!r}")


def _sequential_positions(cache_position, batch: int) -> torch.Tensor:
    """Plain 1-D RoPE: every token takes its absolute index on all three axes.

    ``cache_position`` is the authority rather than ``arange(T)``: during
    generation each decode step carries a single token whose true position is its
    offset in the full sequence, and ``arange`` would reset it to zero.
    """
    return cache_position.view(1, 1, -1).expand(3, batch, -1)


@contextmanager
def ablated(model, spec: AblationSpec) -> Iterator[AblationSpec]:
    """Apply ``spec`` for the duration of the block, then restore exactly."""
    inner = model.model                      # Qwen3VLModel
    lm = inner.language_model                # Qwen3VLTextModel
    undo: list = []

    if spec.mrope_section is not None:
        rot = lm.rotary_emb
        previous = rot.mrope_section
        rot.mrope_section = list(spec.mrope_section)
        undo.append(lambda: setattr(rot, "mrope_section", previous))

    if spec.mrope is not None or spec.deepstack_off or spec.deepstack_shuffle:
        original_forward = lm.forward
        had_own = "forward" in lm.__dict__

        def patched_forward(input_ids=None, attention_mask=None, position_ids=None,
                            past_key_values=None, inputs_embeds=None, use_cache=None,
                            cache_position=None, visual_pos_masks=None,
                            deepstack_visual_embeds=None, **kwargs):
            if spec.mrope is not None:
                reference = inputs_embeds if inputs_embeds is not None else input_ids
                if cache_position is None:
                    seen = past_key_values.get_seq_length() if past_key_values is not None else 0
                    cache_position = torch.arange(
                        seen, seen + reference.shape[1], device=reference.device)
                batch = reference.shape[0]
                if spec.mrope == "1d":
                    position_ids = _sequential_positions(cache_position, batch)
                else:
                    if position_ids is None:
                        position_ids = _sequential_positions(cache_position, batch)
                    elif position_ids.ndim == 2:
                        position_ids = position_ids[None].expand(3, position_ids.shape[0], -1)
                    if position_ids.shape[0] == 4:      # (text, t, h, w) variant
                        head_row, axes = position_ids[:1], position_ids[1:]
                        position_ids = torch.cat(
                            [head_row, _rewrite_positions(axes, spec.mrope, spec.seed)])
                    else:
                        position_ids = _rewrite_positions(position_ids, spec.mrope, spec.seed)

            if deepstack_visual_embeds is not None and (spec.deepstack_off or spec.deepstack_shuffle):
                gen = torch.Generator(device="cpu").manual_seed(spec.seed)
                rebuilt = []
                for i, emb in enumerate(deepstack_visual_embeds):
                    if i in spec.deepstack_off:
                        emb = torch.zeros_like(emb)   # identical to skipping the +=
                    elif spec.deepstack_shuffle:
                        perm = torch.randperm(emb.shape[0], generator=gen).to(emb.device)
                        emb = emb[perm]
                    rebuilt.append(emb)
                deepstack_visual_embeds = rebuilt

            return original_forward(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, use_cache=use_cache,
                cache_position=cache_position, visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds, **kwargs)

        lm.forward = patched_forward
        undo.append(
            (lambda: setattr(lm, "forward", original_forward)) if had_own
            else (lambda: lm.__dict__.pop("forward", None)))

    # rope_deltas is cached across calls; a stale value would leak between arms.
    inner.rope_deltas = None
    try:
        yield spec
    finally:
        for restore in reversed(undo):
            restore()
        inner.rope_deltas = None
