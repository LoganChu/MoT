"""The two-tower token layout: a reasoner prefix, then generator slots.

    <doc> | <bot> rule colour caption <eot> | <boi> current frame <eoi> | <gen> x 9

Everything before the generator slots is the reasoner's, and it is an ordinary
autoregressive vision-language problem: the caption determines the frame, so the
image tokens are predictable only by attending back to the text, exactly as in
the caption/image study. That matters because it gives the reasoner a real
capability which the generator's objective can then be caught damaging.

The generator slots carry no token identity at all. Their embeddings are noised
latents of the *next* frame, injected through the model's continuous path, and
they are scored by flow matching rather than cross-entropy. That is the whole
point of the architecture: one attention operator spanning two towers whose
objectives are not even the same kind of thing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mot.data import PAIRED_TI, TEXT_ONLY, Batch
from mot.vocab import (
    BOI, BOT, COLOR0, DOC, EOI, EOT, GEN, GENERATOR, IMG0, MODALITY_OF, PAD,
    POS0, RULE0, SHAPE0, image_code,
)
from mot.world import (
    LATENT_DIM, N_SLOTS, Transition, decode_frame, encode_frame, frame_accuracy,
    sample_transition,
)

N_CAPTION_OBJECTS = 3          # fixed, so every document has the same length
TEXT_LEN = 4 + 3 * N_CAPTION_OBJECTS      # <bot> rule colour (c s p)*N <eot>
IMAGE_LEN = N_SLOTS + 2                   # <boi> codes <eoi>
SEQ_LEN = 1 + TEXT_LEN + IMAGE_LEN + N_SLOTS


def caption_tokens(transition: Transition) -> list[int]:
    """The rule, its argument, and a caption of the *current* frame.

    Padded to a fixed object count so the layout never changes length; the
    caption is what makes the current frame predictable from the text, and so
    what gives the reasoner tower something to be good at.
    """
    out = [BOT, RULE0 + transition.rule, COLOR0 + transition.argument]
    occupied = [(i, c) for i, c in enumerate(transition.scene.cells) if c]
    for anchor, (shape_idx, colour_idx) in occupied[:N_CAPTION_OBJECTS]:
        out += [COLOR0 + colour_idx, SHAPE0 + shape_idx, POS0 + anchor]
    while len(out) < TEXT_LEN - 1:
        out.append(PAD)
    return out[:TEXT_LEN - 1] + [EOT]


def frame_tokens(transition: Transition) -> list[int]:
    codes = [image_code(*cell) if cell else image_code(None, None)
             for cell in transition.scene.cells]
    return [BOI] + [IMG0 + c for c in codes] + [EOI]


@dataclass(frozen=True)
class CosmosBatch:
    """One batch of world transitions, laid out for both towers."""

    x: torch.Tensor              # (B, T) token ids; generator slots hold <gen>
    y: torch.Tensor              # (B, T) next-token targets
    mod_x: torch.Tensor          # (B, T)
    mod_y: torch.Tensor          # (B, T)
    lm_mask: torch.Tensor        # (B, T) bool: positions with a real CE target
    is_generator: torch.Tensor   # (B, T) bool
    doc_id: torch.Tensor
    doc_type_x: torch.Tensor
    doc_type_y: torch.Tensor
    future: torch.Tensor         # (B, N_SLOTS, LATENT_DIM) the target frame
    noise: torch.Tensor          # (B, N_SLOTS, LATENT_DIM)
    flow_t: torch.Tensor         # (B,)
    rule: torch.Tensor           # (B,)

    def noisy(self) -> torch.Tensor:
        """`z_tau = tau * noise + (1 - tau) * z`, the rectified-flow interpolant."""
        tau = self.flow_t[:, None, None]
        return tau * self.noise + (1.0 - tau) * self.future

    def velocity_target(self) -> torch.Tensor:
        return self.noise - self.future

    def as_lm_batch(self) -> Batch:
        """The reasoner half, in the shape the shared probes already expect."""
        return Batch(x=self.x, y=self.y, mod_x=self.mod_x, mod_y=self.mod_y,
                     doc_id=self.doc_id, doc_type_x=self.doc_type_x,
                     doc_type_y=self.doc_type_y)

    def reasoner_prefix(self) -> Batch:
        """The reasoner half with the generator slots removed entirely.

        Stage 0 pretrains a plain vision-language model on exactly this, and
        both towers are then initialised from it -- which is what Cosmos 3 does
        when it starts both towers from pretrained Qwen3-VL weights.
        """
        keep = self.x.shape[1] - N_SLOTS
        return Batch(x=self.x[:, :keep], y=self.y[:, :keep],
                     mod_x=self.mod_x[:, :keep], mod_y=self.mod_y[:, :keep],
                     doc_id=self.doc_id[:, :keep],
                     doc_type_x=self.doc_type_x[:, :keep],
                     doc_type_y=self.doc_type_y[:, :keep])

    def generator_inputs(self, noisy: torch.Tensor | None = None,
                         flow_t: torch.Tensor | None = None) -> dict:
        return {"noisy": self.noisy() if noisy is None else noisy,
                "flow_t": self.flow_t if flow_t is None else flow_t,
                "is_generator": self.is_generator}


def _row(transition: Transition) -> tuple[list[int], list[int]]:
    tokens = [DOC] + caption_tokens(transition) + frame_tokens(transition)
    kinds = [TEXT_ONLY] + [PAIRED_TI] * (TEXT_LEN + IMAGE_LEN)
    tokens += [GEN] * N_SLOTS
    kinds += [TEXT_ONLY] * N_SLOTS
    return tokens, kinds


def make_batch(rng: np.random.Generator, batch_size: int,
               flow_t: np.ndarray | None = None,
               noise: np.ndarray | None = None) -> CosmosBatch:
    """A batch of transitions.

    `flow_t` and `noise` may be pinned, which is what the probe batches do: with
    them fixed, a change in the generator's loss over training is the model
    moving rather than a different diffusion draw.
    """
    transitions = [sample_transition(rng) for _ in range(batch_size)]
    rows, kinds = zip(*[_row(t) for t in transitions])
    tokens = np.array(rows, dtype=np.int64)
    assert tokens.shape[1] == SEQ_LEN, (tokens.shape, SEQ_LEN)
    kinds = np.array(kinds, dtype=np.int64)

    y = np.concatenate([tokens[:, 1:], np.full((batch_size, 1), PAD, np.int64)], 1)
    mod_x, mod_y = MODALITY_OF[tokens], MODALITY_OF[y]
    is_generator = tokens == GEN

    # A generator slot has no token identity, so nothing may be scored on it --
    # neither as an input nor as a target.
    lm_mask = (mod_y != GENERATOR) & (mod_x != GENERATOR)
    lm_mask[:, -1] = False

    if flow_t is None:
        flow_t = rng.random(batch_size).astype(np.float32)
    if noise is None:
        noise = rng.standard_normal(
            (batch_size, N_SLOTS, LATENT_DIM)).astype(np.float32)

    t = torch.from_numpy
    return CosmosBatch(
        x=t(tokens), y=t(y), mod_x=t(mod_x), mod_y=t(mod_y),
        lm_mask=t(lm_mask), is_generator=t(is_generator),
        doc_id=t(np.zeros_like(tokens)),
        doc_type_x=t(kinds), doc_type_y=t(np.concatenate(
            [kinds[:, 1:], kinds[:, -1:]], axis=1)),
        future=t(np.stack([encode_frame(x.future) for x in transitions])),
        noise=t(np.asarray(noise, dtype=np.float32)),
        flow_t=t(np.asarray(flow_t, dtype=np.float32)),
        rule=t(np.array([x.rule for x in transitions], dtype=np.int64)),
    )


def token_shares() -> dict[str, float]:
    batch = make_batch(np.random.default_rng(0), 8)
    counts = np.bincount(batch.mod_x.reshape(-1).numpy(), minlength=3)
    return {"text": float(counts[0] / counts.sum()),
            "image": float(counts[1] / counts.sum()),
            "generator": float(counts[2] / counts.sum())}


def decode_batch(latents: torch.Tensor, batch: CosmosBatch
                 ) -> tuple[float, float]:
    """Per-cell agreement and exact-frame rate for a decoded batch of futures."""
    predicted = latents.detach().cpu().numpy().astype(np.float64)
    truth = batch.future.cpu().numpy().astype(np.float64)
    cell, exact = [], []
    for i in range(predicted.shape[0]):
        c, e = frame_accuracy(decode_frame(predicted[i]), decode_frame(truth[i]))
        cell.append(c)
        exact.append(e)
    return float(np.mean(cell)), float(np.mean(exact))
