"""Procedural text+image corpus with a genuine two-way cross-modal dependency.

A scene is a 3x3 grid of anchor blocks, 1-3 of which hold a coloured shape.
It is expressed two ways:

  caption : <bot> (color shape position)* <eot>        ~8 tokens
  image   : <boi> (patch code) x 9 <eoi>               11 tokens

The mapping is bijective, so in a `caption -> image` document the image tokens
are *only* predictable by attending back to the caption, and vice versa. That is
what makes emergent cross-modal attention a real measurement rather than noise:
block cross-modal attention and the loss on those positions must get worse.

Patch codes are a VQ-style tokenization -- one code per 3x3 block. `render_scene`
expands them back to a 9x9 pixel grid for display.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mot.vocab import (
    BOI, BOT, COLOR0, DOC, EOI, EOT, IMG0, MODALITY_OF, N_ANCHORS, N_COLORS,
    N_SHAPES, POS0, SHAPE0, SHAPES, image_code,
)

# Document types.
PAIRED_TI, PAIRED_IT, TEXT_ONLY = 0, 1, 2
DOC_TYPE_NAMES = ("paired_ti", "paired_it", "text_only")

MIN_OBJECTS, MAX_OBJECTS = 1, 3

# 3x3 stencils, used only to render patch codes back to pixels for display.
STENCILS = {
    "square":   ((1, 1, 1), (1, 1, 1), (1, 1, 1)),
    "circle":   ((0, 1, 0), (1, 1, 1), (0, 1, 0)),
    "triangle": ((0, 1, 0), (0, 1, 0), (1, 1, 1)),
    "cross":    ((1, 0, 1), (0, 1, 0), (1, 0, 1)),
    "bar":      ((0, 0, 0), (1, 1, 1), (0, 0, 0)),
    "dot":      ((0, 0, 0), (0, 1, 0), (0, 0, 0)),
}


@dataclass(frozen=True)
class Scene:
    """Nine anchor blocks, each empty (None) or a (shape_idx, color_idx) pair."""

    cells: tuple[tuple[int, int] | None, ...]

    def caption_tokens(self) -> list[int]:
        out = [BOT]
        for anchor, cell in enumerate(self.cells):
            if cell is None:
                continue
            shape_idx, color_idx = cell
            out += [COLOR0 + color_idx, SHAPE0 + shape_idx, POS0 + anchor]
        out.append(EOT)
        return out

    def image_tokens(self) -> list[int]:
        codes = [image_code(*cell) if cell else image_code(None, None)
                 for cell in self.cells]
        return [BOI] + [IMG0 + c for c in codes] + [EOI]


def sample_scene(rng: np.random.Generator) -> Scene:
    n_obj = int(rng.integers(MIN_OBJECTS, MAX_OBJECTS + 1))
    anchors = rng.choice(N_ANCHORS, size=n_obj, replace=False)
    cells: list[tuple[int, int] | None] = [None] * N_ANCHORS
    for anchor in anchors:
        cells[int(anchor)] = (int(rng.integers(N_SHAPES)), int(rng.integers(N_COLORS)))
    return Scene(cells=tuple(cells))


def render_scene(scene: Scene) -> np.ndarray:
    """Expand to a 9x9 grid of colour indices (-1 = background). Display only."""
    grid = np.full((9, 9), -1, dtype=np.int64)
    for anchor, cell in enumerate(scene.cells):
        if cell is None:
            continue
        shape_idx, color_idx = cell
        stencil = STENCILS[SHAPES[shape_idx]]
        row0, col0 = (anchor // 3) * 3, (anchor % 3) * 3
        for r in range(3):
            for c in range(3):
                if stencil[r][c]:
                    grid[row0 + r, col0 + c] = color_idx
    return grid


def build_document(scene: Scene, doc_type: int) -> list[int]:
    """A document, opened by its separator so every token belongs to a doc."""
    caption, image = scene.caption_tokens(), scene.image_tokens()
    if doc_type == PAIRED_TI:
        body = caption + image
    elif doc_type == PAIRED_IT:
        body = image + caption
    elif doc_type == TEXT_ONLY:
        body = caption
    else:
        raise ValueError(f"unknown doc_type {doc_type}")
    return [DOC] + body


def solve_text_only_prob(target_text_frac: float) -> float:
    """Probability of emitting a text-only doc that hits a target token ratio.

    Every document contributes the same expected text length; only paired docs
    add image tokens, so the ratio is monotone in p and inverts in closed form.
    """
    text_per_doc = 1 + 2 + 3 * (MIN_OBJECTS + MAX_OBJECTS) / 2   # <doc> <bot> .. <eot>
    image_per_doc = 2 + N_ANCHORS                                # <boi> .. <eoi>
    floor = text_per_doc / (text_per_doc + image_per_doc)
    if not floor <= target_text_frac < 1.0:
        raise ValueError(
            f"target_text_frac={target_text_frac} unreachable; this corpus supports "
            f"[{floor:.3f}, 1.0). Below the floor there are not enough text tokens "
            f"even with zero text-only documents."
        )
    p = 1.0 - text_per_doc * (1.0 - target_text_frac) / (image_per_doc * target_text_frac)
    return float(np.clip(p, 0.0, 0.99))


def _pack_stream(
    rng: np.random.Generator, length: int, p_text_only: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill exactly `length` tokens with whole documents, truncating the last."""
    tokens: list[int] = []
    doc_ids: list[int] = []
    doc_types: list[int] = []
    doc_id = 0
    while len(tokens) < length:
        if rng.random() < p_text_only:
            doc_type = TEXT_ONLY
        else:
            doc_type = PAIRED_TI if rng.random() < 0.5 else PAIRED_IT
        doc = build_document(sample_scene(rng), doc_type)
        tokens += doc
        doc_ids += [doc_id] * len(doc)
        doc_types += [doc_type] * len(doc)
        doc_id += 1
    return (
        np.array(tokens[:length], dtype=np.int64),
        np.array(doc_ids[:length], dtype=np.int64),
        np.array(doc_types[:length], dtype=np.int64),
    )


@dataclass(frozen=True)
class Batch:
    """One training batch. `x`/`y` are the usual shifted LM pair.

    `mod_x` routes inputs to experts; `mod_y` attributes loss to a modality;
    `doc_type_y` lets us score, say, image loss on caption->image documents only.
    """

    x: torch.Tensor           # (B, T) input ids
    y: torch.Tensor           # (B, T) target ids
    mod_x: torch.Tensor       # (B, T) modality of each input token
    mod_y: torch.Tensor       # (B, T) modality of each target token
    doc_id: torch.Tensor      # (B, T) document index, aligned to inputs
    doc_type_x: torch.Tensor  # (B, T) document type, aligned to inputs
    doc_type_y: torch.Tensor  # (B, T) document type, aligned to targets

    def to(self, device: torch.device) -> "Batch":
        return Batch(*(t.to(device) for t in (
            self.x, self.y, self.mod_x, self.mod_y, self.doc_id,
            self.doc_type_x, self.doc_type_y)))


def make_batch(
    rng: np.random.Generator, batch_size: int, seq_len: int, p_text_only: float
) -> Batch:
    streams = [_pack_stream(rng, seq_len + 1, p_text_only) for _ in range(batch_size)]
    tokens = np.stack([s[0] for s in streams])
    doc_ids = np.stack([s[1] for s in streams])
    doc_types = np.stack([s[2] for s in streams])

    x, y = tokens[:, :-1], tokens[:, 1:]
    return Batch(
        x=torch.from_numpy(x),
        y=torch.from_numpy(y),
        mod_x=torch.from_numpy(MODALITY_OF[x]),
        mod_y=torch.from_numpy(MODALITY_OF[y]),
        doc_id=torch.from_numpy(doc_ids[:, :-1].copy()),
        doc_type_x=torch.from_numpy(doc_types[:, :-1].copy()),
        doc_type_y=torch.from_numpy(doc_types[:, 1:].copy()),
    )


def measure_token_ratio(
    rng: np.random.Generator, p_text_only: float, n_seq: int = 64, seq_len: int = 256
) -> float:
    """Empirical text-token fraction, reported so the configured ratio is checked."""
    counts = np.zeros(2, dtype=np.int64)
    for _ in range(n_seq):
        tokens, _, _ = _pack_stream(rng, seq_len, p_text_only)
        mods = MODALITY_OF[tokens]
        counts += np.bincount(mods, minlength=2)
    return float(counts[0] / counts.sum())
