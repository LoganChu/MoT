"""Union vocabulary for the toy text+image corpus.

A token's modality is a deterministic function of its id, mirroring how a real
MoT pipeline hands the model a `modality_mask` alongside the token stream.
Text and image occupy disjoint id ranges, so routing never has to guess.
"""

from __future__ import annotations

import numpy as np

COLORS = ("red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta")
SHAPES = ("square", "circle", "triangle", "cross", "bar", "dot")
POSITIONS = (
    "top-left", "top-mid", "top-right",
    "mid-left", "center", "mid-right",
    "bot-left", "bot-mid", "bot-right",
)

N_COLORS = len(COLORS)
N_SHAPES = len(SHAPES)
N_ANCHORS = len(POSITIONS)

# --- special tokens ---------------------------------------------------------
PAD = 0
DOC = 1   # document separator
BOT = 2   # begin text
EOT = 3   # end text
BOI = 4   # begin image
EOI = 5   # end image

# --- id ranges --------------------------------------------------------------
COLOR0 = 6
SHAPE0 = COLOR0 + N_COLORS          # 14
POS0 = SHAPE0 + N_SHAPES            # 20
IMG0 = POS0 + N_ANCHORS             # 29

# One image code per 3x3 anchor block: 0 = empty, else 1 + shape*N_COLORS + color
N_IMAGE_CODES = 1 + N_SHAPES * N_COLORS   # 49
VOCAB_SIZE = IMG0 + N_IMAGE_CODES         # 78

# --- two-tower (Cosmos 3) ids ---------------------------------------------
# Appended above VOCAB_SIZE so the caption/image study's vocabulary -- and
# therefore its committed run logs -- are untouched.
COSMOS0 = VOCAB_SIZE                 # 78
RULE0 = COSMOS0                      # 78  world-transition rules
N_WORLD_RULES = 5
GEN = RULE0 + N_WORLD_RULES          # 83  a generator-tower slot

COSMOS_VOCAB_SIZE = GEN + 1          # 84

# --- modality ---------------------------------------------------------------
TEXT, IMAGE = 0, 1
N_MODALITIES = 2
MODALITY_NAMES = ("text", "image")

# The generator tower's slots. They carry no vocabulary meaning -- their
# embeddings are noised latents injected by the model's continuous path -- but
# they need a modality so that deterministic routing sends them to their own
# tower.
GENERATOR = 2
N_COSMOS_MODALITIES = 3
COSMOS_MODALITY_NAMES = ("text", "image", "generator")


def _build_modality_table() -> np.ndarray:
    """Map every token id to its modality. Image brackets count as image."""
    table = np.full(COSMOS_VOCAB_SIZE, TEXT, dtype=np.int64)
    table[BOI] = IMAGE
    table[EOI] = IMAGE
    table[IMG0:IMG0 + N_IMAGE_CODES] = IMAGE
    table[GEN] = GENERATOR
    return table


MODALITY_OF = _build_modality_table()
MODALITY_OF.flags.writeable = False


def image_code(shape_idx: int | None, color_idx: int | None) -> int:
    """Encode one 3x3 anchor block as a single patch code (VQ-style)."""
    if shape_idx is None or color_idx is None:
        return 0
    return 1 + shape_idx * N_COLORS + color_idx


def decode_image_code(code: int) -> tuple[int, int] | None:
    """Inverse of `image_code`; None for an empty block."""
    if code <= 0:
        return None
    code -= 1
    return code // N_COLORS, code % N_COLORS


def describe(token_id: int) -> str:
    """Human-readable name for a token id, used in figures and the artifact."""
    specials = {PAD: "<pad>", DOC: "<doc>", BOT: "<bot>",
                EOT: "<eot>", BOI: "<boi>", EOI: "<eoi>",
                GEN: "<gen>"}
    if token_id in specials:
        return specials[token_id]
    if COLOR0 <= token_id < SHAPE0:
        return COLORS[token_id - COLOR0]
    if SHAPE0 <= token_id < POS0:
        return SHAPES[token_id - SHAPE0]
    if POS0 <= token_id < IMG0:
        return POSITIONS[token_id - POS0]
    if RULE0 <= token_id < RULE0 + N_WORLD_RULES:
        from mot.world import RULES
        return RULES[token_id - RULE0]
    decoded = decode_image_code(token_id - IMG0)
    if decoded is None:
        return "img:empty"
    shape_idx, color_idx = decoded
    return f"img:{COLORS[color_idx]}-{SHAPES[shape_idx]}"
