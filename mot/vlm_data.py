"""The three multimodal corpora VILA compares, plus the text corpus underneath them.

The whole comparison turns on one difference, and it is a difference in the
*text*, not in the images:

  pairs        every document is a caption and its image. The only text the
               model ever sees during multimodal training is a caption. This is
               the COYO analogue, the one that costs 17.2% of text accuracy.

  interleaved  a document is prose, then a captioned image, then more prose,
               then another captioned image. The captions still ground the
               images, so the cross-modal dependency this repo measures is
               intact -- but the model also keeps seeing the text distribution
               it was pretrained on. The MMC4 analogue, the one that costs ~5%.

  blend        pairs, with text-only documents mixed back in. VILA's mitigation,
               applied during pretraining rather than only at instruction tuning.

Segment type is tracked per *token*, not per document, so a caption-image pair
sitting inside an interleaved document is still labelled `PAIRED_TI` and still
scored by `conditional_losses`. Without that the interleaved regime would look
like it had no cross-modal structure at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mot.data import PAIRED_IT, PAIRED_TI, TEXT_ONLY, Batch, Scene, sample_scene
from mot.text import TextMix, association_tokens, statement_tokens
from mot.vocab import DOC, MODALITY_OF

REGIME_NAMES = ("text_only", "pairs", "interleaved", "blend")


@dataclass(frozen=True)
class Regime:
    """What a training stage samples from.

    `text_prob` is the share of documents that are text-only; `interleave_prob`
    the share of the remainder that interleave prose with their captioned
    images rather than being a bare pair.
    """

    name: str
    text_prob: float = 0.0
    interleave_prob: float = 0.0
    segments: int = 3               # prose blocks per interleaved document
    mix: TextMix = TextMix()

    @property
    def is_text_only(self) -> bool:
        return self.text_prob >= 1.0


REGIMES: dict[str, Regime] = {
    "text_only": Regime("text_only", text_prob=1.0),
    "pairs": Regime("pairs", text_prob=0.0, interleave_prob=0.0),
    "interleaved": Regime("interleaved", text_prob=0.0, interleave_prob=1.0),
    "blend": Regime("blend", text_prob=0.25, interleave_prob=0.0),
}


def caption_and_image(scene: Scene, text_first: bool) -> tuple[list[int], list[int]]:
    """One captioned image, and the segment type of each of its tokens."""
    caption, image = scene.caption_tokens(), scene.image_tokens()
    kind = PAIRED_TI if text_first else PAIRED_IT
    body = caption + image if text_first else image + caption
    return body, [kind] * len(body)


def _text_segment(rng: np.random.Generator, mix: TextMix) -> tuple[list[int], list[int]]:
    tokens = mix.sample(rng)
    return tokens, [TEXT_ONLY] * len(tokens)


def build_document(rng: np.random.Generator, regime: Regime
                   ) -> tuple[list[int], list[int]]:
    """One document: its tokens, and the segment type behind each token."""
    tokens, kinds = [DOC], [TEXT_ONLY]

    if rng.random() < regime.text_prob:
        body, kind = _text_segment(rng, regime.mix)
        return tokens + body, kinds + kind

    if rng.random() < regime.interleave_prob:
        for i in range(regime.segments):
            body, kind = _text_segment(rng, regime.mix)
            tokens += body
            kinds += kind
            if i < regime.segments - 1:
                body, kind = caption_and_image(sample_scene(rng),
                                               bool(rng.random() < 0.5))
                tokens += body
                kinds += kind
        return tokens, kinds

    body, kind = caption_and_image(sample_scene(rng), bool(rng.random() < 0.5))
    return tokens + body, kinds + kind


def _pack_stream(rng: np.random.Generator, length: int, regime: Regime
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill exactly `length` tokens with whole documents, truncating the last."""
    tokens: list[int] = []
    doc_ids: list[int] = []
    kinds: list[int] = []
    doc_id = 0
    while len(tokens) < length:
        body, kind = build_document(rng, regime)
        tokens += body
        kinds += kind
        doc_ids += [doc_id] * len(body)
        doc_id += 1
    return (np.array(tokens[:length], dtype=np.int64),
            np.array(doc_ids[:length], dtype=np.int64),
            np.array(kinds[:length], dtype=np.int64))


def _to_batch(tokens: np.ndarray, doc_ids: np.ndarray, kinds: np.ndarray) -> Batch:
    x, y = tokens[:, :-1], tokens[:, 1:]
    return Batch(
        x=torch.from_numpy(x.copy()),
        y=torch.from_numpy(y.copy()),
        mod_x=torch.from_numpy(MODALITY_OF[x]),
        mod_y=torch.from_numpy(MODALITY_OF[y]),
        doc_id=torch.from_numpy(doc_ids[:, :-1].copy()),
        doc_type_x=torch.from_numpy(kinds[:, :-1].copy()),
        doc_type_y=torch.from_numpy(kinds[:, 1:].copy()),
    )


def make_batch(rng: np.random.Generator, batch_size: int, seq_len: int,
               regime: Regime) -> Batch:
    streams = [_pack_stream(rng, seq_len + 1, regime) for _ in range(batch_size)]
    return _to_batch(np.stack([s[0] for s in streams]),
                     np.stack([s[1] for s in streams]),
                     np.stack([s[2] for s in streams]))


def token_shares(rng: np.random.Generator, regime: Regime, n_seq: int = 32,
                 seq_len: int = 256) -> dict[str, float]:
    """Measured text/image split, and how much of the text is prose rather than caption.

    The second number is the one that separates the regimes: `pairs` has none of
    it by construction, and that is the entire difference the experiment tests.
    """
    counts = np.zeros(2, dtype=np.int64)
    prose = caption = 0
    for _ in range(n_seq):
        tokens, _, kinds = _pack_stream(rng, seq_len, regime)
        counts += np.bincount(MODALITY_OF[tokens], minlength=2)
        text = MODALITY_OF[tokens] == 0
        prose += int(((kinds == TEXT_ONLY) & text).sum())
        caption += int(((kinds != TEXT_ONLY) & text).sum())
    total_text = max(prose + caption, 1)
    return {
        "text_frac": float(counts[0] / counts.sum()),
        "image_frac": float(counts[1] / counts.sum()),
        "prose_share_of_text": float(prose / total_text),
    }


# --- fixed-layout evaluation batches ---------------------------------------
# One document per row and no packing, so the position being scored is known
# exactly rather than searched for.

def statement_eval_batch(rng: np.random.Generator, n: int) -> Batch:
    rows = [[DOC] + statement_tokens(rng) for _ in range(n)]
    tokens = np.array(rows, dtype=np.int64)
    kinds = np.full_like(tokens, TEXT_ONLY)
    doc_ids = np.zeros_like(tokens)
    return _to_batch(tokens, doc_ids, kinds)


def association_eval_batch(rng: np.random.Generator, n: int, n_shots: int) -> Batch:
    rows = [[DOC] + association_tokens(rng, n_shots) for _ in range(n)]
    tokens = np.array(rows, dtype=np.int64)
    kinds = np.full_like(tokens, TEXT_ONLY)
    doc_ids = np.zeros_like(tokens)
    return _to_batch(tokens, doc_ids, kinds)
