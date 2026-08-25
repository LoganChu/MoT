"""The text-only corpus the language model is pretrained on, before it sees pixels.

VILA's central pre-training results are all statements about what multimodal
training *costs* a language model, so reproducing them needs a language model
with something to lose. This module supplies two capabilities, chosen because
each one degrades in a different way and each can be scored as an accuracy
rather than only as a loss:

  statements   `<bot> <stmt> c1 c2 REL <eot>` -- the relation is a fixed
               function of the two colours, drawn once from a hidden table the
               model can only memorise. Recall of a fact learned in pretraining,
               so this is the MMLU analogue: it can be forgotten.

  associations `<bot> <assoc> c s  c s  <query> c s <eot>` -- the
               colour-to-shape mapping is resampled for *every document*, so it
               is unlearnable from weights and answerable only by attending to
               the demonstrations in context. Accuracy at zero shots versus four
               is the in-context-learning analogue, which VILA finds is the
               capability that caption-only pretraining destroys.

Both live in the same token space as the captions, which is what makes the
comparison sharp: nothing about the *tokens* differs between a caption corpus
and this one, only the structure over them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mot.vocab import (
    ASSOC, BOT, COLOR0, EOT, N_COLORS, N_RELATIONS, N_SHAPES, QUERY, REL0,
    SHAPE0, STMT,
)  # noqa: F401 -- N_SHAPES/SHAPE0 are used by the association task

# Colours adjacent, and no shapes: a statement is locally distinguishable from
# an association, which puts a shape after every colour. An earlier version
# wrote statements as `c1 s1 c2 s2 REL`, which is the *same* local bigram
# pattern an association has, so the model could only tell the two apart by
# retrieving the document marker -- and measured over 2500 steps, fact recall
# stalled at 0.27 against a chance of 0.20.
STATEMENT_LEN = 6            # <bot> <stmt> c1 c2 REL <eot>
ASSOC_FIXED_LEN = 6          # <bot> <assoc> ... <query> c s <eot>
ASSOC_DEMO_LEN = 2           # c s

# A pair is written with the shape *immediately* after its colour, so answering
# is prefix-match-and-copy-the-next-token: the canonical induction circuit. An
# earlier version separated them with an arrow token, which put the copy at
# offset +2 and -- measured over 1200 steps -- the circuit simply never formed.

# The hidden fact table. Sampled once from a fixed seed and never resampled, so
# "the model knows the relation between red and blue" is a property of the
# weights and of nothing else -- which is what makes losing it forgetting.
RELATION_TABLE = np.random.default_rng(1234).integers(
    0, N_RELATIONS, size=(N_COLORS, N_COLORS))


def statement_tokens(rng: np.random.Generator) -> list[int]:
    """One factual statement. The relation is determined by the two colours."""
    c1, c2 = (int(x) for x in rng.integers(0, N_COLORS, size=2))
    relation = int(RELATION_TABLE[c1, c2])
    return [BOT, STMT, COLOR0 + c1, COLOR0 + c2, REL0 + relation, EOT]


def association_tokens(rng: np.random.Generator, n_shots: int) -> list[int]:
    """One in-context association task with `n_shots` demonstrations.

    The mapping is fresh per document and the query colour is always one that
    was demonstrated, so a model that has learned to look things up in its
    context answers perfectly and a model that has not cannot do better than the
    shape prior -- no matter how much it memorised during pretraining.
    """
    colours = rng.choice(N_COLORS, size=max(n_shots, 1) + 1, replace=False)
    shapes = rng.integers(0, N_SHAPES, size=len(colours))

    out = [BOT, ASSOC]
    for i in range(n_shots):
        out += [COLOR0 + int(colours[i]), SHAPE0 + int(shapes[i])]

    # With demonstrations, ask about one of them; with none, ask about an
    # unseen colour, which is the honest zero-shot case.
    ask = int(rng.integers(n_shots)) if n_shots else len(colours) - 1
    out += [QUERY, COLOR0 + int(colours[ask]), SHAPE0 + int(shapes[ask]), EOT]
    return out


def association_length(n_shots: int) -> int:
    return ASSOC_FIXED_LEN + ASSOC_DEMO_LEN * n_shots


def association_answer_index(n_shots: int) -> int:
    """Index of the answer token inside the document -- the shape after `->`.

    It is the second-to-last token, `<eot>` being last. In a shifted
    language-model pair the same token is the *target* at index one lower, which
    is what `association_answer_target_index` gives.
    """
    return association_length(n_shots) - 2


def association_answer_target_index(n_shots: int) -> int:
    return association_answer_index(n_shots) - 1


@dataclass(frozen=True)
class TextMix:
    """How much of the text corpus is each kind of document."""

    # 0.4 with a 3000-step budget is the configuration that reliably produces a
    # language model with perfect fact recall -- it reaches 1.000 by step 900
    # and holds. The capability arrives as a phase transition rather than a
    # smooth climb, and it is sensitive: an even split over 2000 steps left it
    # at 0.223, barely above the chance of 0.200. Change this and check the
    # curve `ensure_text_base` prints before trusting anything downstream.
    statement_prob: float = 0.4
    max_shots: int = 4

    def sample(self, rng: np.random.Generator) -> list[int]:
        if rng.random() < self.statement_prob:
            return statement_tokens(rng)
        return association_tokens(rng, int(rng.integers(0, self.max_shots + 1)))


def statement_relation_index() -> int:
    """Index of the relation token inside a statement document."""
    return STATEMENT_LEN - 2


def statement_relation_target_index() -> int:
    return statement_relation_index() - 1
