"""A tiny world, and the rule that moves it forward.

Cosmos 3's generator tower predicts *future observations* conditioned on what
the reasoner understood about the present. Reproducing that needs a world with
a next state, and the next state has to be determined by the reasoner's two
inputs together and by neither alone -- otherwise the generator could ignore
half of what it is conditioned on and the cross-tower measurements would have
nothing to measure.

So: a 3x3 board of coloured shapes, and an instruction naming one of a handful
of transformations. The instruction says *what happens*; the board says *what it
happens to*. The next frame is a deterministic function of the pair, and the
tests assert that a rule alone and a board alone each leave it undetermined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mot.data import Scene
from mot.vocab import N_ANCHORS, N_COLORS, N_SHAPES

# Transformations. Deliberately all *rearrangements or recolourings* of what is
# already there, so the future frame is never predictable from the rule alone --
# every rule maps different boards to different places.
RULES = ("shift_right", "shift_down", "transpose", "recolour", "mirror")
N_RULES = len(RULES)
RECOLOUR = RULES.index("recolour")

# Capped at the number of objects a caption can name. With more objects than
# that the caption stops determining the frame, the reasoner's cross-modal
# objective becomes unlearnable, and `image_given_text` sits at the same value
# as the unconditional image loss -- which is exactly what it did at four.
MIN_OBJECTS, MAX_OBJECTS = 2, 3


def sample_scene(rng: np.random.Generator) -> Scene:
    n_obj = int(rng.integers(MIN_OBJECTS, MAX_OBJECTS + 1))
    anchors = rng.choice(N_ANCHORS, size=n_obj, replace=False)
    cells: list[tuple[int, int] | None] = [None] * N_ANCHORS
    for anchor in anchors:
        cells[int(anchor)] = (int(rng.integers(N_SHAPES)), int(rng.integers(N_COLORS)))
    return Scene(cells=tuple(cells))


def _permute(cells: tuple, order: list[int]) -> tuple:
    return tuple(cells[i] for i in order)


def apply_rule(scene: Scene, rule: int, argument: int) -> Scene:
    """The next frame. Deterministic given (scene, rule, argument)."""
    cells = scene.cells
    name = RULES[rule]
    if name == "shift_right":
        order = [(row * 3) + ((col - 1) % 3) for row in range(3) for col in range(3)]
        return Scene(cells=_permute(cells, order))
    if name == "shift_down":
        order = [(((row - 1) % 3) * 3) + col for row in range(3) for col in range(3)]
        return Scene(cells=_permute(cells, order))
    if name == "transpose":
        order = [col * 3 + row for row in range(3) for col in range(3)]
        return Scene(cells=_permute(cells, order))
    if name == "mirror":
        order = [row * 3 + (2 - col) for row in range(3) for col in range(3)]
        return Scene(cells=_permute(cells, order))
    if name == "recolour":
        return Scene(cells=tuple(None if c is None else (c[0], argument)
                                 for c in cells))
    raise ValueError(f"unknown rule {name!r}")


@dataclass(frozen=True)
class Transition:
    """One (frame, instruction) pair and the frame it determines."""

    scene: Scene
    rule: int
    argument: int          # the colour, for `recolour`; ignored otherwise
    future: Scene

    @property
    def uses_argument(self) -> bool:
        return self.rule == RECOLOUR


def sample_transition(rng: np.random.Generator) -> Transition:
    scene = sample_scene(rng)
    rule = int(rng.integers(N_RULES))
    argument = int(rng.integers(N_COLORS))
    return Transition(scene=scene, rule=rule, argument=argument,
                      future=apply_rule(scene, rule, argument))


# --- the generator's latent space -------------------------------------------
# A frame is nine cells; each cell becomes a short continuous vector so the
# generator tower can denoise it the way Cosmos 3's does, rather than emitting
# discrete tokens. Occupancy is kept separate from colour and shape so an empty
# cell is not forced to invent values for either.

LATENT_DIM = 3          # (occupied, colour, shape), each scaled to [-1, 1]
N_SLOTS = N_ANCHORS


def encode_frame(scene: Scene) -> np.ndarray:
    """(N_SLOTS, LATENT_DIM) in [-1, 1]."""
    out = np.zeros((N_SLOTS, LATENT_DIM), dtype=np.float32)
    for i, cell in enumerate(scene.cells):
        if cell is None:
            out[i] = (-1.0, 0.0, 0.0)
            continue
        shape_idx, colour_idx = cell
        out[i] = (1.0,
                  2.0 * colour_idx / (N_COLORS - 1) - 1.0,
                  2.0 * shape_idx / (N_SHAPES - 1) - 1.0)
    return out


def decode_frame(latent: np.ndarray) -> Scene:
    """Nearest valid frame to a decoded latent. The inverse of `encode_frame`."""
    cells: list[tuple[int, int] | None] = []
    for row in latent:
        if row[0] < 0.0:
            cells.append(None)
            continue
        colour = int(np.clip(round((row[1] + 1.0) / 2.0 * (N_COLORS - 1)),
                             0, N_COLORS - 1))
        shape = int(np.clip(round((row[2] + 1.0) / 2.0 * (N_SHAPES - 1)),
                            0, N_SHAPES - 1))
        cells.append((shape, colour))
    return Scene(cells=tuple(cells))


def frame_accuracy(predicted: Scene, truth: Scene) -> tuple[float, bool]:
    """Per-cell agreement, and whether the whole frame is exactly right.

    Reported next to the diffusion loss because a small mean squared error and
    a frame with the wrong object in it are different failures, and only one of
    them is what a world model is for.
    """
    matches = sum(p == t for p, t in zip(predicted.cells, truth.cells))
    return matches / N_SLOTS, predicted.cells == truth.cells
