"""What multimodal training costs the language model underneath it.

VILA reports three numbers and this module measures all three, as accuracies
rather than losses so that "degrades by 17.2%" has a counterpart here:

  `statement_accuracy`    recall of a fact learned during text pretraining.
                          The MMLU analogue -- it can only be forgotten.
  `association_accuracy`  answering from demonstrations in the context, at zero
                          shots and at four. The in-context-learning analogue,
                          and the capability VILA finds caption-only training
                          destroys while leaving zero-shot accuracy intact.
  `expert_drift`          how far each expert has moved from the pretrained
                          checkpoint. Not a capability, a mechanism: paired with
                          `grad_attribution` it says *which objective* moved the
                          weights that used to hold the fact.

The last one is what makes this worth doing on a Mixture-of-Transformers rather
than on a dense model. The text expert never processes an image token, so the
naive expectation is that decoupling protects it. It does not have to: global
attention still delivers image-loss gradient into it, and that path is exactly
the off-diagonal the caption/image study already measures.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from mot.data import Batch
from mot.model import MoTModel
from mot.text import association_answer_index, statement_relation_index
from mot.vocab import N_RELATIONS, N_SHAPES, REL0, SHAPE0

RELATION_OPTIONS = torch.arange(REL0, REL0 + N_RELATIONS)
SHAPE_OPTIONS = torch.arange(SHAPE0, SHAPE0 + N_SHAPES)


@torch.no_grad()
def _scored(model: MoTModel, batch: Batch, index: int, options: torch.Tensor
            ) -> dict[str, float]:
    """Accuracy at one fixed target position, both multiple-choice and free.

    The headline number is the *constrained* one: the argmax taken only over the
    candidate answers, which is how MMLU -- the benchmark VILA reports these
    degradations on -- is scored. Free-form top-1 is reported beside it because
    the two answer different questions, and in a shared vocabulary they come
    apart badly. After multimodal training the loss at this position reached 18
    nats, meaning the output head had driven the answer tokens to a probability
    of about 1e-8; free-form accuracy is then zero no matter what the model
    still knows, while the constrained score still reads the knowledge.
    """
    logits, _ = model(batch.x, batch.mod_x)
    scores, targets = logits[:, index, :], batch.y[:, index]
    options = options.to(scores.device)
    chosen = options[scores[:, options].argmax(-1)]
    return {
        "acc": float((chosen == targets).float().mean()),
        "acc_free": float((scores.argmax(-1) == targets).float().mean()),
        "loss": float(F.cross_entropy(scores, targets)),
    }


@torch.no_grad()
def statement_accuracy(model: MoTModel, batch: Batch) -> dict[str, float]:
    """Recall of the hidden relation table, scored on the relation token alone.

    Chance is 1/5. Everything else in the document is either free or given, so
    scoring the whole sequence would bury the one token that carries the fact.
    """
    return _scored(model, batch, statement_relation_index(), RELATION_OPTIONS)


@torch.no_grad()
def association_accuracy(model: MoTModel, batch: Batch, n_shots: int
                         ) -> dict[str, float]:
    """Answering a fresh mapping from its demonstrations. Chance is 1/6."""
    return _scored(model, batch, association_answer_index(n_shots), SHAPE_OPTIONS)


@torch.no_grad()
def text_capability(model: MoTModel, evals: dict) -> dict[str, float]:
    """The full text-side report card, on fixed held-out batches."""
    out: dict[str, float] = {}
    for key, value in statement_accuracy(model, evals["statement"]).items():
        out[f"statement_{key}"] = value

    for n_shots, batch in evals["association"].items():
        for key, value in association_accuracy(model, batch, n_shots).items():
            out[f"assoc{n_shots}_{key}"] = value

    shots = sorted(evals["association"])
    if len(shots) >= 2:
        # The in-context-learning gap: what the demonstrations are worth. VILA
        # reports a model whose four-shot accuracy is *below* its zero-shot,
        # which is a negative value here.
        out["icl_gain"] = out[f"assoc{shots[-1]}_acc"] - out[f"assoc{shots[0]}_acc"]
    return out


class ExpertDrift:
    """Distance of each expert from a reference checkpoint, per layer.

    Snapshot the model after text pretraining and this reads out, at any later
    point, how far multimodal training has dragged the language model away from
    what it knew -- separately for the expert that sees text and the one that
    never does.
    """

    def __init__(self, model: MoTModel):
        self.grid = model.expert_params()
        self.reference = {k: [p.detach().clone() for p in v]
                          for k, v in self.grid.items()}
        self.scale = {
            k: float(np.sqrt(sum(float(p.pow(2).sum()) for p in v)))
            for k, v in self.reference.items()
        }

    @torch.no_grad()
    def rebase(self, model: MoTModel) -> None:
        """Make the model's current weights the new reference."""
        for key, params in self.grid.items():
            for stored, live in zip(self.reference[key], params):
                stored.copy_(live)
            self.scale[key] = float(
                np.sqrt(sum(float(p.pow(2).sum()) for p in params)))

    @torch.no_grad()
    def drift(self) -> dict[tuple[int, int], float]:
        out: dict[tuple[int, int], float] = {}
        for key, params in self.grid.items():
            moved = sum(float((p - q).pow(2).sum())
                        for p, q in zip(params, self.reference[key]))
            out[key] = float(np.sqrt(moved)) / max(self.scale[key], 1e-12)
        return out


def forgetting(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """Change in each text capability, in accuracy points.

    Negative is forgetting. Reported as a delta rather than a ratio because a
    capability that starts near chance has no meaningful ratio.
    """
    return {k: after[k] - before[k] for k in before if k.endswith("_acc")}
