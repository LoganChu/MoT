"""Muon, for the question the caption/image study left open.

That study's headline result is that weight movement is close to scale-free in
the gradient *because Adam divides by a running second moment*: with image
tokens at 14% of the stream the starved expert still moved 1.04x as far as the
fed one, and swapping in plain SGD dropped that to 0.69x. Adam and SGD are two
answers to "how much does an expert move for a given gradient". Muon is a third,
and a structurally different one: it does not rescale the gradient element by
element, it orthogonalises the update, so every direction of a weight matrix
moves by about the same amount regardless of how the gradient was distributed
across them.

Whether an expert starved of tokens is protected by that, or exposed by it, is
not something the architecture answers -- which is the point of measuring.

Only 2-D parameters are orthogonalised, which is the standard use: the
embedding table, the norms and anything 1-D fall back to AdamW, exactly as in
the reference implementation.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def newton_schulz(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7
                  ) -> torch.Tensor:
    """Approximately orthogonalise a matrix with a fixed quintic iteration.

    The coefficients are the usual tuned ones. Five iterations is enough to
    push every singular value into roughly [0.7, 1.3], which is all the update
    rule needs -- exact orthogonalisation would cost an SVD per step per matrix.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return (x.T if transposed else x).to(matrix.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum, orthogonalised. AdamW handles everything that is not a matrix.

    Two parameter groups are built for you by `build`: `use_muon=True` for the
    2-D weights, and a standard AdamW group for the rest.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5,
                 adamw_lr: float = 3e-3, adamw_betas: tuple[float, float] = (0.9, 0.95),
                 adamw_eps: float = 1e-8, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, adamw_lr=adamw_lr,
                        adamw_betas=adamw_betas, adamw_eps=adamw_eps,
                        weight_decay=weight_decay, use_muon=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            if group.get("use_muon", False):
                self._muon_group(group)
            else:
                self._adamw_group(group)
        return loss

    def _muon_group(self, group) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)
            buf = state["momentum_buffer"]
            buf.lerp_(p.grad, 1 - group["momentum"])
            update = (p.grad.lerp(buf, group["momentum"])
                      if group["nesterov"] else buf)
            update = newton_schulz(update, steps=group["ns_steps"])
            # Keeps the update size comparable across differently shaped
            # matrices, which is the whole reason to orthogonalise.
            scale = max(1.0, p.shape[-2] / p.shape[-1]) ** 0.5
            if group["weight_decay"]:
                p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(update, alpha=-group["lr"] * scale)

    def _adamw_group(self, group) -> None:
        beta1, beta2 = group["adamw_betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.lerp_(p.grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
            bias1 = 1 - beta1 ** state["step"]
            bias2 = 1 - beta2 ** state["step"]
            denom = (exp_avg_sq / bias2).sqrt().add_(group["adamw_eps"])
            if group["weight_decay"]:
                p.mul_(1 - group["adamw_lr"] * group["weight_decay"])
            p.addcdiv_(exp_avg / bias1, denom, value=-group["adamw_lr"])


def build(params, lr: float, weight_decay: float = 0.0,
          adamw_lr: float | None = None) -> Muon:
    """Split parameters into the matrix group and everything else."""
    params = [p for p in params if p.requires_grad]
    matrices = [p for p in params if p.ndim == 2]
    others = [p for p in params if p.ndim != 2]
    adamw_lr = adamw_lr if adamw_lr is not None else lr
    return Muon([
        {"params": matrices, "use_muon": True},
        {"params": others, "use_muon": False},
    ], lr=lr, adamw_lr=adamw_lr, weight_decay=weight_decay)
