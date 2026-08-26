"""A small but faithful Mixture-of-Transformers.

Following Liang et al. (arXiv:2411.04996) and the reference implementation, every
*non-embedding* parameter is decoupled by modality -- feed-forward networks,
the Q/K/V/O projections, the QK-norms, and the layer norms. Embeddings, the
output head, and crucially the attention operation itself stay global: after
per-modality projection the Q/K/V are re-packed into sequence order and one
causal attention runs over the whole interleaved stream.

Two knobs exist purely to serve as experimental controls:

  `expert_of_modality`  collapse both modalities onto one expert -> dense baseline.
  `blocked_attention`   forbid cross-modal attention -> isolates what joint
                        attention actually contributes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

SUBMODULE_NAMES = ("attn_norm", "wq", "wk", "wv", "q_norm", "k_norm",
                   "wo", "ffn_norm", "ffn")


@dataclass(frozen=True)
class GeneratorSpec:
    """The continuous side of a two-tower model.

    Cosmos 3 pairs an autoregressive reasoner with a diffusion generator that
    shares one transformer architecture and one joint attention operator with
    it. The generator's tokens are not vocabulary entries at all -- they are
    noised latents of what is being generated -- so the model needs a second
    input path for them and a head that reads them back out. Those modules
    belong to no tower: they are the analogue of the embedding table, which MoT
    also leaves global.
    """

    latent_dim: int
    n_slots: int


@dataclass
class MoTConfig:
    vocab_size: int
    n_layers: int = 6
    d_model: int = 128
    n_heads: int = 4
    n_modalities: int = 2
    expert_of_modality: tuple[int, ...] = (0, 1)
    ffn_mult: float = 8 / 3
    seq_len: int = 256
    rope_theta: float = 10_000.0
    init_std: float = 0.02
    tie_embeddings: bool = True
    blocked_attention: bool = False
    identical_expert_init: bool = True

    # Queries of this modality read the rest of the stream through a detached
    # key/value, so their loss cannot train any other modality's expert. The
    # forward pass is unchanged -- only the gradient path is cut. Setting it to
    # IMAGE asks the question freezing cannot: is the language model degraded by
    # the image objective *reaching it through attention*, as opposed to by
    # anything else multimodal training does?
    insulate_modality: int | None = None

    # --- two-tower (Cosmos 3) settings -------------------------------------
    # `generator` switches on the continuous path; `generator_modality` says
    # which modality id those tokens carry, so routing sends them to their own
    # tower. `tower_attention` installs the mask the architecture implies:
    # the reasoner is causal and never reads the generator -- which is what
    # lets it be called on its own -- while the generator's noisy tokens see
    # each other in full and read the whole reasoner.
    generator: GeneratorSpec | None = None
    generator_modality: int | None = None
    tower_attention: bool = False

    # --- partial and tapered decoupling -----------------------------------
    # MoT decouples every non-embedding parameter. Neither of these is required
    # by the architecture, and this study's own divergence numbers argue against
    # both: the big projections pull far apart (wq 1.31, wk 1.28) while the
    # layer norms barely differentiate at all (0.08-0.11), and alignment between
    # the modalities recovers in the top layers after collapsing in the middle.
    #
    # `decoupled_submodules` names which sub-modules get one copy per modality;
    # everything else gets a single shared copy. `shared_from_layer` shares
    # *every* sub-module from that layer upwards, which is the depth taper.
    decoupled_submodules: tuple[str, ...] = SUBMODULE_NAMES
    shared_from_layer: int | None = None

    def decoupled_here(self, layer: int, name: str) -> bool:
        if self.shared_from_layer is not None and layer >= self.shared_from_layer:
            return False
        return name in self.decoupled_submodules

    @property
    def n_experts(self) -> int:
        return max(self.expert_of_modality) + 1

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")
        return self.d_model // self.n_heads

    @property
    def ffn_hidden(self) -> int:
        # Round to a multiple of 8, as Llama-style SwiGLU implementations do.
        return int(math.ceil(self.d_model * self.ffn_mult / 8) * 8)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def expert_indices(expert_idx: torch.Tensor, n_experts: int) -> list[torch.Tensor]:
    """Row indices belonging to each expert.

    The modality of every token is fixed for the whole forward pass, so this is
    computed once per forward and reused by all ~50 routing calls rather than
    recomputed inside each one.
    """
    return [torch.nonzero(expert_idx == e, as_tuple=True)[0] for e in range(n_experts)]


def route(
    x: torch.Tensor,
    experts: nn.ModuleList,
    indices: list[torch.Tensor],
) -> torch.Tensor:
    """Gather rows per expert, apply, scatter back into original sequence order.

    `x` is (N, ...) with N = B*T. This is the deterministic modality routing from
    the reference implementation -- no learned router.
    """
    if len(experts) == 1 or len(indices) == 1:
        # A shared sub-module, or a dense model: every row takes the same path.
        return experts[0](x)
    out = torch.zeros_like(x)
    for expert, idx in zip(experts, indices):
        if idx.numel() == 0:
            continue
        out = out.index_copy(0, idx, expert(x.index_select(0, idx)))
    return out


def route_rmsnorm(
    x: torch.Tensor,
    norms: nn.ModuleList,
    expert_idx: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-modality RMSNorm without any gather/scatter.

    RMSNorm is `normalize(x) * weight`, and `normalize` has no parameters, so it
    is identical across experts. Only the weight is modality-specific -- gathering
    a small weight vector per token is far cheaper than routing the activations,
    and the result is exactly equal to `route(x, norms, ...)`.
    """
    normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    if len(norms) == 1:
        return normed.type_as(x) * norms[0].weight
    weights = torch.stack([n.weight for n in norms])          # (E, dim)
    gathered = weights.index_select(0, expert_idx)             # (N, dim)
    if normed.dim() == 3:                                      # (N, heads, dim)
        gathered = gathered.unsqueeze(1)
    return normed.type_as(x) * gathered


def build_rope_cache(seq_len: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x is (B, T, H, head_dim); cos/sin are (T, head_dim/2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class MoTBlock(nn.Module):
    def __init__(self, cfg: MoTConfig, layer_index: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_index = layer_index
        n_e, d, hd = cfg.n_experts, cfg.d_model, cfg.head_dim

        def per_expert(name, fn):
            count = n_e if cfg.decoupled_here(layer_index, name) else 1
            return nn.ModuleList([fn() for _ in range(count)])

        self.attn_norm = per_expert("attn_norm", lambda: RMSNorm(d))
        self.wq = per_expert("wq", lambda: nn.Linear(d, d, bias=False))
        self.wk = per_expert("wk", lambda: nn.Linear(d, d, bias=False))
        self.wv = per_expert("wv", lambda: nn.Linear(d, d, bias=False))
        self.q_norm = per_expert("q_norm", lambda: RMSNorm(hd))
        self.k_norm = per_expert("k_norm", lambda: RMSNorm(hd))
        self.wo = per_expert("wo", lambda: nn.Linear(d, d, bias=False))
        self.ffn_norm = per_expert("ffn_norm", lambda: RMSNorm(d))
        self.ffn = per_expert("ffn", lambda: FeedForward(d, cfg.ffn_hidden))

    def is_decoupled(self, name: str) -> bool:
        return len(getattr(self, name)) > 1

    def forward(
        self,
        h: torch.Tensor,              # (B, T, d)
        expert_idx: torch.Tensor,     # (B*T,)
        indices: list[torch.Tensor],  # rows per expert, precomputed once
        rope: tuple[torch.Tensor, torch.Tensor],
        attn_allowed: torch.Tensor,   # (B|1, 1, T, T) bool, True = may attend
        need_probs: bool = False,
        insulated_query: torch.Tensor | None = None,   # (B, T) bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cfg = self.cfg
        B, T, d = h.shape
        H, hd = cfg.n_heads, cfg.head_dim
        flat = h.reshape(B * T, d)

        normed = route_rmsnorm(flat, self.attn_norm, expert_idx)
        q = route(normed, self.wq, indices).view(B * T, H, hd)
        k = route(normed, self.wk, indices).view(B * T, H, hd)
        v = route(normed, self.wv, indices).view(B * T, H, hd)

        # QK-norm is itself a per-modality expert in the reference implementation.
        q = route_rmsnorm(q, self.q_norm, expert_idx).view(B, T, H, hd)
        k = route_rmsnorm(k, self.k_norm, expert_idx).view(B, T, H, hd)
        v = v.view(B, T, H, hd)

        cos, sin = rope
        q = apply_rope(q, cos[:T], sin[:T]).transpose(1, 2)   # (B, H, T, hd)
        k = apply_rope(k, cos[:T], sin[:T]).transpose(1, 2)
        v = v.transpose(1, 2)

        # --- the one global operation: attention over the full mixed sequence ---
        probs = None
        if need_probs:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            scores = scores.masked_fill(~attn_allowed, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            attn = probs @ v
        elif insulated_query is not None:
            # The insulated queries read exactly the same keys and values -- the
            # forward pass is bit-identical -- but read them detached, so no
            # gradient from the loss at those positions travels into any other
            # expert. Doing it at every layer closes every path, because
            # attention is the only route between modalities.
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_allowed)
            cut = F.scaled_dot_product_attention(
                q, k.detach(), v.detach(), attn_mask=attn_allowed)
            attn = torch.where(insulated_query[:, None, :, None], cut, attn)
        else:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_allowed)

        attn = attn.transpose(1, 2).reshape(B * T, d)
        flat = flat + route(attn, self.wo, indices)
        flat = flat + route(route_rmsnorm(flat, self.ffn_norm, expert_idx),
                            self.ffn, indices)
        return flat.view(B, T, d), probs

    def submodules(self, expert: int) -> dict[str, nn.Module]:
        """The sub-modules that belong to one expert alone.

        A shared sub-module belongs to no expert and is deliberately absent. It
        receives gradient from every modality, so counting it as part of an
        expert would put mass on `grad_attribution`'s off-diagonal for a reason
        that has nothing to do with attention -- and would break the
        blocked-attention control, which has to come out at exactly zero.
        """
        if self.cfg.n_experts == 1:
            # Nothing is decoupled because there is nothing to decouple from,
            # and with a single expert there is no off-diagonal to protect --
            # the one transformer is the expert.
            return {name: getattr(self, name)[0] for name in SUBMODULE_NAMES}
        return {name: getattr(self, name)[expert]
                for name in SUBMODULE_NAMES if self.is_decoupled(name)}

    def shared_submodules(self) -> dict[str, nn.Module]:
        return {name: getattr(self, name)[0]
                for name in SUBMODULE_NAMES if not self.is_decoupled(name)}


class MoTModel(nn.Module):
    def __init__(self, cfg: MoTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            [MoTBlock(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = nn.ModuleList(
            [RMSNorm(cfg.d_model) for _ in range(cfg.n_experts)])
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        if cfg.generator is not None:
            spec, d = cfg.generator, cfg.d_model
            self.gen_in = nn.Linear(spec.latent_dim, d)
            self.gen_out = nn.Linear(d, spec.latent_dim)
            self.gen_slot = nn.Embedding(spec.n_slots, d)
            self.time_mlp = nn.Sequential(
                nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

        cos, sin = build_rope_cache(cfg.seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer(
            "expert_lookup",
            torch.tensor(cfg.expert_of_modality, dtype=torch.long),
            persistent=False,
        )

        self.apply(self._init_weights)
        if cfg.identical_expert_init:
            self._make_experts_identical()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def _make_experts_identical(self) -> None:
        """Copy expert 0 onto every other expert, per layer.

        A measurement choice, not standard practice: it pins inter-expert
        divergence to exactly zero at step 0, so any divergence observed later is
        provably caused by training rather than by initialisation noise.
        """
        with torch.no_grad():
            groups = [getattr(layer, name)
                      for layer in self.layers for name in SUBMODULE_NAMES]
            groups.append(self.final_norm)
            for group in groups:
                reference = group[0].state_dict()
                for expert in list(group)[1:]:
                    expert.load_state_dict(reference)

    def timestep_embedding(self, flow_t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal features of the diffusion time, then an MLP. (B,) -> (B, d)."""
        half = self.cfg.d_model // 2
        freqs = torch.exp(
            -math.log(10_000.0) * torch.arange(half, device=flow_t.device) / half)
        angles = flow_t[:, None].float() * freqs[None, :] * 1000.0
        return self.time_mlp(torch.cat([angles.cos(), angles.sin()], dim=-1))

    def tower_mask(self, is_generator: torch.Tensor) -> torch.Tensor:
        """(B, 1, T, T) bool: the mask a two-tower model implies.

        Causal everywhere, then two amendments. The generator's noisy tokens see
        each other in full, because they are denoised jointly and a causal order
        over them would be arbitrary. And every edge from a reasoner query to a
        generator key is removed, so the reasoner's forward pass does not depend
        on the generator at all -- which is what makes "the reasoner can be
        called independently" true of this model rather than merely asserted.
        """
        B, T = is_generator.shape
        allowed = torch.tril(
            torch.ones(T, T, dtype=torch.bool, device=is_generator.device))
        allowed = allowed.view(1, 1, T, T).expand(B, 1, T, T).clone()
        gen = is_generator[:, None, :, None] & is_generator[:, None, None, :]
        allowed |= gen
        reads_generator = (~is_generator)[:, None, :, None] &             is_generator[:, None, None, :]
        allowed &= ~reads_generator
        return allowed

    def forward(
        self,
        x: torch.Tensor,          # (B, T) token ids
        modality: torch.Tensor,   # (B, T) modality ids
        need_probs: bool = False,
        need_hidden: bool = False,
        generator: dict | None = None,
        attn_allowed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        B, T = x.shape
        expert_idx = self.expert_lookup[modality].reshape(B * T)
        indices = expert_indices(expert_idx, self.cfg.n_experts)

        if attn_allowed is not None:
            allowed = attn_allowed
        elif self.cfg.tower_attention and generator is not None:
            allowed = self.tower_mask(generator["is_generator"])
        else:
            allowed = torch.tril(
                torch.ones(T, T, dtype=torch.bool, device=x.device)).view(1, 1, T, T)
        if self.cfg.blocked_attention:
            same = (modality[:, :, None] == modality[:, None, :]).unsqueeze(1)
            allowed = allowed & same

        insulated = None
        if self.cfg.insulate_modality is not None:
            insulated = modality == self.cfg.insulate_modality

        h = self.embed(x)
        if generator is not None:
            slots = self.gen_slot.weight[None, :generator["noisy"].shape[1], :]
            time = self.timestep_embedding(generator["flow_t"])[:, None, :]
            injected = torch.zeros_like(h)
            injected[generator["is_generator"]] = (
                self.gen_in(generator["noisy"]) + slots + time
            ).reshape(-1, self.cfg.d_model)
            h = h + injected
        probs, hidden = [], []
        for layer in self.layers:
            h, p = layer(h, expert_idx, indices, (self.rope_cos, self.rope_sin),
                         allowed, need_probs, insulated)
            if need_probs:
                probs.append(p)
            if need_hidden:
                hidden.append(h)

        h = route_rmsnorm(h.reshape(B * T, self.cfg.d_model), self.final_norm, expert_idx)
        h = h.view(B, T, self.cfg.d_model)
        logits = self.lm_head(h)

        aux: dict = {"probs": probs, "hidden": hidden}
        if generator is not None:
            slots = h[generator["is_generator"]].view(B, -1, self.cfg.d_model)
            aux["velocity"] = self.gen_out(slots)
        return logits, aux

    # --- parameter bookkeeping used by the probes ---------------------------

    def expert_params(self) -> dict[tuple[int, int], list[nn.Parameter]]:
        """(layer, expert) -> its parameters. The unit the study treats as 'a transformer'."""
        out: dict[tuple[int, int], list[nn.Parameter]] = {}
        for layer_idx, layer in enumerate(self.layers):
            for expert in range(self.cfg.n_experts):
                params: list[nn.Parameter] = []
                for module in layer.submodules(expert).values():
                    params += list(module.parameters())
                out[(layer_idx, expert)] = params
        return out

    def expert_submodule_params(
        self, layer_idx: int, expert: int
    ) -> dict[str, list[nn.Parameter]]:
        subs = self.layers[layer_idx].submodules(expert)
        return {name: list(module.parameters()) for name, module in subs.items()}

    def param_summary(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        embed = self.embed.weight.numel()
        if not self.cfg.tie_embeddings:
            embed += self.lm_head.weight.numel()
        per_expert = sum(p.numel() for p in self.expert_params()[(0, 0)])
        return {
            "total": total,
            "embedding": embed,
            "non_embedding": total - embed,
            "per_expert_per_layer": per_expert,
        }
