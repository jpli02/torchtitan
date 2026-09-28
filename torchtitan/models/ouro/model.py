# Adapted from vLLM Ouro model (https://github.com/vllm-project/vllm)
#
# Inference-only Ouro model compatible with HuggingFace weights.
# Implements looped Transformer with total_ut_steps.

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType, GQAttention
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.tools.logging import logger


def build_router(dim: int, spec: dict) -> nn.Module:
    """Construct the early-exit gate/router from an architecture spec (Tier-1 NAS).

    spec keys (all optional):
        layers (int): number of hidden layers, 0 => plain linear
        hidden (int): hidden width
        act (str):    'relu' | 'gelu' | 'silu' | 'tanh'
        norm (bool):  LayerNorm on the input features
    Maps [..., dim] -> [..., 1]. An empty spec reproduces the original
    ``nn.Linear(dim, 1)`` exactly, so default behaviour is unchanged.
    """
    layers = int(spec.get("layers", 0))
    hidden = int(spec.get("hidden", 128))
    act = str(spec.get("act", "gelu")).lower()
    norm = bool(spec.get("norm", False))
    if layers <= 0 and not norm:
        return nn.Linear(dim, 1, bias=True)
    acts = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
    if act not in acts:
        raise ValueError(f"router spec: unknown act {act!r}; choose {list(acts)}")
    mods: list[nn.Module] = []
    if norm:
        mods.append(nn.LayerNorm(dim))
    d = dim
    for _ in range(layers):
        mods.append(nn.Linear(d, hidden))
        mods.append(acts[act]())
        d = hidden
    mods.append(nn.Linear(d, 1))
    return nn.Sequential(*mods)


def router_spec_from_env() -> dict:
    """Read the router architecture spec (JSON) from ``OURO_ROUTER_SPEC``."""
    raw = os.environ.get("OURO_ROUTER_SPEC")
    if not raw:
        return {}
    try:
        spec = json.loads(raw)
        return spec if isinstance(spec, dict) else {}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"OURO_ROUTER_SPEC is not valid JSON ({e}); using default linear gate.")
        return {}


class OuroTransformerBlock(TransformerBlock):
    """
    Ouro TransformerBlock with looped support.  
    Use sandwich norm style.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        depth_init: bool = True

    def __init__(self, config: Config, *, layer_id: int, dim: int, n_layers: int):
        super().__init__()

        self.moe_enabled = False  # Dense model; required by apply_fsdp from llama4
        self.attention = config.attention.build(dim=dim)
        assert config.feed_forward is not None
        self.feed_forward = config.feed_forward.build(dim=dim)

        self.input_layernorm = config.attention_norm.build(normalized_shape=dim)
        self.input_layernorm_2 = config.ffn_norm.build(normalized_shape=dim)
        self.post_attention_layernorm = config.attention_norm.build(normalized_shape=dim)
        self.post_attention_layernorm_2 = config.ffn_norm.build(normalized_shape=dim)

        if config.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Mirror HF Ouro block semantics:
        # residual add happens after attention/mlp output is normalized.
        residual = x
        h = self.input_layernorm(x)
        h = self.attention(h, freqs_cis, attention_masks, positions)
        h = self.input_layernorm_2(h)
        h = residual + h

        residual = h
        h = self.post_attention_layernorm(h)
        h = self.feed_forward(h)
        h = self.post_attention_layernorm_2(h)
        h = residual + h

        return h

    def init_weights(self, **kwargs):
        buffer_device: torch.device | None = kwargs.get("buffer_device")
        for norm in (
            self.input_layernorm,
            self.input_layernorm_2,
            self.post_attention_layernorm,
            self.post_attention_layernorm_2,
        ):
            norm.init_weights()
        self.attention.init_weights(self.weight_init_std, buffer_device=buffer_device)
        self.feed_forward.init_weights(self.weight_init_std, buffer_device=buffer_device)


class OuroModel(Decoder):
    """
    Ouro model: looped Transformer
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        total_ut_steps: int = 4
        early_exit_threshold: float = 1.0
        early_exit_step: int | None = None
        layer: TransformerBlock.Config
        # LoopLM-style training objectives (see torchtitan.components.loss).
        ouro_loss_stage: Literal["standard", "stage1_entropy", "stage2_adaptive"] = (
            "stage1_entropy"
        )
        """``standard``: CE on expected logits; ``stage1_entropy``: Eq. (4); ``stage2_adaptive``: Eq. (6)."""
        entropy_beta: float = 0.01
        """Entropy regularization weight for Stage I (beta in Eq. (4))."""
        adaptive_k: float = 50.0
        adaptive_gamma: float = 0.005
        """Sigmoid slope and threshold for ideal continuation labels in Stage II."""

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum "
                    f"{self.rope.max_seq_len}."
                )
            import dataclasses as _dc

            self.rope = _dc.replace(self.rope, max_seq_len=seq_len)

            if (
                parallelism.context_parallel_degree > 1
                and self.layer.attention.attn_backend == "varlen"
            ):
                raise NotImplementedError(
                    "Context Parallel only supports SDPA and FlexAttention. "
                    f"Got attn_backend='{self.layer.attention.attn_backend}'. "
                    "Varlen attention is not supported with CP."
                )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_dense_model_nparams_and_flops(
                self,
                model,
                self.layer.attention.n_heads,
                2 * self.layer.attention.head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.total_ut_steps = config.total_ut_steps
        self.early_exit_threshold = config.early_exit_threshold
        self.early_exit_step = config.early_exit_step
        self.ouro_loss_stage = config.ouro_loss_stage
        _router_spec = router_spec_from_env()
        self.early_exit_gate = build_router(config.dim, _router_spec)
        if _router_spec:
            logger.info(
                f"OURO_ROUTER_SPEC={_router_spec}: early_exit_gate architecture = "
                f"{self.early_exit_gate}"
            )
        if config.ouro_loss_stage == "stage2_adaptive":
            for name, p in self.named_parameters():
                if "early_exit_gate" not in name:
                    p.requires_grad_(False)
        # Inference efficiency instrumentation: running sum/count of the number
        # of UT loops (recurrence steps) the adaptive early-exit path uses for
        # the *last* token position of each forward, i.e. loops per generated
        # token. Populated only by _adaptive_forward (inference; threshold < 1.0
        # or fixed early_exit_step), never by the training forward() path. Read
        # and reset per eval by scripts/evaluate_humaneval.py.
        self._loop_sum: float = 0.0
        self._loop_count: int = 0
        # Diagnostic: how many tokens the (now-removed) threshold-gather would
        # have scored from a pre-final UT step at threshold>=1.0 (see forward()).
        self._gather_early_exits: int = 0
        self._gather_total: int = 0

    def reset_loop_stats(self) -> None:
        """Zero the loops-per-token accumulators (call before an eval pass)."""
        self._loop_sum = 0.0
        self._loop_count = 0

    @property
    def avg_loops(self) -> float | None:
        """Mean UT loops per generated token since the last reset, or None."""
        if self._loop_count == 0:
            return None
        return self._loop_sum / self._loop_count

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        def _ut_step(hidden: torch.Tensor) -> torch.Tensor:
            """Run one universal-transformer step: all layers then final norm.
            The normed output feeds the next step (the recurrence is over the
            normed hidden state), matching the training loop below."""
            for layer in self.layers.values():
                hidden = layer(hidden, self.freqs_cis, attention_masks, positions)
            return self.norm(hidden)

        # Adaptive inference (early_exit_step or threshold < 1.0) terminates the
        # UT loop as soon as every token position has exited, giving real
        # compute/latency savings.  Training and the threshold==1.0 fallback keep
        # the original semantics (always run all total_ut_steps).
        adaptive_eval = (
            not self.training
            and self.output is not None
            and (
                self.early_exit_step is not None
                or (
                    self.early_exit_threshold is not None
                    and self.early_exit_threshold < 1.0
                )
            )
        )
        if adaptive_eval:
            return self._adaptive_forward(h, _ut_step)

        hidden_states_list: list[torch.Tensor] = []
        gate_list: list[torch.Tensor] = []

        for _ in range(self.total_ut_steps):
            h = _ut_step(h)
            hidden_states_list.append(h)
            gate_list.append(self.early_exit_gate(h))

        if self.output is None:
            return h

        # Build per-token probability mass function over UT exit steps.
        # Shapes:
        # - gate tensors: [batch, seq, 1]
        # - stacked_exit_pdf: [batch, seq, total_ut_steps]
        # Compute exit PDF in float32 to avoid bfloat16 saturation (sigmoid can
        # return exactly 1.0 in bfloat16 for moderate gate values, driving
        # remaining_prob to 0 and causing inf gradients through entr).
        pdf_list: list[torch.Tensor] = []
        remaining_prob = torch.ones(
            gate_list[0].squeeze(-1).shape,
            dtype=torch.float32,
            device=gate_list[0].device,
        )
        for idx, gate_tensor in enumerate(gate_list):
            lambda_i = torch.sigmoid(gate_tensor.squeeze(-1).float())
            if idx < len(gate_list) - 1:
                p_i = lambda_i * remaining_prob
                remaining_prob = remaining_prob * (1.0 - lambda_i)
            else:
                p_i = remaining_prob
            pdf_list.append(p_i)
        stacked_exit_pdf = torch.stack(pdf_list, dim=2)

        # During training, use expected logits over all UT steps so gradients
        # propagate through every refinement step.
        if self.training:
            step_logits_list: list[torch.Tensor] = []
            expected_logits: torch.Tensor | None = None
            for step_idx, hidden in enumerate(hidden_states_list):
                step_logits = self.output(hidden)
                step_logits_list.append(step_logits)
                weight = stacked_exit_pdf[..., step_idx].unsqueeze(-1).to(
                    step_logits.dtype
                )
                expected_logits = (
                    step_logits * weight
                    if expected_logits is None
                    else expected_logits + step_logits * weight
                )
            assert expected_logits is not None
            if self.ouro_loss_stage == "standard":
                return expected_logits
            stacked_step_logits = torch.stack(step_logits_list, dim=-1)
            gate_lambda = torch.stack(
                [torch.sigmoid(g.squeeze(-1).float()) for g in gate_list], dim=-1
            )
            out: dict[str, Any] = {
                "logits": expected_logits,
                "stacked_exit_pdf": stacked_exit_pdf,
                "stacked_step_logits": stacked_step_logits,
                "gate_lambda": gate_lambda,
                "total_ut_steps": self.total_ut_steps,
            }
            return out

        # Non-adaptive eval (threshold >= 1.0 => full recurrence): score from the
        # FINAL UT step, exactly like the HF/vLLM reference. The trained exit gate
        # is not consulted here.  (Previously a threshold-gather picked the first
        # step whose cumulative exit-PDF >= threshold; at threshold==1.0 that is
        # *meant* to be the last step, but a gate saturating to 1.0 in fp32 makes
        # the cumsum reach 1.0 early and silently scores a token from a less-
        # refined step -- a divergence from HF.  The counters below record how
        # often that would have happened so we can quantify the prior bug.)
        if self.early_exit_threshold is not None:
            cumulative_probs = torch.cumsum(stacked_exit_pdf, dim=2)
            threshold_mask = cumulative_probs >= self.early_exit_threshold
            exit_steps = torch.argmax(threshold_mask.float(), dim=2)
            last_step_idx = stacked_exit_pdf.shape[2] - 1
            never_exceeded = ~threshold_mask.any(dim=2)
            exit_steps[never_exceeded] = last_step_idx
            self._gather_early_exits += int((exit_steps != last_step_idx).sum())
            self._gather_total += int(exit_steps.numel())
            return self.output(hidden_states_list[-1])

        output = self.output(h)
        return output

    def _adaptive_forward(self, h: torch.Tensor, ut_step) -> torch.Tensor:
        """Inference with real early termination of the UT loop.

        Per token position, snapshot the hidden state at the step where it first
        exits, then break the loop once all positions have exited so the
        remaining (expensive) UT steps are skipped entirely.  Output is
        equivalent to the gather-based threshold path in ``forward`` but avoids
        computing steps no position needs.
        """
        # Fixed-step exit: run only up to the requested step, then stop.
        if self.early_exit_step is not None:
            step = max(0, min(self.early_exit_step, self.total_ut_steps - 1))
            for _ in range(step + 1):
                h = ut_step(h)
            # Fixed exit: every position (incl. the last/generated one) runs
            # step+1 loops. Record one sample per batch element.
            self._loop_sum += float(step + 1) * h.shape[0]
            self._loop_count += int(h.shape[0])
            return self.output(h)

        threshold = self.early_exit_threshold
        exit_hidden: torch.Tensor | None = None
        exited: torch.Tensor | None = None
        cumulative: torch.Tensor | None = None
        remaining: torch.Tensor | None = None
        # Per-position UT step index at which each token exited (0-based); loops
        # used == index + 1. Defaults to the last step for any position that
        # never crosses the threshold (the is_last branch guarantees it does).
        exit_step_idx: torch.Tensor | None = None

        for idx in range(self.total_ut_steps):
            h = ut_step(h)
            if exit_hidden is None:
                exit_hidden = torch.zeros_like(h)
                exited = torch.zeros(
                    h.shape[:-1], dtype=torch.bool, device=h.device
                )
                cumulative = torch.zeros(
                    h.shape[:-1], dtype=torch.float32, device=h.device
                )
                remaining = torch.ones_like(cumulative)
                exit_step_idx = torch.full(
                    h.shape[:-1],
                    self.total_ut_steps - 1,
                    dtype=torch.long,
                    device=h.device,
                )

            # Same stick-breaking exit PDF as forward(): lambda_i is the
            # conditional exit prob at step i; the last step takes all remaining
            # mass so every position is guaranteed to have exited by the end.
            lambda_i = torch.sigmoid(self.early_exit_gate(h).squeeze(-1).float())
            is_last = idx == self.total_ut_steps - 1
            p_i = remaining if is_last else lambda_i * remaining
            remaining = remaining * (1.0 - lambda_i)
            cumulative = cumulative + p_i

            newly = (~exited) & ((cumulative >= threshold) | is_last)
            exit_hidden = torch.where(newly.unsqueeze(-1), h, exit_hidden)
            exit_step_idx = torch.where(
                newly, torch.full_like(exit_step_idx, idx), exit_step_idx
            )
            exited = exited | newly
            if bool(exited.all()):
                break

        # Loops for the last token position (the one whose logits generate the
        # next token), one sample per batch element.
        last = exit_step_idx[..., -1]
        self._loop_sum += float((last + 1).sum().item())
        self._loop_count += int(last.numel())

        return self.output(exit_hidden)

    def init_weights(
        self,
        *,
        buffer_device: torch.device | None = None,
        **kwargs,
    ):
        super().init_weights(buffer_device=buffer_device, **kwargs)
        # Zero the gate's final linear so the exit probability starts neutral
        # (sigmoid(0)=0.5). Works for the plain Linear gate and any configurable
        # router (OURO_ROUTER_SPEC); intermediate router layers keep default init.
        gate = self.early_exit_gate
        if isinstance(gate, nn.Linear):
            last_linear = gate
        else:
            last_linear = None
            for module in gate.modules():
                if isinstance(module, nn.Linear):
                    last_linear = module
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)
