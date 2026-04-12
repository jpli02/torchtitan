# Adapted from vLLM Ouro model (https://github.com/vllm-project/vllm)
#
# Inference-only Ouro model compatible with HuggingFace weights.
# Implements looped Transformer with total_ut_steps.

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType, GQAttention
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.tools.logging import logger


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
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


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
                2 * (self.dim // self.layer.attention.n_heads),
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.total_ut_steps = config.total_ut_steps
        self.early_exit_threshold = config.early_exit_threshold
        self.early_exit_step = config.early_exit_step
        self.ouro_loss_stage = config.ouro_loss_stage
        self.early_exit_gate = nn.Linear(config.dim, 1, bias=True)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        hidden_states_list: list[torch.Tensor] = []
        gate_list: list[torch.Tensor] = []

        for _ in range(self.total_ut_steps):
            for layer in self.layers.values():
                h = layer(h, self.freqs_cis, attention_masks, positions)

            h = self.norm(h)
            hidden_states_list.append(h)
            gate_list.append(self.early_exit_gate(h))

        if self.output is None:
            return h

        # If no UT states were collected, fallback to last hidden states.
        if not hidden_states_list:
            return self.output(h)

        # Build per-token probability mass function over UT exit steps.
        # Shapes:
        # - gate tensors: [batch, seq, 1]
        # - stacked_exit_pdf: [batch, seq, total_ut_steps]
        pdf_list: list[torch.Tensor] = []
        remaining_prob = torch.ones_like(gate_list[0].squeeze(-1))
        for idx, gate_tensor in enumerate(gate_list):
            lambda_i = torch.sigmoid(gate_tensor.squeeze(-1))
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
                [torch.sigmoid(g.squeeze(-1)) for g in gate_list], dim=-1
            )
            out: dict[str, Any] = {
                "logits": expected_logits,
                "stacked_exit_pdf": stacked_exit_pdf,
                "stacked_step_logits": stacked_step_logits,
                "gate_lambda": gate_lambda,
                "total_ut_steps": self.total_ut_steps,
            }
            return out

        # In eval/inference, either force a fixed exit step, use thresholded
        # cumulative probabilities, or fallback to the final UT step.
        if self.early_exit_step is not None:
            step = max(0, min(self.early_exit_step, len(hidden_states_list) - 1))
            return self.output(hidden_states_list[step])

        if self.early_exit_threshold is not None:
            cumulative_probs = torch.cumsum(stacked_exit_pdf, dim=2)
            threshold_mask = cumulative_probs >= self.early_exit_threshold
            exit_steps = torch.argmax(threshold_mask.float(), dim=2)
            last_step_idx = stacked_exit_pdf.shape[2] - 1
            if last_step_idx >= 0:
                never_exceeded = ~threshold_mask.any(dim=2)
                exit_steps[never_exceeded] = last_step_idx

            stacked_hidden = torch.stack(hidden_states_list, dim=2)
            gather_index = (
                exit_steps.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, 1, stacked_hidden.size(-1))
            )
            final_hidden_states = torch.gather(stacked_hidden, 2, gather_index).squeeze(
                2
            )
            return self.output(final_hidden_states)

        output = self.output(h)
        return output

    def init_weights(
        self,
        *,
        buffer_device: torch.device | None = None,
        **kwargs,
    ):
        super().init_weights(buffer_device=buffer_device, **kwargs)
        nn.init.zeros_(self.early_exit_gate.weight)
        nn.init.zeros_(self.early_exit_gate.bias)
