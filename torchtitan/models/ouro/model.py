# Adapted from vLLM Ouro model (https://github.com/vllm-project/vllm)
#
# Inference-only Ouro model compatible with HuggingFace weights.
# Implements looped Transformer with total_ut_steps.

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from torchtitan.components.loss import IGNORE_INDEX, LoopLMLoss

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
        residual = x
        x = self.input_layernorm(x)
        x = self.attention(x, freqs_cis, attention_masks, positions)
        x = self.input_layernorm_2(x)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.feed_forward(x)
        x = self.post_attention_layernorm_2(x)
        x = residual + x

        return x

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
        early_exit_threshold: float = 1.0  # 1.0 = no early exit; lower = exit earlier
        layer: TransformerBlock.Config
        # Loop LM: expected CE + entropy (stage1) or gate-only BCE (stage2). Eval uses plain CE.
        loop_lm_loss: Literal["none", "stage1", "stage2"] = "none"
        loop_lm_beta_entropy: float = 0.05
        loop_lm_gamma_improve: float = 0.005
        loop_lm_k_improve: float = 50.0

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
        self.early_exit_gate = nn.Linear(config.dim, 1, bias=True)
        self.loop_lm_loss = config.loop_lm_loss
        if config.loop_lm_loss != "none":
            self._loop_lm = LoopLMLoss(
                beta_entropy=config.loop_lm_beta_entropy,
                gamma_improve=config.loop_lm_gamma_improve,
                k_improve=config.loop_lm_k_improve,
                ignore_index=IGNORE_INDEX,
            )
        else:
            self._loop_lm = None

    def early_exit_logits(self, h: torch.Tensor) -> torch.Tensor:
        """
        Per-position logits from ``early_exit_gate`` (shape [B, S, 1]).
        Loop LM treats ``sigmoid(early_exit_logits(h))`` as the instantaneous exit
        probability λ_t at each recurrent step.
        """
        return self.early_exit_gate(h)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        collect_loop_lm = self.training and self.loop_lm_loss != "none"
        step_logits_list: list[torch.Tensor] = []
        step_exit_logits_list: list[torch.Tensor] = []

        for step in range(self.total_ut_steps):
            for layer in self.layers.values():
                h = layer(h, self.freqs_cis, attention_masks, positions)

            h = self.norm(h)

            if collect_loop_lm:
                step_exit_logits_list.append(self.early_exit_gate(h))
                assert self.output is not None
                step_logits_list.append(self.output(h))
            elif step < self.total_ut_steps - 1 and self.early_exit_threshold < 1.0:
                stop_logit = self.early_exit_gate(h)  # (B, S, 1)
                stop_prob = torch.sigmoid(stop_logit[:, -1, :]).mean()
                if stop_prob > self.early_exit_threshold:
                    break

        if collect_loop_lm:
            self._ouro_loop_lm_cache = (step_logits_list, step_exit_logits_list)
            return step_logits_list[-1]

        output = self.output(h) if self.output is not None else h
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
