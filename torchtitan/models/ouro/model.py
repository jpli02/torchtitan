# Adapted from vLLM Ouro model (https://github.com/vllm-project/vllm)
#
# Inference-only Ouro model compatible with HuggingFace weights.
# Implements looped Transformer with total_ut_steps.

from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType, GQAttention
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.tools.logging import logger


def _rmsnorm_with_residual(
    x: torch.Tensor,
    norm: nn.Module,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RMSNorm with Pre-norm style.
    
    When residual is None: return norm(x), x
    When residual is not None: return norm(x + residual), x + residual
    """
    if residual is not None:
        x = x + residual
    output = norm(x)
    return output, x


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
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-attention norm with residual
        h_norm, residual = _rmsnorm_with_residual(x, self.input_layernorm, residual)
        h = x + self.attention(h_norm, freqs_cis, attention_masks, positions)
        h = self.input_layernorm_2(h)

        # Pre-MLP norm with residual
        h_norm, residual = _rmsnorm_with_residual(
            h, self.post_attention_layernorm, residual
        )
        h = h + self.feed_forward(h_norm)
        h = self.post_attention_layernorm_2(h)

        return h, residual

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
        layer: TransformerBlock.Config

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
        self.early_exit_gate = nn.Linear(config.dim, 1, bias=True)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for _ in range(self.total_ut_steps):
            residual = None
            for layer in self.layers.values():
                h, residual = layer(
                    h, self.freqs_cis, attention_masks, positions, residual=residual
                )

            h, _ = _rmsnorm_with_residual(h, self.norm, residual)

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
