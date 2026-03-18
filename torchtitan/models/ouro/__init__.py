# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Ouro model - Iterative refinement (Unified Transformer) architecture

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, FeedForward, GQAttention, RMSNorm, RoPE
from torchtitan.protocols.model_spec import ModelSpec

from .model import OuroModel, OuroTransformerBlock
from .parallelize import parallelize_ouro
from .state_dict_adapter import OuroStateDictAdapter

__all__ = [
    "parallelize_ouro",
    "OuroModel",
    "ouro_configs",
]

ouro_configs = {
    "debugmodel": OuroModel.Config(
        vocab_size=2048,
        dim=256,
        n_layers=6,
        total_ut_steps=4,
        early_exit_threshold=1.0,
        norm=RMSNorm.Config(eps=1e-6),
        tok_embeddings=Embedding.Config(),
        layer=OuroTransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6),
            ffn_norm=RMSNorm.Config(eps=1e-6),
            feed_forward=FeedForward.Config(hidden_dim=512),
            attention=GQAttention.Config(
                n_heads=8,
                n_kv_heads=4,
                head_dim=64,
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=64,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    # ByteDance/Ouro-1.4B
    "1.4B": OuroModel.Config(
        vocab_size=49152,
        dim=2048,
        n_layers=24,
        total_ut_steps=4,
        early_exit_threshold=1.0,
        norm=RMSNorm.Config(eps=1e-6),
        tok_embeddings=Embedding.Config(),
        layer=OuroTransformerBlock.Config(
            attention_norm=RMSNorm.Config(eps=1e-6),
            ffn_norm=RMSNorm.Config(eps=1e-6),
            feed_forward=FeedForward.Config(hidden_dim=5632),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=16,
                head_dim=128,
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=65536,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="ouro",
        flavor=flavor,
        model=ouro_configs[flavor],
        parallelize_fn=parallelize_ouro,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=OuroStateDictAdapter,
    )
