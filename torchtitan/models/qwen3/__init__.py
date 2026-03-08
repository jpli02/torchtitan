# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, FeedForward, GQAttention, RoPE
from torchtitan.models.common.moe import MoE
from torchtitan.protocols.model_spec import ModelSpec

from .model import Qwen3Model, Qwen3TransformerBlock
from .parallelize import parallelize_qwen3
from .state_dict_adapter import Qwen3StateDictAdapter

__all__ = [
    "parallelize_qwen3",
    "Qwen3Model",
    "qwen3_configs",
    "build_custom_qwen3_spec",
]


def build_custom_qwen3_spec(
    *,
    dim: int = 1024,
    n_layers: int = 28,
    n_heads: int = 16,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    ffn_hidden_dim: int | None = None,
    vocab_size: int = 151936,
    norm_eps: float = 1e-6,
    max_seq_len: int = 4096,
    enable_weight_tying: bool = True,
) -> ModelSpec:
    """Build a Qwen3 ModelSpec with user-defined architecture hyperparameters.

    Based on Qwen3-0.6B architecture. Use this for custom model sizes and experiments.

    Args:
        dim: Model dimension (hidden size).
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads for queries.
        n_kv_heads: Number of key/value heads (for grouped-query attention).
        head_dim: Dimension per attention head.
        ffn_hidden_dim: FFN intermediate dimension. Defaults to 3*dim if None.
        vocab_size: Vocabulary size (use 151936 for Qwen tokenizer).
        norm_eps: Epsilon for RMSNorm.
        max_seq_len: Maximum sequence length for RoPE.
        enable_weight_tying: Tie input and output embeddings.

    Returns:
        ModelSpec ready for training.
    """
    if ffn_hidden_dim is None:
        ffn_hidden_dim = 3 * dim

    model_config = Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        enable_weight_tying=enable_weight_tying,
        tok_embeddings=Embedding.Config(),
        layer=Qwen3TransformerBlock.Config(
            norm_eps=norm_eps,
            feed_forward=FeedForward.Config(hidden_dim=ffn_hidden_dim),
            attention=GQAttention.Config(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                qk_norm=True,
                norm_eps=norm_eps,
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=max_seq_len,
            theta=1000000.0,
            backend="cos_sin",
        ),
    )

    return ModelSpec(
        name="qwen3",
        flavor="custom",
        model=model_config,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3StateDictAdapter,
    )

# Adding different variants of the model

qwen3_configs = {
    "debugmodel": Qwen3Model.Config(
        vocab_size=2048,
        dim=256,
        n_layers=8,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(),
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            feed_forward=FeedForward.Config(hidden_dim=3072),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    "0.6B": Qwen3Model.Config(
        vocab_size=151936,
        dim=1024,
        n_layers=28,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(),
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            feed_forward=FeedForward.Config(hidden_dim=3072),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    "1.7B": Qwen3Model.Config(
        vocab_size=151936,
        dim=2048,
        n_layers=28,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(),
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            feed_forward=FeedForward.Config(hidden_dim=6144),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    "4B": Qwen3Model.Config(
        vocab_size=151936,
        dim=2560,
        n_layers=36,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(),
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            feed_forward=FeedForward.Config(hidden_dim=9728),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    "8B": Qwen3Model.Config(
        vocab_size=151936,
        dim=4096,
        n_layers=36,
        tok_embeddings=Embedding.Config(),
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            feed_forward=FeedForward.Config(hidden_dim=12288),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    "14B": Qwen3Model.Config(
        vocab_size=151936,
        dim=5120,
        n_layers=40,
        tok_embeddings=Embedding.Config(),
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            feed_forward=FeedForward.Config(hidden_dim=17408),
            attention=GQAttention.Config(
                n_heads=40,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    "32B": Qwen3Model.Config(
        vocab_size=151936,
        dim=5120,
        n_layers=64,
        tok_embeddings=Embedding.Config(),
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            feed_forward=FeedForward.Config(hidden_dim=25600),
            attention=GQAttention.Config(
                n_heads=64,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    # Qwen3-MoE models
    "debugmodel_moe": Qwen3Model.Config(
        vocab_size=2048,
        dim=256,
        n_layers=8,
        tok_embeddings=Embedding.Config(),
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            moe_enabled=True,
            moe=MoE.Config(
                hidden_dim=768,
                num_experts=64,
                num_shared_experts=0,
                top_k=8,
                score_func="softmax",
                route_norm=True,
                route_scale=1.0,
                score_before_experts=False,
            ),
            feed_forward=FeedForward.Config(hidden_dim=3072),
            attention=GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    "30B-A3B": Qwen3Model.Config(
        vocab_size=151936,
        dim=2048,
        n_layers=48,
        tok_embeddings=Embedding.Config(),
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            moe_enabled=True,
            moe=MoE.Config(
                hidden_dim=768,
                num_experts=128,
                num_shared_experts=0,
                top_k=8,
                score_func="softmax",
                route_norm=True,
                route_scale=1.0,
                score_before_experts=False,
            ),
            feed_forward=FeedForward.Config(hidden_dim=6144),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=4,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=262144,
            theta=1000000.0,
            backend="cos_sin",
        ),
    ),
    "235B-A22B": Qwen3Model.Config(
        vocab_size=151936,
        dim=4096,
        n_layers=94,
        tok_embeddings=Embedding.Config(),
        layer=Qwen3TransformerBlock.Config(
            norm_eps=1e-6,
            moe_enabled=True,
            moe=MoE.Config(
                hidden_dim=1536,
                num_experts=128,
                num_shared_experts=0,
                top_k=8,
                score_func="softmax",
                route_norm=True,
                route_scale=1.0,
                score_before_experts=False,
            ),
            feed_forward=FeedForward.Config(hidden_dim=12288),
            attention=GQAttention.Config(
                n_heads=64,
                n_kv_heads=4,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                rope_backend="cos_sin",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=5000000.0,
            backend="cos_sin",
        ),
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="qwen3",
        flavor=flavor,
        model=qwen3_configs[flavor],
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
