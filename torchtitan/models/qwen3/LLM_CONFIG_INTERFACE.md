# Qwen3 Custom Model Configuration Interface

**Purpose**: This document defines a YAML schema for configuring a custom Qwen3 model architecture and training. Give this to an LLM to generate valid configuration files.

---

## Schema

Generate a YAML file with the following structure. All fields under `model` map to `build_custom_qwen3_spec()`. Omitted fields use defaults.

```yaml
# Required: model architecture (Qwen3-based, user-defined)
model:
  dim: int              # Hidden size. Default: 1024. Examples: 512, 1024, 2048
  n_layers: int         # Number of transformer layers. Default: 28
  n_heads: int          # Number of query attention heads. Default: 16
  n_kv_heads: int       # Number of key/value heads (GQA). Must divide n_heads. Default: 8
  head_dim: int         # Dimension per head. Typically 64 or 128. Default: 128
  ffn_hidden_dim: int   # FFN intermediate size. Often 3*dim. Optional, defaults to 3*dim
  vocab_size: int       # Vocabulary size. Use 151936 for Qwen tokenizer. Default: 151936
  max_seq_len: int      # Maximum sequence length for RoPE. Default: 4096
  norm_eps: float       # RMSNorm epsilon. Default: 1e-6
  enable_weight_tying: bool  # Tie input/output embeddings. Default: true

# Required: training
training:
  local_batch_size: int   # Per-device batch size. Default: 4
  seq_len: int            # Sequence length. Must be <= model.max_seq_len. Default: 2048
  steps: int              # Training steps. Default: 1000

# Required: dataset name
dataset: str  # One of: "gsm8k", "gsm8k_validation", "c4", "c4_test", "c4_validation"

# Optional: optimizer (defaults shown)
optimizer:
  lr: float           # Learning rate. Default: 3e-4
  weight_decay: float  # Default: 0.1

# Optional: LR scheduler (defaults shown)
lr_scheduler:
  warmup_steps: int     # Default: 100
  decay_ratio: float     # Portion of steps for decay. Default: 0.1
  decay_type: str       # "linear" | "sqrt" | "cosine". Default: "cosine"
  min_lr_factor: float  # Min LR as fraction of base. Default: 0.1

# Optional: paths
hf_assets_path: str   # Path to Qwen tokenizer. Default: "./assets/hf/Qwen3-0.6B"
dump_folder: str      # Output folder. Default: "./outputs"

# Optional: checkpoint
checkpoint:
  interval: int       # Save every N steps. Default: 100
  export_dtype: str   # "float16" | "bfloat16" | "float32". Default: "float16"
```

---

## Constraints

- `n_kv_heads` must divide `n_heads` (e.g., n_heads=16, n_kv_heads=8).
- `head_dim` is typically 64 or 128.
- `training.seq_len` must not exceed `model.max_seq_len`.
- `dataset` must be one of the supported names.

---

## Example: Small model for GSM8K

```yaml
model:
  dim: 512
  n_layers: 12
  n_heads: 8
  n_kv_heads: 4
  head_dim: 64
  ffn_hidden_dim: 1536
  vocab_size: 151936
  max_seq_len: 1024

training:
  local_batch_size: 8
  seq_len: 1024
  steps: 2000

dataset: gsm8k

optimizer:
  lr: 3e-4

lr_scheduler:
  warmup_steps: 100
  decay_ratio: 0.1
  decay_type: cosine
  min_lr_factor: 0.1

hf_assets_path: ./assets/hf/Qwen3-0.6B
```

---

## Example: Qwen3-0.6B equivalent on GSM8K

```yaml
model:
  dim: 1024
  n_layers: 28
  n_heads: 16
  n_kv_heads: 8
  head_dim: 128
  ffn_hidden_dim: 3072
  vocab_size: 151936
  max_seq_len: 2048

training:
  local_batch_size: 4
  seq_len: 2048
  steps: 5000

dataset: gsm8k

hf_assets_path: ./assets/hf/Qwen3-0.6B
```

---

## Usage

**Prerequisites**: `pip install pyyaml` (required for YAML parsing)

After generating a YAML config, run training:

```bash
# From torchtitan repo root
cd /path/to/torchtitan
torchrun --nproc_per_node=N scripts/train_qwen3_from_yaml.py --config-yaml path/to/config.yaml
```

Replace `N` with the number of GPUs (e.g., 1 for single GPU). Example with the included config:

```bash
torchrun --nproc_per_node=1 scripts/train_qwen3_from_yaml.py \
  --config-yaml torchtitan/models/qwen3/configs/example.yaml
```
