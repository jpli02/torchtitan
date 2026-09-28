#!/bin/bash
cd /home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft || exit 1
export CUDA_VISIBLE_DEVICES=${GPUS:-0,2}
export OURO_SFT_MIX=tb80sft
export OURO_SFT_LOCAL_JSONL=/home/jli199/boptim_scratch/tb80sft/train_v3.jsonl
unset OURO_INIT_FROM OURO_SFT_REQUIRE_COMPLETE OURO_SFT_MIN_CMDS
export WANDB_MODE=disabled PYTORCH_ALLOC_CONF=expandable_segments:True
# log ALL ranks (--tee 3), short timeout so a rank-1 crash surfaces fast
/home/jli199/torchtitan/.venv/bin/torchrun --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --tee 3 -m torchtitan.train \
  --module ouro --config ouro_2_6b_thinking_terminal_sft --parallelism.data_parallel_shard_degree 2 \
  --dump_folder /home/jli199/boptim_scratch/ouro_26b_fsmoke --checkpoint.folder /home/jli199/boptim_scratch/ouro_26b_fsmoke/ckpt \
  --training.steps 6 --training.seq_len 1536 --checkpoint.interval 100000 --metrics.log_freq 1 \
  --comm.init_timeout_seconds 300 --comm.train_timeout_seconds 300 2>&1
