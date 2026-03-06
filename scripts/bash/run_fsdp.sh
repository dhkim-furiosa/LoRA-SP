#!/usr/bin/env bash

# Comma-separated list of GPUs to use
DEVICES=4,5,6,7

# Environment variables -----------------------------------------------------
export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false
# FSDP communication bucket size (MB) — adjust if needed
export TORCH_DISTRIBUTED_BUCKET_CAP_MB=10

# Experiment settings -------------------------------------------------------
BASELINE="pi0"
DATA_ROOT_DIR="piper_multitask"

# ---------------------------------------------------------------------------
# Launch FSDP distributed training
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --master-port=29350 \
    --nproc_per_node=4 \
    scripts/train.py \
    --policy.path=/path/to/pi0_checkpoint \
    --dist_mode=fsdp \
    --lora_cfg='{"r":32,"alpha":64}' \
    --target_keywords='["all-linear"]' \
    --train_dataset.repo_id="/path/to/data/${DATA_ROOT_DIR}" \
    --train_dataset.root="/path/to/data/${DATA_ROOT_DIR}" \
    --test_dataset.repo_id="/path/to/data/${DATA_ROOT_DIR}" \
    --test_dataset.root="/path/to/data/${DATA_ROOT_DIR}" \
    --wandb.project=LoRA-SP \
    --wandb.enable=false \
    --wandb.disable_artifact=true \
    --output_dir=/path/to/output/${BASELINE}_fsdp \
    --job_name="${BASELINE}_fsdp" \
    --batch_size=8 \
    --num_workers=4 \
    --log_freq=10 \
    --save_freq=1000 \
    --test_freq=1000 \
    --steps=30000
