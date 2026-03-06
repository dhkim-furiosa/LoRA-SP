DEVICES=0
TORCH_DISTRIBUTED_BUCKET_CAP_MB=10
export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false

BASELINE="SmolVLA"
DATA_ROOT_DIR="piper_multitask_push2"

METHOD="lora_ada_r128_r60_lin"
DATASET="multitask"

CUDA_VISIBLE_DEVICES=${DEVICES} \
  torchrun \
    --nproc_per_node=1 \
    --master-port=29000 \
    scripts/train.py \
    --policy.path=/ckpt/smolvla_base \
    --dist_mode="none" \
    --method.core="lora_ada" \
    --gradient_checkpointing=false \
    --train_dataset.repo_id="/data/${DATA_ROOT_DIR}/lerobot_5hz" \
    --train_dataset.root="/data/${DATA_ROOT_DIR}/lerobot_5hz" \
    --test_dataset.repo_id="/data/${DATA_ROOT_DIR}/lerobot_5hz" \
    --test_dataset.root="/data/${DATA_ROOT_DIR}/lerobot_5hz" \
    --wandb.project=SmolVLA \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --output_dir=/result/${BASELINE}_${METHOD}_${DATASET} \
    --job_name=${BASELINE}_${METHOD}_${DATASET} \
    --batch_size=16 \
    --num_workers=16 \
    --log_freq=10 \
    --k_plot_freq=10 \
    --save_freq=5000 \
    --test_freq=100 \
    --steps=30000
