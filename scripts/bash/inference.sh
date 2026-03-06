DEVICES=0
export PYTHONPATH=$(pwd)

BASELINE="pi0"
DATA_ROOT_DIR="ep1200_openthepot"

BASELINE_PATH=/path/to/pi0_checkpoint
ADAPTER_PATH=/path/to/finetuned_model/pretrained_model

CUDA_VISIBLE_DEVICES=${DEVICES} \
  python scripts/eval_real_time.py \
    --policy.path=${BASELINE_PATH} \
    --adapter_path=${ADAPTER_PATH} \
    --method.core="lora_msp" \
    --train_dataset.repo_id="/path/to/data/${DATA_ROOT_DIR}" \
    --train_dataset.root="/path/to/data/${DATA_ROOT_DIR}" \
    --use_devices=true \
    --target_keywords='["all-linear"]'
