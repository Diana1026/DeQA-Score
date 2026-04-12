#!/bin/bash
export PYTHONPATH=./:$PYTHONPATH

LOAD="${MODEL_PATH:-../ModelZoo/mplug-owl2-llama2-7b/}"
GPUS=$1
shift

is_local_path=0
case "$LOAD" in
    /*|./*|../*)
        is_local_path=1
        ;;
esac

if [ "$is_local_path" -eq 1 ] && [ ! -d "$LOAD" ]; then
    echo "Local model path not found: $LOAD"
    echo "If using Hugging Face repo_id, pass it directly, e.g.:"
    echo "MODEL_PATH=zhiyuanyou/DeQA-Score-Mix3 sh scripts/train_lora_alignment_yesprob.sh 0 --max_steps 50"
    exit 1
fi

DEEPSPEED_BIN="deepspeed"
if [ -x "./.venv/bin/deepspeed" ]; then
    DEEPSPEED_BIN="./.venv/bin/deepspeed"
fi
LOGGING_STEPS="${LOGGING_STEPS:-20}"
ALIGN_SOFT_KL_WEIGHT="${ALIGN_SOFT_KL_WEIGHT:-0.2}"
ALIGN_SOFT_KL_TAU="${ALIGN_SOFT_KL_TAU:-0.5}"
ALIGN_CONSISTENCY_WEIGHT="${ALIGN_CONSISTENCY_WEIGHT:-0.2}"

$DEEPSPEED_BIN --include localhost:$GPUS --master_port 6688 src/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path "$LOAD" \
    --version v1 \
    --dataset_type single \
    --softkl_loss False \
    --weight_desp 0.0 \
    --weight_next_token 0.0 \
    --weight_align 1.0 \
    --alignment_only_loss True \
    --use_decomp_embeddings True \
    --num_decomp_tokens 8 \
    --weight_decomp 1.0 \
    --weight_diversity 0.1 \
    --decomp_fusion_type gated_sum \
    --align_loss_type vqa_yesprob_bce \
    --align_yes_token yes \
    --align_no_token no \
    --align_soft_kl_weight "${ALIGN_SOFT_KL_WEIGHT}" \
    --align_soft_kl_tau "${ALIGN_SOFT_KL_TAU}" \
    --align_consistency_weight "${ALIGN_CONSISTENCY_WEIGHT}" \
    --align_rank_weight 0.0 \
    --align_rank_margin 0.0 \
    --data_paths ./data/AGIQA3K/metas/train_alignment.json \
    --data_weights 1 \
    --image_folder ./ \
    --output_dir ./checkpoints/test \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps "${LOGGING_STEPS}" \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --tune_visual_abstractor True \
    --freeze_vision_model False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    "$@"
