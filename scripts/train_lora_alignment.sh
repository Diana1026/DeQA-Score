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
    echo "MODEL_PATH=zhiyuanyou/DeQA-Score-Mix3 sh scripts/train_lora_alignment.sh 0 --max_steps 50"
    exit 1
fi

DEEPSPEED_BIN="deepspeed"
if [ -x "./.venv/bin/deepspeed" ]; then
    DEEPSPEED_BIN="./.venv/bin/deepspeed"
fi

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
    --weight_diversity 0.01 \
    --decomp_fusion_type gated_sum \
    --align_loss_type mse \
    --data_paths ./data/AGIQA3K/metas/train_alignment.json \
    --data_weights 1 \
    --image_folder ./ \
    --output_dir ./checkpoints/deqa_lora_alignment_decomp \
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
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --tune_visual_abstractor True \
    --freeze_vision_model False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    "$@"
