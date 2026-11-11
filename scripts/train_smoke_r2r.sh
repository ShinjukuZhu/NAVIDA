#!/bin/bash
set -euo pipefail
cd /mnt/autodl_tmp1/zhuyanhao/code/NAVIDA
export PYTHONPATH="./:${PYTHONPATH:-}"
export NCCL_P2P_LEVEL=NVL
# After eval: use free GPUs. Default 2 cards for smoke.
GPUS=${GPUS:-0,1}
OUT=${OUT:-/mnt/autodl_tmp1/zhuyanhao/runs/navida_repro_20261029/train_smoke}
DATA_JSONL=${DATA_JSONL:-data/navida_train_data_r2r_mini.jsonl}
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPUS deepspeed --master_port 25421 src/train/train.py \
    --deepspeed scripts/zero2.json \
    --dataset_name "$DATA_JSONL" \
    --model_name_or_path .cache/Qwen2.5-VL-3B-Instruct \
    --num_train_epochs 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --attn_implementation sdpa \
    --lr_scheduler_type cosine \
    --gradient_checkpointing True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 2 \
    --learning_rate 2.0e-5 \
    --logging_steps 1 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 50 \
    --max_steps 20 \
    --output_dir "$OUT" \
    --report_to none
