#!/bin/bash
#SBATCH --job-name=qwen3_0_6b_hgrn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=g95
#SBATCH --output=logs/qwen3_0_6b_hgrn_%j.out
#SBATCH --error=logs/qwen3_0_6b_hgrn_%j.err

# --- Setup ---
mkdir -p logs

source .venv/bin/activate
wandb disabled
export TRITON_CACHE_DIR="/tmp/triton_cache"
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR" > .deepspeed_env

# --- Paths ---
STAGE1_CONFIG="configs/qwen3_0_6b_hgrn_v1_hybrid_1_0_uniform/stage1.yaml"
STAGE2_CONFIG="configs/qwen3_0_6b_hgrn_v1_hybrid_1_0_uniform/stage2.yaml"
STAGE3_CONFIG="configs/qwen3_0_6b_hgrn_v1_hybrid_1_0_uniform/stage3.yaml"
CHAT_DATA_DIR="/export/work/apierro/data_cache/chat_chunked_context4096"

# ================================================================
# Step 1: Convert Qwen3-0.6B from HuggingFace to FLA format
# ================================================================
echo "================================================================"
echo "Step 1: Converting Qwen3-0.6B to FLA format"
echo "================================================================"

if [ -d "converted/Qwen3-0.6B" ]; then
    echo "Converted model already exists at converted/Qwen3-0.6B, skipping conversion."
else
    python convert/convert_from_qwen3.py \
        --model Qwen/Qwen3-0.6B \
        --config 0.6b \
        --output converted/Qwen3-0.6B \
        --precision bfloat16
    echo "✅ Conversion complete."
fi

# ================================================================
# Step 2: Stage 1 — Attention Output Alignment
# ================================================================
echo "================================================================"
echo "Step 2: Stage 1 Training (Attention Output Alignment)"
echo "Config: $STAGE1_CONFIG"
echo "================================================================"

if [ ! -f "$STAGE1_CONFIG" ]; then
    echo "Error: Stage 1 config not found at $STAGE1_CONFIG"
    exit 1
fi

deepspeed train.py --cfg "$STAGE1_CONFIG"
echo "✅ Stage 1 training complete."

# ================================================================
# Step 3: Convert Stage 1 checkpoint for Stage 2 initialization
# ================================================================
echo "================================================================"
echo "Step 3: Converting Stage 1 checkpoint"
echo "================================================================"

python convert_ckpt.py --cfg "$STAGE1_CONFIG"
echo "✅ Stage 1 checkpoint converted."

# ================================================================
# Step 4: Stage 2 — Knowledge Distillation with hybrid layers
# ================================================================
echo "================================================================"
echo "Step 4: Stage 2 Training (KD with hybrid HGRN attention layers)"
echo "Config: $STAGE2_CONFIG"
echo "================================================================"

if [ ! -f "$STAGE2_CONFIG" ]; then
    echo "Error: Stage 2 config not found at $STAGE2_CONFIG"
    exit 1
fi

deepspeed train.py --cfg "$STAGE2_CONFIG"
echo "✅ Stage 2 training complete."

# ================================================================
# Step 5: Convert Stage 2 checkpoint
# ================================================================
echo "================================================================"
echo "Step 5: Converting Stage 2 checkpoint"
echo "================================================================"

python convert_ckpt.py --cfg "$STAGE2_CONFIG"
echo "✅ Stage 2 checkpoint converted."

# ================================================================
# Step 6: Preprocess chat dataset for Stage 3 SFT
# ================================================================
echo "================================================================"
echo "Step 6: Preprocessing chat dataset (ultrachat_200k)"
echo "================================================================"

if [ -d "$CHAT_DATA_DIR" ]; then
    echo "Chat dataset already exists at $CHAT_DATA_DIR, skipping preprocessing."
else
    python preprocess_chat.py \
        --dataset_name HuggingFaceH4/ultrachat_200k \
        --dataset_config default \
        --split train_sft \
        --tokenizer Qwen/Qwen3-0.6B \
        --context_length 4096 \
        --output_dir "$CHAT_DATA_DIR"
    echo "✅ Chat dataset preprocessing complete."
fi

# ================================================================
# Step 7: Stage 3 — Supervised Fine-Tuning on chat data
# ================================================================
echo "================================================================"
echo "Step 7: Stage 3 Training (SFT on chat data)"
echo "Config: $STAGE3_CONFIG"
echo "================================================================"

if [ ! -f "$STAGE3_CONFIG" ]; then
    echo "Error: Stage 3 config not found at $STAGE3_CONFIG"
    exit 1
fi

deepspeed train.py --cfg "$STAGE3_CONFIG"
echo "✅ Stage 3 training complete."

echo "================================================================"
echo "All done! Qwen3-0.6B HGRN hybrid distillation + SFT pipeline finished."
echo "================================================================"
