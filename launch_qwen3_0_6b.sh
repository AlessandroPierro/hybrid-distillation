#!/bin/bash

# launch_qwen3_0_6b.sh
# End-to-end pipeline for Qwen3-0.6B hybrid distillation:
#   1. Convert HF weights to FLA format
#   2. Stage 1: Attention Output Alignment
#   3. Convert Stage 1 checkpoint
#   4. Stage 2: Knowledge Distillation with hybrid layers
#
# Usage: ./launch_qwen3_0_6b.sh

set -e  # Exit on any error

# --- Slurm Configuration (Optional) ---
#SBATCH --job-name=qwen3_0_6b_hybrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=g95
#SBATCH --output=logs/qwen3_0_6b_%j.out
#SBATCH --error=logs/qwen3_0_6b_%j.err

# --- Setup ---
mkdir -p logs

source .venv/bin/activate

export TRITON_CACHE_DIR="/tmp/triton_cache"
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR" > .deepspeed_env

# --- Paths ---
STAGE1_CONFIG="configs/qwen3_0_6b_gdn_v4_hybrid_1_0_uniform/stage1.yaml"
STAGE2_CONFIG="configs/qwen3_0_6b_gdn_v4_hybrid_1_0_uniform/stage2.yaml"

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
echo "Step 4: Stage 2 Training (KD with hybrid attention layers)"
echo "Config: $STAGE2_CONFIG"
echo "================================================================"

if [ ! -f "$STAGE2_CONFIG" ]; then
    echo "Error: Stage 2 config not found at $STAGE2_CONFIG"
    exit 1
fi

deepspeed train.py --cfg "$STAGE2_CONFIG"
echo "✅ Stage 2 training complete."

echo "================================================================"
echo "All done! Qwen3-0.6B hybrid distillation pipeline finished."
echo "================================================================"
