#!/bin/bash

# launch_stage1.sh
# Script to launch Stage 1 training (Attention Output Alignment)
# Usage: ./launch_stage1.sh [config_path]

# --- Slurm Configuration (Optional - based on other launch scripts) ---
#SBATCH --job-name=stage1_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=32
#SBATCH --time=24:00:00
#SBATCH --partition=pgi15-h100
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err

# --- Setup ---
# Create logs directory if it doesn't exist
mkdir -p logs

# Environment variables setup
export WANDB_ENTITY="alessandro-pierro-lmu-munich"
export WANDB_PROJECT="distill-hgrn"
export TRITON_CACHE_DIR="/tmp/triton_cache"
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR" > .deepspeed_env
echo "WANDB_ENTITY=alessandro-pierro-lmu-munich" >> .deepspeed_env
echo "WANDB_PROJECT=distill-hgrn" >> .deepspeed_env

# --- Configuration ---
# Default config path from README example
DEFAULT_CONFIG="configs/qwen3_0.6B/stage1.yaml"

CONFIG_PATH="${1:-$DEFAULT_CONFIG}"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Configuration file not found at $CONFIG_PATH"
    echo "Usage: $0 [path_to_stage1_config]"
    exit 1
fi

echo "================================================================"
echo "Launching Stage 1 Training"
echo "Config: $CONFIG_PATH"
echo "Output logs: logs/stage1_<jobid>.out (if running via sbatch)"
echo "================================================================"

# --- Launch Training ---
uv run deepspeed train.py --cfg "$CONFIG_PATH"
