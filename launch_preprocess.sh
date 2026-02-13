#!/bin/bash
#SBATCH --job-name=preprocess_dclm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --dependency=singleton
#SBATCH --partition=pgi15-cpu
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load environment or activate venv if necessary
# source .venv/bin/activate

uv run python preprocess_download_tokenize.py \
  --dataset_name HuggingFaceFW/fineweb-edu \
  --dataset_config sample-10BT \
  --split train \
  --text_field text \
  --tokenizer fla-hub/Qwen2.5-3B-Instruct \
  --output_dir data_cache/tokenized_dataset \
  --num_proc 96

# Chunking for Stage 1 (Short Context)
uv run python preprocess_chunk.py \
  --tokenized_dataset_path data_cache/tokenized_dataset \
  --context_length 512 \
  --output_dir data_cache \
  --npy_cache_path data_cache/tokenized_5pct_512.npy

# Chunking for Stage 2 (Long Context)
uv run python preprocess_chunk.py \
  --tokenized_dataset_path data_cache/tokenized_dataset \
  --context_length 4096 \
  --output_dir data_cache \
  --npy_cache_path data_cache/tokenized_5pct_4096.npy
