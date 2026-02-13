#!/usr/bin/env python3
"""
preprocess_chat.py — Download a chat/instruction dataset, apply a chat template,
tokenize, and chunk into fixed-length sequences compatible with the training pipeline.

The output is a HuggingFace Dataset saved to disk with a single column `input_ids`,
where each row is a list of token IDs of length `context_length` — the same format
produced by preprocess_chunk.py for pre-training data.

Usage:
    python preprocess_chat.py \
        --dataset_name HuggingFaceH4/ultrachat_200k \
        --dataset_config default \
        --split train_sft \
        --tokenizer Qwen/Qwen3-0.6B \
        --context_length 4096 \
        --output_dir /export/work/apierro/data_cache/chat_chunked_context4096

Supported dataset formats:
    1. "messages" column (list of {role, content} dicts) — e.g. ultrachat_200k
    2. "conversations" column (same schema, different name) — e.g. SlimOrca
    3. "prompt"/"response" columns — simple single-turn datasets
"""

import argparse
import os
import numpy as np
import pyarrow as pa
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess a chat dataset for SFT fine-tuning")
    p.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrachat_200k",
                   help="HF dataset name")
    p.add_argument("--dataset_config", type=str, default="default",
                   help="HF dataset config name")
    p.add_argument("--split", type=str, default="train_sft",
                   help="Dataset split")
    p.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-0.6B",
                   help="Tokenizer name or path")
    p.add_argument("--context_length", type=int, default=4096,
                   help="Fixed sequence length for each chunk")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save the chunked dataset")
    p.add_argument("--max_examples", type=int, default=None,
                   help="Limit number of conversations to process (for debugging)")
    p.add_argument("--num_proc", type=int, default=8,
                   help="Number of processes for tokenization")
    return p.parse_args()


def detect_and_format_messages(example, tokenizer):
    """
    Detect the conversation format and apply the tokenizer's chat template.
    Returns the full conversation as a single string.
    """
    # Format 1: "messages" column (ultrachat_200k, etc.)
    if "messages" in example:
        messages = example["messages"]
    # Format 2: "conversations" column (SlimOrca, etc.)
    elif "conversations" in example:
        messages = example["conversations"]
        # Normalize role names if needed (e.g. "human" -> "user", "gpt" -> "assistant")
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        messages = [
            {"role": role_map.get(m.get("from", m.get("role", "")), m.get("from", m.get("role", ""))),
             "content": m.get("value", m.get("content", ""))}
            for m in messages
        ]
    # Format 3: prompt/response columns
    elif "prompt" in example and "response" in example:
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
    else:
        raise ValueError(f"Unknown dataset format. Columns: {list(example.keys())}")

    # Apply the chat template — produces a full string with special tokens
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return text


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset_name} ({args.dataset_config}) split={args.split}")
    ds_kwargs = {}
    if args.dataset_config:
        ds_kwargs["name"] = args.dataset_config
    dataset = load_dataset(args.dataset_name, **ds_kwargs, split=args.split)

    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    print(f"Dataset has {len(dataset):,} examples")
    print(f"Columns: {dataset.column_names}")

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 1: Apply chat template and tokenize each conversation
    print("Applying chat template and tokenizing...")

    def tokenize_chat(example):
        text = detect_and_format_messages(example, tokenizer)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        # Append EOS to mark end of conversation
        tokens.append(tokenizer.eos_token_id)
        return {"input_ids": tokens}

    tokenized = dataset.map(
        tokenize_chat,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing conversations",
    )

    # Step 2: Concatenate all tokens and chunk into fixed-length sequences
    print("Concatenating and chunking...")
    chunked_array = tokenized.data.column("input_ids")
    flattened_chunks = [chunk.flatten() for chunk in chunked_array.chunks]
    flat_array = pa.concat_arrays(flattened_chunks)
    all_tokens = flat_array.to_numpy(zero_copy_only=False).astype("uint32", copy=False)

    total_tokens = len(all_tokens)
    num_chunks = total_tokens // args.context_length
    usable_tokens = num_chunks * args.context_length
    all_tokens = all_tokens[:usable_tokens]
    chunks = all_tokens.reshape(-1, args.context_length)

    print(f"Total tokens: {total_tokens:,}")
    print(f"Usable tokens: {usable_tokens:,} ({num_chunks:,} chunks of {args.context_length})")
    print(f"Dropped tokens: {total_tokens - usable_tokens:,}")

    # Step 3: Save as Arrow dataset (same format as preprocess_chunk.py)
    arrow_type = pa.list_(pa.uint32())
    flat_pa = pa.array(chunks.flatten(), type=pa.uint32())
    lists_array = pa.FixedSizeListArray.from_arrays(flat_pa, args.context_length)

    arrow_table = pa.table({"input_ids": lists_array})
    chunked_dataset = Dataset(arrow_table)
    chunked_dataset.save_to_disk(args.output_dir)

    print(f"✅ Saved {len(chunked_dataset):,} chunks to: {args.output_dir}")


if __name__ == "__main__":
    main()
