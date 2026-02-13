#!/usr/bin/env python3
"""
chat_hgrn.py — Interactive chat with a distilled HGRN student model.

Usage:
    python chat_hgrn.py --stage 1      # load Stage 1 checkpoint
    python chat_hgrn.py --stage 2      # load Stage 2 checkpoint (default)
    python chat_hgrn.py --model PATH   # load arbitrary checkpoint

Type 'switch' during chat to swap between Stage 1 and Stage 2 without restarting.
"""

import argparse
import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from fla.layers.hgrn import HGRNAttention
from distill_model import StudentConfig, StudentForCausalLM

# Register custom model so AutoModel can load the student checkpoint
AutoConfig.register('student', StudentConfig, exist_ok=True)
AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)

STAGE_PATHS = {
    1: "/export/work/apierro/checkpoints/qwen3_0_6b_hgrn_v1_hybrid_1_0_uniform/stage1/converted-hf",
    2: "/export/work/apierro/checkpoints/qwen3_0_6b_hgrn_v1_hybrid_1_0_uniform/stage2/converted-hf",
    3: "/export/work/apierro/checkpoints/qwen3_0_6b_hgrn_v1_hybrid_1_0_uniform/stage3/converted-hf",
}
DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"


def load_model(model_path: str, tokenizer_name: str):
    print(f"Loading tokenizer from {tokenizer_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    # Verify every layer uses HGRNAttention (not softmax Attention)
    for i, layer in enumerate(model.model.layers):
        print(f"Layer {i} attention is HGRNAttention: {isinstance(layer.attn, HGRNAttention)}")

    print(f"Model loaded on {next(model.parameters()).device}")
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512,
             temperature: float = 0.7, top_p: float = 0.9):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only the generated tokens (skip the prompt)
    new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Chat with a distilled HGRN student model")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=3,
                        help="Which training stage checkpoint to load (1, 2, or 3)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override: path to a specific converted-hf checkpoint")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER,
                        help="Tokenizer name or path")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    current_stage = args.stage
    model_path = args.model or STAGE_PATHS[current_stage]
    model, tokenizer = load_model(model_path, args.tokenizer)

    print("\n" + "=" * 60)
    print(f"  HGRN Student Model Chat  (Stage {current_stage})")
    print("  Commands:")
    print("    'switch'    — cycle to next stage (1→2→3→1)")
    print("    'switch N'  — jump to stage N (1, 2, or 3)")
    print("    'clear'     — reset conversation history")
    print("    'quit'      — exit")
    print("=" * 60 + "\n")

    history = []

    while True:
        try:
            user_input = input(f"[Stage {current_stage}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            history.clear()
            print("(history cleared)\n")
            continue
        if user_input.lower().startswith("switch"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1] in ("1", "2", "3"):
                new_stage = int(parts[1])
            else:
                new_stage = (current_stage % 3) + 1  # cycle 1→2→3→1
            print(f"\n⏳ Switching from Stage {current_stage} → Stage {new_stage} ...")
            del model
            torch.cuda.empty_cache()
            model, tokenizer = load_model(STAGE_PATHS[new_stage], args.tokenizer)
            current_stage = new_stage
            history.clear()
            print(f"✅ Now using Stage {current_stage}. History cleared.\n")
            continue

        # Build a simple multi-turn prompt
        history.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)

        response = generate(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        history.append({"role": "assistant", "content": response})
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    main()
