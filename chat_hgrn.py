#!/usr/bin/env python3
"""
chat_hgrn.py — Interactive chat with a distilled HGRN student model.

Usage:
    python chat_hgrn.py --stage 1      # load Stage 1 checkpoint
    python chat_hgrn.py --stage 2      # load Stage 2 checkpoint (default)
    python chat_hgrn.py --model PATH   # load arbitrary checkpoint

Type 'switch' during chat to swap between Stage 1 and Stage 2 without restarting.

Sampling is auto-tuned per message based on detected task type.
Type 'mode' to see or override the current sampling preset.
"""

import argparse
import re
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
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Respond concisely and accurately."

# ---------------------------------------------------------------------------
# Sampling presets — each maps to (temperature, top_p, top_k, rep_penalty,
#                                   max_new_tokens)
# ---------------------------------------------------------------------------
SAMPLING_PRESETS = {
    "precise": {
        "description": "Factual / code / math — greedy-ish, low randomness",
        "temperature": 0.1,
        "top_p": 0.85,
        "top_k": 40,
        "repetition_penalty": 1.05,
        "max_new_tokens": 1024,
    },
    "balanced": {
        "description": "General Q&A / conversation — moderate creativity",
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "max_new_tokens": 512,
    },
    "creative": {
        "description": "Stories / brainstorm / open-ended — high diversity",
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 0,       # disabled — rely on top_p
        "repetition_penalty": 1.15,
        "max_new_tokens": 1024,
    },
}

# Keyword / pattern rules for automatic mode detection.
# Order matters — first match wins.
_TASK_RULES: list[tuple[str, re.Pattern]] = [
    # ── precise ──
    ("precise", re.compile(
        r"(?i)"
        r"\b(?:code|function|implement|debug|fix|compile|syntax|error|traceback"
        r"|python|java|rust|c\+\+|javascript|typescript|html|css|sql|bash|regex"
        r"|calculate|compute|solve|math|equation|integral|derivative|proof"
        r"|translate .* to (?:english|french|german|spanish|chinese|japanese)"
        r"|summarize|summary|extract|json|xml|csv|parse|convert|define|definition"
        r"|true or false|yes or no|correct or incorrect|what is the"
        r"|how many|how much|exact|precisely|step by step"
        r"|explain (?:the|this|how)|what does .* mean"
        r")\b"
    )),
    # ── creative ──
    ("creative", re.compile(
        r"(?i)"
        r"\b(?:write (?:a |me )?(?:story|poem|song|essay|script|dialogue|fiction|haiku|limerick)"
        r"|brainstorm|ideas? for|imagine|creative|invent|make up"
        r"|role.?play|pretend|act as|you are a"
        r"|what if|hypothetical|fantasy|dream"
        r"|funny|joke|humor|parody|satirize"
        r")\b"
    )),
    # ── balanced (fallback) ──
]


def classify_task(user_msg: str) -> str:
    """Return the best sampling preset name for a user message."""
    for preset_name, pattern in _TASK_RULES:
        if pattern.search(user_msg):
            return preset_name
    return "balanced"


def get_sampling_params(preset_name: str) -> dict:
    """Return a copy of the sampling dict for a preset."""
    return dict(SAMPLING_PRESETS[preset_name])


def fmt_preset(name: str) -> str:
    p = SAMPLING_PRESETS[name]
    return (f"  {name:10s}  temp={p['temperature']:.1f}  top_p={p['top_p']:.2f}  "
            f"top_k={p['top_k']}  rep_pen={p['repetition_penalty']:.2f}  "
            f"max_tok={p['max_new_tokens']}  — {p['description']}")


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
def generate(model, tokenizer, prompt: str, *,
             max_new_tokens: int = 512,
             temperature: float = 0.7,
             top_p: float = 0.9,
             top_k: int = 50,
             repetition_penalty: float = 1.0):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
    )
    if top_k > 0:
        gen_kwargs["top_k"] = top_k

    outputs = model.generate(**gen_kwargs)

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
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
                        help="System prompt to prepend to conversation")
    args = parser.parse_args()

    current_stage = args.stage
    model_path = args.model or STAGE_PATHS[current_stage]
    model, tokenizer = load_model(model_path, args.tokenizer)

    # mode_override: None = auto-detect per message, or a preset name to lock
    mode_override: str | None = None

    print("\n" + "=" * 60)
    print(f"  HGRN Student Model Chat  (Stage {current_stage})")
    print("  Sampling auto-tunes per message (precise / balanced / creative).")
    print()
    print("  Commands:")
    print("    'switch'       — cycle to next stage (1→2→3→1)")
    print("    'switch N'     — jump to stage N (1, 2, or 3)")
    print("    'mode'         — show current sampling mode & presets")
    print("    'mode auto'    — auto-detect mode per message (default)")
    print("    'mode <name>'  — lock to a preset (precise/balanced/creative)")
    print("    'clear'        — reset conversation history")
    print("    'quit'         — exit")
    print("=" * 60 + "\n")

    system_msg = {"role": "system", "content": args.system_prompt}
    history = [system_msg]

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
            history = [system_msg]
            print("(history cleared)\n")
            continue

        # ── mode command ──
        if user_input.lower().startswith("mode"):
            parts = user_input.split()
            if len(parts) == 1:
                # Show current setting + all presets
                current = mode_override or "auto"
                print(f"\n  Current mode: {current}")
                print("  Available presets:")
                for name in SAMPLING_PRESETS:
                    marker = " ◀" if name == mode_override else ""
                    print(f"  {fmt_preset(name)}{marker}")
                print("  Use 'mode auto' to auto-detect, or 'mode <name>' to lock.\n")
            elif parts[1].lower() == "auto":
                mode_override = None
                print("  Sampling mode set to auto-detect.\n")
            elif parts[1].lower() in SAMPLING_PRESETS:
                mode_override = parts[1].lower()
                print(f"  Sampling mode locked to '{mode_override}'.")
                print(f"  {fmt_preset(mode_override)}\n")
            else:
                print(f"  Unknown mode '{parts[1]}'. "
                      f"Choose from: auto, {', '.join(SAMPLING_PRESETS)}.\n")
            continue

        # ── switch command ──
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
            history = [system_msg]
            print(f"✅ Now using Stage {current_stage}. History cleared.\n")
            continue

        # ── Classify task & pick sampling params ──
        if mode_override:
            detected = mode_override
        else:
            detected = classify_task(user_input)
        params = get_sampling_params(detected)

        # Brief feedback so the user knows what's happening
        desc = SAMPLING_PRESETS[detected]["description"]
        print(f"  ⚙ mode={detected}  temp={params['temperature']:.1f}  "
              f"top_p={params['top_p']:.2f}  "
              f"('{desc}')")

        # Build a simple multi-turn prompt
        history.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)

        response = generate(
            model, tokenizer, prompt,
            max_new_tokens=params["max_new_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            top_k=params["top_k"],
            repetition_penalty=params["repetition_penalty"],
        )

        history.append({"role": "assistant", "content": response})
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    main()
