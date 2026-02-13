# Copilot Instructions — Hybrid Distillation

## Project Overview

This codebase distills **softmax-attention** transformer teachers (Qwen2.5, LLaMA3) into **hybrid** students that mix linear-attention layers (e.g., GatedDeltaNet, GLA, PaTH) with a subset of full-attention layers. The pipeline follows the RADLADS approach: two-stage training plus an optional layer-selection phase.

## Architecture & Data Flow

1. **Stage 1 (Attention Output Alignment)** — `train.py → build_student_for_stage1()` loads the teacher, wraps each attention layer with `AttentionDistillationWrapper` (in `wrapper.py`), and trains only the `student_attn` sub-modules via per-layer MSE loss collected by `DistillTrainer` in `hf_trainer.py`. The student is **all-linear** (`keep_full_attention_layers: []`). Output: `<output_dir>/converted-hf/`.
2. **Stage 2 (Logit Distillation)** — `build_student_for_stage2_and_3()` constructs a hybrid `StudentForCausalLM` from the Stage-1 checkpoint, restores teacher weights for the chosen full-attention layers, then uses `KDTrainer` with FLA's fused KL-divergence (`forward_kl`) against the teacher. DeepSpeed ZeRO-3 shards the frozen teacher.
3. **Stage 3 (optional, Long-Context Finetuning)** — Uses `FinetuneTrainer` with standard causal-LM cross-entropy, no teacher.

Key files:
- `train.py` — entry point; config parsing, model building, trainer dispatch
- `hf_trainer.py` — `DistillTrainer` (Stage 1), `KDTrainer` (Stage 2), `FinetuneTrainer` (Stage 3)
- `wrapper.py` — `AttentionDistillationWrapper` pairs teacher/student attention per-layer
- `distill_model/modeling_distilled_student.py` — `StudentForCausalLM`, `StudentBlock`, `forward_kl`
- `distill_model/student_layers.py` — student attention variants (GatedDeltaNet v1–v6, GLA, PaTH, SWA, etc.)
- `distill_model/config_distilled_student.py` — `StudentConfig` with `keep_full_attention_layers`, `student_name`
- `convert_ckpt.py` — converts DeepSpeed checkpoints to HF-format `converted-hf/`

## Config System

YAML configs live under `configs/{teacher}_{size}_{student_variant}_hybrid_{ratio}_{selector}/`. Each folder has `stage1.yaml` and/or `stage2.yaml`. Key fields:

```yaml
stage: 1  # or 2, 3
student_model:
  name: 'gdn_v4'                     # maps to STUDENT_ATTENTION_MAP in modeling_distilled_student.py
  keep_full_attention_layers: []      # Stage 1 always []; Stage 2 lists layer indices
train:
  student_init_ckpt: '...'           # Stage 2 points to Stage 1's converted-hf/
  target_tokens: 100_000_000         # drives max_steps computation
  lr_attn: 0.001                     # separate LR for attention params
  lr: 0.001                          # LR for everything else
```

One Stage-1 checkpoint is shared across many Stage-2 configs (different ratios/selectors).

## Commands

Training runs on a **Slurm** cluster. Launch scripts (`launch_stage1.sh`, `launch_preprocess.sh`) contain `#SBATCH` headers and can be submitted via `sbatch` or run locally with `bash`.

```bash
# Training (always via deepspeed, typically inside an sbatch script)
deepspeed train.py --cfg configs/<config_folder>/stage1.yaml
deepspeed train.py --cfg configs/<config_folder>/stage2.yaml

# Preprocessing (CPU-only Slurm job — see launch_preprocess.sh)
python preprocess_download_tokenize.py --dataset_name <hf_dataset> --tokenizer <tokenizer> --output_dir <dir>
python preprocess_chunk.py --tokenized_dataset_path <dir> --context_length 512 --output_dir <dir>

# Evaluation (uses lm-evaluation-harness, must be installed separately)
bash eval.sh /path/to/converted-hf/

# Checkpoint conversion (auto-called after training, but can run manually)
python convert_ckpt.py --cfg <config.yaml>
```

**Data paths are machine-specific.** YAML configs and launch scripts contain absolute paths (e.g. `/export/work/apierro/...`). Update `data.cache_dir`, `train.output_dir`, `train.student_init_ckpt`, and `HF_HOME` to match your local storage before running.

## Adding a New Student Attention Type

1. Implement the attention class in `distill_model/student_layers.py`, subclassing from the relevant FLA layer (e.g., `GatedDeltaNet`). Must implement `init_from_teacher(self, teacher_attn)` to copy teacher Q/K/V/O weights.
2. Register it in `STUDENT_ATTENTION_MAP` inside `distill_model/modeling_distilled_student.py → get_student_attention_class()`.
3. Add a corresponding convert script if needed (see section below).
4. Reference the new name in YAML configs via `student_model.name`.

## Teacher Convert Scripts (`convert/`)

Each supported teacher family has its own weight-mapping script that converts HF weights into the FLA `TransformerConfig` format. These are **not interchangeable** — each encodes architecture-specific config dicts and weight-copy logic.

| Script | Teacher family | Supported sizes | Notable differences |
|---|---|---|---|
| `convert/convert_from_qwen2.5.py` | Qwen 2.5 | `1.5b`, `7b` | `qkv_bias=True`, `tie_word_embeddings` varies by size |
| `convert/convert_from_qwen3.py` | Qwen 3 | `0.6b`, `8b` | `qk_norm=True` (0.6b), `qkv_bias=False` |
| `convert/convert_from_llama3.2.py` | LLaMA 3.2 | `3b` | `qkv_bias=False`, `rope_theta=500000`, multi-EOS tokens |

When adding a **new teacher architecture**, create a new `convert/convert_from_<family>.py` with the correct `config_dict` and weight-mapping logic. Key things that differ per teacher: `qkv_bias`, `qk_norm`, `tie_word_embeddings`, `rope_theta`, `norm_eps`, token IDs.

## Conventions

- **No test suite** — the project has no unit or integration tests. Validation is done by monitoring W&B loss curves and running `eval.sh` on converted checkpoints.
- **Slurm cluster** — training jobs are submitted via `sbatch`. Launch scripts write a `.deepspeed_env` file to propagate env vars (`TRITON_CACHE_DIR`, W&B keys) across Slurm nodes.
- **bf16 everywhere** — models load and train in `torch.bfloat16`; DeepSpeed configs enable `bf16`.
- **DeepSpeed ZeRO stages** — Stage 1 uses ZeRO-1 (`ds_configs/stage_1.json`), Stage 2 uses ZeRO-2 for the student + ZeRO-3 for the frozen teacher (`ds_configs/stage_2_teacher.json`).
- **Dual learning rates** — `train.py → get_optimizer()` splits params into `attn` vs `other` groups with `lr_attn` and `lr` respectively.
- **`fuse_swiglu = False`** is required at Stage 2 for DeepSpeed ZeRO-3 compatibility.
- **Custom HF model registration** — `StudentConfig` and `StudentForCausalLM` are registered via `AutoConfig.register` / `AutoModelForCausalLM.register` at the top of `train.py` and `convert_ckpt.py`. Any script that loads student checkpoints must do the same.
- **FLA dependency** — the project heavily depends on `flash-linear-attention` (fla) for linear attention kernels, fused losses (`FusedKLDivLoss`, `FusedCrossEntropyLoss`), and `RMSNorm`.
- **Environment variables** — `HF_TOKEN`, `HF_HOME`, `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`, `TRITON_CACHE_DIR` must be set. `TRITON_CACHE_DIR` is overridden per-rank in `train.py`.

## Layer Selection (GA-S2)

Scripts in `layer_selection/` implement the KL-guided layer selection:
1. `generate_layer_configs_ga_s2.py` — generates N Stage-2 YAML configs (one per layer, each with `keep_full_attention_layers: [idx]`)
2. Run each via `deepspeed train.py --cfg ...` (independent, parallelizable)
3. `retrieve_loss_log_from_wandb.py` — pulls loss curves from W&B (requires `ENTITY`/`PROJECT` constants)
4. `get_ranking_from_wandb_loss_log.py` — ranks layers by distillation loss improvement
