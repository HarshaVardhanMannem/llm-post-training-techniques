<div align="center">

# 🧮 GRPO Fine-Tuning — Qwen3-4B Math Reasoning

**Teaching a 4B model to reason step-by-step through reinforcement learning from verifiable rewards**

[![Model](https://img.shields.io/badge/🤗_Base_Model-Qwen3--4B--Base-FFD21E)](https://huggingface.co/unsloth/Qwen3-4B-Base)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-DAPO--Math--17k-blue)](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed)
[![Hardware](https://img.shields.io/badge/Hardware-ROCm_190GB_Cluster-red?logo=amd)](https://rocm.docs.amd.com/)
[![Framework](https://img.shields.io/badge/Framework-Unsloth_%2B_TRL-green)](https://github.com/unslothai/unsloth)
[![ROCm](https://img.shields.io/badge/ROCm-7.0.51831-orange)](https://rocm.docs.amd.com/)
[![Torch](https://img.shields.io/badge/PyTorch-2.9.0a0-EE4C2C?logo=pytorch)](https://pytorch.org)

---

*Using Group Relative Policy Optimization (GRPO) to elicit structured chain-of-thought math reasoning from a Qwen3-4B base model — without any human preference labels.*

</div>

---

## 📌 Overview

This experiment applies **GRPO** (Group Relative Policy Optimization) — a reinforcement learning algorithm from DeepSeek-R1 — to teach Qwen3-4B to solve math problems using a custom structured reasoning format. The model learns to:

1. **Think through problems** in a `<start_working_out>...</end_working_out>` scratchpad
2. **Produce a clean answer** inside `<SOLUTION>...</SOLUTION>` tags
3. **Self-correct** its reasoning format via rule-based reward signals — without any human preference labels

| Property | Value |
|---|---|
| **Base Model** | `unsloth/Qwen3-4B-Base` |
| **Training Method** | GRPO (Group Relative Policy Optimization) |
| **LoRA Rank** | r=32, α=64 |
| **Training Dataset** | `open-r1/DAPO-Math-17k-Processed` (14,116 → ~12,709 filtered) |
| **Warm-up Dataset** | `unsloth/OpenMathReasoning-mini` (59 high-quality SFT examples) |
| **Total Trainable Params** | 66,060,288 of 4,088,528,384 (1.62%) |
| **Hardware** | AMD ROCm 190 GB VRAM cluster |
| **vLLM Version** | 0.11.1rc3 (ROCm 7.0) |
| **Inference Engine** | vLLM with LoRA + CUDA graph capture |
| **Max Sequence Length** | 2,048 tokens |

---

## 🔬 Method: GRPO

GRPO is a policy optimization algorithm that avoids the need for a separate critic network. Instead, it:

1. **Samples a group** of completions (`num_generations=4`) for each prompt using vLLM
2. **Scores each completion** with a set of reward functions (format + correctness)
3. **Normalizes rewards** within the group (group-relative advantage)
4. **Updates the policy** using PPO-style clipping, with a KL penalty to prevent distribution collapse

```
For each prompt x:
  ├── Sample G=4 completions: {y₁, y₂, y₃, y₄}
  ├── Score each: r(yᵢ) = format_reward + answer_reward
  ├── Compute group advantage: Aᵢ = (rᵢ - mean(r)) / std(r)
  └── Gradient step: maximize Σ min(ratio × Aᵢ, clip(ratio, 1±ε) × Aᵢ) - β·KL
```

This allows the model to improve purely from **verifiable scalar rewards** — no human labels, no reward model.

---

## 🏗️ Training Pipeline

### Phase 1 — SFT Warm-up (Format Priming)

Before GRPO, the model gets 2 epochs of supervised fine-tuning on **59 curated examples** from NVIDIA's OpenMathReasoning dataset (DeepSeek-R1 traces filtered for high quality). This primes the model to understand the custom output format.

```
Training: 59 examples × 2 epochs = 118 steps
Batch size: 1 | LR: 2e-4 | Optimizer: AdamW 8-bit
Final training loss: 0.352
```

### Phase 2 — GRPO Reinforcement Learning

The main training phase runs on the full **DAPO-Math-17k-Processed** dataset:

```
Filtered dataset: 12,709 examples (top 90% by prompt length)
Batch size (effective): 4 completions per prompt
Learning rate: 5e-6 | Warmup ratio: 10%
Optimizer: AdamW 8-bit | LR schedule: Linear
Gradient accumulation: 1 | Save every: 100 steps
```

---

## 📐 Custom Reasoning Format

A bespoke chat template was designed to be distinct from Qwen3's native `<think>` tokens, enabling clean separation between reasoning traces and final answers:

```
SYSTEM: You are given a problem.
        Think about the problem and provide your working out.
        Place it between <start_working_out> and <end_working_out>.
        Then, provide your solution between <SOLUTION></SOLUTION>

USER:   <math problem>

ASST:   <start_working_out>
        [extended reasoning trace]
        <end_working_out>
        <SOLUTION>42</SOLUTION>
```

> **Why a custom format?** Using custom tags instead of native `<think>` tokens lets the reward functions precisely detect malformed outputs and train the model to produce well-structured responses from scratch.

---

## 🎯 Reward Functions

Four complementary reward signals are combined to guide training:

| Reward Function | Signal | Score Range |
|---|---|---|
| `match_format_exactly` | Full regex match for all 3 tags in correct order | +3.0 if matched, else 0 |
| `match_format_approximately` | Per-tag presence check (rewards partial compliance) | –1.0 to +1.5 |
| `check_answer` | Exact / near-exact string match with ground truth | –4.5 to +5.0 |
| `check_numbers` | Numeric equivalence (handles `123,456` formatting) | –2.5 to +3.5 |

**Design rationale:**
- Format rewards prevent the model from ignoring the output structure entirely
- Two answer-checking functions handle both exact string matching and numerical tolerance
- Approximate format scoring provides gradient signal even for partially correct outputs early in training
- Penalizing wrong answers harder than rewarding correct ones discourages hallucination

```python
# Reward scoring snippet
if guess == true_answer:         score += 5.0   # Exact match
elif guess.strip() == true_ans:  score += 3.5   # Strip whitespace
elif 0.9 <= ratio <= 1.1:        score += 2.0   # Within 10%
elif 0.8 <= ratio <= 1.2:        score += 1.5   # Within 20%
else:                            score -= 2.5   # Wrong — penalize
```

---

## 💻 Hardware & Infrastructure

This experiment was trained on an **AMD ROCm cluster** with ~190 GB of VRAM:

| Component | Spec |
|---|---|
| **GPU VRAM** | 191.688 GB (CUDA compute capability 9.4) |
| **ROCm Toolkit** | 7.0.51831-a3e329ad8 |
| **PyTorch** | 2.9.0a0+git1c57644 |
| **Triton** | 3.4.0 |
| **Attention** | Flash Attention 2 (FA2=True) |
| **Inference** | vLLM v1 engine with Triton backend |
| **Precision** | BFloat16 throughout |
| **KV Cache** | 165.26 GB available (~1.17M token cache) |
| **Max Concurrency** | 574× for 2,048-token sequences |

**vLLM configuration highlights:**
- Prefix caching enabled (avoids re-computing system prompt KV)
- Chunked prefill with 2,048 max batched tokens
- CUDA graph capture for decode (106 sizes) and piecewise prefill (134 sizes)
- LoRA hot-swapping with `max_lora_rank=32`

> The large VRAM budget allows vLLM to hold the full KV cache for hundreds of concurrent sequences, enabling the high-throughput sampling needed for GRPO's multi-completion generation.

---

## 📊 Training Metrics (Selected Steps)

Early GRPO training behavior observed from the training log:

| Step | Reward | Format (Exact) | Format (Approx) | Answer | Numbers |
|---|---|---|---|---|---|
| 1 | −7.50 | 0.00 | −3.00 | −2.00 | −2.50 |
| 2 | +1.63 | +2.25 | +0.38 | −0.50 | −0.50 |
| 5 | **+13.00** | +3.00 | +1.50 | **+5.00** | **+3.50** |
| 11 | +2.75 | +1.50 | −0.75 | +1.50 | +0.50 |
| 15 | **+13.00** | +3.00 | +1.50 | **+5.00** | **+3.50** |
| 22 | +11.00 | +3.00 | +1.50 | +4.25 | +2.25 |

**Observations:**
- Steps with reward = −7.5 indicate all 4 sampled completions hit max sequence length (clipped_ratio = 1.0) — early training instability is expected
- By step 5, the model correctly learns the format AND produces the right answer
- KL divergence stabilizes around 0.12–0.19, indicating the policy stays close to the reference
- Reward variance (`reward_std`) is high early and gradually decreases as the policy improves

---

## 🗂️ Repository Structure

```
GRPO - Qwen3 4B Math Reasoning/
│
├── README.md                           # ← You are here
└── Qwen3_4B_GRPO_Math_Reasoning.ipynb  # Complete training pipeline
    │
    ├── Cell 1:  Model Loading          # Unsloth + vLLM + LoRA setup
    ├── Cell 2:  Reasoning Format       # Custom token/tag definition
    ├── Cell 3:  Chat Template          # Jinja2 template construction
    ├── Cell 4:  Template Verification  # Sanity check on tokenization
    │
    ├── ─── Phase 1: SFT Warm-up ─────────────────────────────────────
    │
    ├── Cell 5:  Dataset Load           # OpenMathReasoning-mini (CoT)
    ├── Cell 6:  Format Dataset         # Convert to message format
    ├── Cell 7:  Template Verification  # Inspect formatted examples
    ├── Cell 8:  Length Filter          # Keep examples ≤ 1024 tokens
    ├── Cell 9:  HF Dataset Convert     # Pandas → HuggingFace Dataset
    ├── Cell 10: SFT Trainer Setup      # TRL SFTTrainer config
    ├── Cell 11: SFT Training           # 118 steps, loss → 0.352
    ├── Cell 12: Inference Test         # Verify format learned
    ├── Cell 13: Memory Cleanup         # Free VRAM for GRPO phase
    │
    ├── ─── Phase 2: GRPO Training ───────────────────────────────────
    │
    ├── Cell 14: GRPO Dataset Load      # DAPO-Math-17k-Processed
    ├── Cell 15: Dataset Inspection     # Sample prompt inspection
    ├── Cell 16: Answer Extraction      # extract_hash_answer function
    ├── Cell 17: Dataset Mapping        # Apply system prompt + answer
    ├── Cell 18: Dataset Verification   # Confirm format
    ├── Cell 19: Regex Builder          # match_format regex compile
    ├── Cell 20: Regex Test 1           # Partial match test
    ├── Cell 21: Regex Test 2           # Whitespace tolerance test
    ├── Cell 22: Reward: Format Exact   # match_format_exactly()
    ├── Cell 23: Reward: Format Approx  # match_format_approximately()
    ├── Cell 24: Reward: Answer         # check_answer()
    ├── Cell 25: Number Match Test      # Verify regex on edge cases
    ├── Cell 26: Reward: Numbers        # check_numbers() + logging
    ├── Cell 27: Prompt Length Filter   # 90th percentile cut
    └── Cell 28: GRPO Training          # GRPOConfig + GRPOTrainer
```

---

## 🚀 Getting Started

### Prerequisites

```bash
# Install Unsloth with ROCm support
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Or for stable release
pip install unsloth

# Additional dependencies
pip install trl vllm datasets pandas numpy
```

### Running the Notebook

1. Ensure you have a ROCm-compatible GPU (or adapt to CUDA by removing ROCm-specific flags)
2. Open `Qwen3_4B_GRPO_Math_Reasoning.ipynb`
3. Run cells sequentially — each phase has a clear separator

> **Memory note:** Phase 1 (SFT) and Phase 2 (GRPO) share the same model/tokenizer. `gc.collect()` and `torch.cuda.empty_cache()` are called between phases. On smaller GPUs, adjust `gpu_memory_utilization` and `num_generations`.

### Key Hyperparameters to Tune

| Hyperparameter | Current | Guidance |
|---|---|---|
| `lora_rank` | 32 | Increase to 64 for better capacity |
| `max_seq_length` | 2048 | Increase for longer reasoning traces |
| `num_generations` | 4 | Decrease if OOM; min practical = 2 |
| `learning_rate` (GRPO) | 5e-6 | Lower = more stable; try 1e-6 |
| `gradient_accumulation_steps` | 1 | Increase to 4 for smoother gradients |
| `gpu_memory_utilization` | 0.9 | Reduce to 0.7 if vLLM OOM |

---

## 🔧 Enhancements & Future Work

### Implemented Enhancements

- [x] **Two-phase training:** SFT warm-up → GRPO (avoids cold-start format collapse)
- [x] **Multi-signal rewards:** 4 complementary reward functions for robust learning
- [x] **Numeric tolerance:** Partial credit for near-correct numerical answers
- [x] **Prefix caching:** vLLM prefix cache reduces re-computation for repeated system prompts
- [x] **Chunked prefill:** Enables long-context processing without memory spikes
- [x] **Debug logging:** `PRINTED_TIMES` / `PRINT_EVERY_STEPS` prints Q&A samples periodically

### Potential Next Steps

- [ ] **Multi-GPU GRPO** — Enable `data_parallel_size > 1` for faster sampling with vLLM
- [ ] **Longer training** — Run full 12,709 steps (currently at 902 in the log)
- [ ] **Evaluation harness** — Add MATH benchmark (AIME, AMC, Minerva) evaluation
- [ ] **Reward shaping** — Add partial credit for correct intermediate reasoning steps
- [ ] **Adaptive format penalty** — Decay format reward weight as training progresses
- [ ] **Symbolic math matching** — Use `sympy` to verify equivalent algebraic expressions
- [ ] **Temperature annealing** — Decrease sampling temperature over training
- [ ] **WandB integration** — Replace `report_to="none"` with W&B for full training curves
- [ ] **Merge and publish** — Merge LoRA weights and push to Hugging Face Hub

---

## 📚 Background & References

| Resource | Description |
|---|---|
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | Original GRPO algorithm and reasoning-via-RL methodology |
| [DAPO-Math-17k](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed) | Training dataset — high-quality competition math problems |
| [OpenMathReasoning-mini](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini) | NVIDIA's filtered DeepSeek-R1 reasoning traces (SFT warm-up) |
| [Unsloth](https://github.com/unslothai/unsloth) | 2× faster training with kernel-fused LoRA and vLLM integration |
| [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) | Hugging Face TRL GRPO implementation |
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput inference engine for GRPO sampling |
| [Qwen3](https://huggingface.co/Qwen/Qwen3-4B) | Base model architecture (4B dense MoE variant) |

---

## ⚙️ Technical Notes

### Why Qwen3-4B-Base (not Instruct)?

The **base model** is used instead of the instruct variant because:
- Base models have no prior instruction-following behavior to "unlearn"
- GRPO works best when the policy starts from a neutral prior
- The SFT warm-up step provides just enough formatting signal before RL begins

### Why not 4-bit quantization?

The notebook explicitly sets `load_in_4bit=False`. This is because:
- ROCm's `bitsandbytes` 4-bit support was unstable at training time
- BF16 LoRA-16bit is preferred for gradient stability in GRPO
- 191 GB of VRAM makes full BF16 loading trivially feasible

### vLLM and LoRA Co-training

Unsloth patches vLLM to support hot-swapping LoRA weights during GRPO sampling. This means:
- vLLM generates completions with the **current** LoRA weights
- After each gradient step, updated weights are synced back to vLLM
- No separate inference model is maintained — saving substantial VRAM

---

## 📝 License

This experiment is open-sourced under the [MIT License](../LICENSE).

---

<div align="center">

**Trained with 🔥 ROCm + ❄️ Unsloth on 190 GB of AMD silicon**

*Part of the [LLM Post-Training Techniques](https://github.com/HarshaVardhanMannem/llm-post-training-techniques) repository*

</div>
