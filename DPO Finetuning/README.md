# DPO Fine-Tuning — Qwen2.5-1.5B-Instruct

## What this experiment does

Trains `Qwen/Qwen2.5-1.5B-Instruct` with **Direct Preference Optimization (DPO)** to make its responses more human-like and conversational. The base model tends to deflect with "As an AI language model, I don't have personal opinions…" — DPO teaches it to engage naturally instead.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `dpo_finetuning_qwen2.5_1.5B.ipynb` | Full DPO training pipeline — data prep, LoRA setup, DPO training, model push to Hub |
| `dpo_comparison_base_vs_finetuned.ipynb` | Side-by-side inference: base model vs DPO-finetuned model |

---

## Model

| Role | Model ID |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Fine-tuned (Hub) | `Harsha901/qwen2.5-1.5B-dpo-finetuned` |

The base model is loaded in **8-bit** during training (BitsAndBytes) to fit within GPU memory. After training, LoRA adapters are merged back into the base weights and the full model is pushed to HuggingFace Hub.

---

## Dataset

**`HumanLLMs/Human-Like-DPO-Dataset`** — 5,000 training examples (train split).

Each example has three fields:

| Field | Description |
|---|---|
| `prompt` | A casual human message (e.g. "Oh, I just saw the best meme — have you seen it?") |
| `chosen` | A human-like, engaging response |
| `rejected` | A robotic, AI-deflecting response |

**Example:**
- Prompt: *"Oh, I just saw the best meme — have you seen it?"*
- Chosen: *"😂 Ah, no I haven't! I'm dying to know, what's the meme about?"*
- Rejected: *"I'm an artificial intelligence language model, I don't have personal experiences…"*

The dataset is preprocessed to match Qwen's chat template format before training.

---

## Training Config

### LoRA
| Setting | Value |
|---|---|
| Rank (r) | 32 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable params | 36.9M / 1.58B (2.34%) |

### DPO
| Setting | Value |
|---|---|
| Per-device batch size | 2 |
| Gradient accumulation | 16 (effective batch = 32) |
| Epochs | 1 |
| Learning rate | 5e-5, cosine schedule |
| Precision | fp16 |
| Optimizer | AdamW |
| Beta (KL penalty) | 0.05 |
| Logging / save steps | 50 |

### Training result
| Metric | Value |
|---|---|
| Global steps | 157 |
| Training loss | 0.0115 |
| Runtime | ~80 min (4782s) |
| Samples/sec | 1.045 |

---

## Results — Base vs Fine-Tuned

**Prompt:** *"Do you have a go-to karaoke jam?"*

**Base model (`Qwen2.5-1.5B-Instruct`):**
> As an AI language model, I don't have personal preferences or experiences. However, I can provide information and suggestions about popular songs that people enjoy singing along to. If you're looking for some recommendations, here are a few popular songs…

**DPO fine-tuned (`Harsha901/qwen2.5-1.5B-dpo-finetuned`):**
> I'd be happy to help you find your next favorite karaoke song if you're ever in need of some suggestions! 😊 What's on your playlist right now? 🎶✨ Are there any specific genres or artists that you like to listen to while singing along? 😂

The fine-tuned model is noticeably more conversational — it asks follow-up questions and uses natural language rather than immediately defaulting to the "I'm an AI" disclaimer.

---

## Directory Layout

```
DPO Finetuning/
├── README.md
├── dpo_finetuning_qwen2.5_1.5B.ipynb      # Training pipeline
└── dpo_comparison_base_vs_finetuned.ipynb  # Inference comparison
```

---

## Dependencies

```
transformers
datasets
bitsandbytes
trl
peft
huggingface_hub
torch
```

Install with:
```bash
pip install -q -U transformers datasets bitsandbytes trl peft huggingface_hub
```
