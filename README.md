<div align="center">

# 🧪 LLM Post-Training Techniques

**Systematic experiments in fine-tuning, alignment, and knowledge distillation for large language models**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/HarshaVardhanMannem/llm-post-training-techniques)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Models-FFD21E)](https://huggingface.co/Harsha901)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)

---

*A research-oriented repository for hands-on experimentation with advanced LLM post-training methods — comparing training strategies, measuring generalization, and publishing reproducible results.*

</div>

---

## 📌 Overview

This repository is a living lab for **post-training research on large language models**. Each experiment is self-contained with its own notebooks, data pipelines, evaluation harnesses, and published model weights. The focus areas include:

| Research Area | Status | Key Finding |
|---|---|---|
| **Knowledge Distillation vs Supervised Fine-Tuning** | ✅ Complete (10k + 50k) | KD outperforms SFT at scale, especially on cross-domain generalization |
| **Direct Preference Optimization (DPO)** | ✅ Complete | DPO successfully removes robotic "As an AI" patterns |
| **GRPO Math Reasoning (Qwen3-4B)** | ✅ Complete | GRPO with verifiable rewards elicits structured chain-of-thought on AMD ROCm 190 GB cluster |
| **Soft-Label KD (KL-divergence)** | 🔜 Planned | — |
| **Scaling Laws for Distillation** | 🔜 Planned (25k crossover) | — |

> **Design philosophy:** Every experiment is structured for reproducibility — fixed seeds, identical hyperparameters across control/treatment groups, deterministic evaluation, and all results tracked in version-controlled CSVs.

---

## 🏗️ Repository Structure

```
llm-post-training-techniques/
│
├── README.md                                    # ← You are here
├── .gitignore
│
├── Knowledge distillation vs Finetuning/        # 🔬 Experiment 1: KD vs SFT
│   ├── RESULTS.md                               # Consolidated findings with plots
│   ├── kd_vs_sft_10k_cnn_dailymail/             # 10k training samples run
│   │   ├── README.md
│   │   ├── notebooks/
│   │   │   ├── 01_baseline_eval_*.ipynb          # Teacher vs student zero-shot
│   │   │   ├── 02_*_generate_kd_targets.ipynb    # Teacher generates soft targets
│   │   │   ├── 03_kd_lora_train_*.ipynb          # Student trains on teacher outputs
│   │   │   ├── 04_sft_lora_finetune_*.ipynb      # Student trains on gold labels
│   │   │   ├── 05_kd_vs_sft_eval_*.ipynb         # In-domain + cross-domain eval
│   │   │   └── 06_eval_with_bertscore_*.ipynb    # BERTScore + generation analysis
│   │   ├── baseline_results/
│   │   ├── kd_teacher_data/
│   │   └── kd_vs_sft_eval_results/
│   │
│   ├── kd_vs_sft_50k_cnn_dailymail/             # 50k training samples run
│   │   ├── notebooks/
│   │   ├── kd_teacher_data/
│   │   └── kd_vs_sft_eval_results/
│   │
│   ├── results/                                  # Cross-run comparison & visualization
│   │   ├── comparison_output/                    # 📊 Plots and aggregated CSVs
│   │   ├── compare_10k_50k_eval.py               # Script: aggregate and compare runs
│   │   └── generate_plots.py                     # Script: generate all publication plots
│   │
│   └── huggingface_model_cards/                  # 🤗 Model card READMEs for Hub
│       ├── qwen2.5-0.5b-kd-merged-cnndm-50k/
│       ├── qwen2.5-0.5b-sft-merged-cnndm-50k/
│       └── upload_readmes.py
│
├── DPO Finetuning/                              # 🎯 Experiment 2: DPO Alignment
│   ├── README.md
│   ├── dpo_finetuning_qwen2.5_1.5B.ipynb         # Full training pipeline
│   └── dpo_comparison_base_vs_finetuned.ipynb     # Side-by-side inference
│
└── GRPO/                                        # 🧮 All GRPO experiments
    ├── README.md                                # GRPO collection overview
    └── Qwen3_4B_GRPO_Math_Reasoning.ipynb      # Exp 1: Qwen3-4B math reasoning
        ├── Phase 1: SFT warm-up (59 examples, 2 epochs)
        └── Phase 2: GRPO training (12,709 examples, vLLM + ROCm)
```

---

## 🔬 Experiments

### Experiment 1 — Knowledge Distillation vs Fine-Tuning with LoRA

**Research question:** *Does distilling from a larger teacher model outperform standard fine-tuning on gold labels when both use LoRA — and how does this change with training data scale?*

<table>
<tr><td><b>Teacher</b></td><td>Qwen2.5-7B-Instruct</td></tr>
<tr><td><b>Student</b></td><td>Qwen2.5-0.5B-Instruct</td></tr>
<tr><td><b>Task</b></td><td>Abstractive summarization (CNN/DailyMail)</td></tr>
<tr><td><b>Method</b></td><td>LoRA r=32, α=64, 4 epochs, completion-only loss</td></tr>
<tr><td><b>Evaluation</b></td><td>ROUGE-1/2/L, BERTScore — in-domain + 3 cross-domain datasets</td></tr>
</table>

#### Headline Results (ROUGE-1)

| Model | CNN/DM | XSum | SAMSum | DialogSum |
|---|:---:|:---:|:---:|:---:|
| Teacher 7B *(ceiling)* | 37.5 | 28.5 | 41.7 | 34.0 |
| Student 0.5B *(baseline)* | 25.2 | 14.8 | 24.7 | 20.5 |
| **KD-LoRA 0.5B** (10k) | 31.0 | **22.7** | 25.6 | 21.4 |
| **SFT-LoRA 0.5B** (10k) | **31.8** | 20.4 | **26.9** | **23.7** |
| **KD-LoRA 0.5B** (50k) | **32.9** | **23.0** | **30.0** | **25.5** |
| **SFT-LoRA 0.5B** (50k) | 32.2 | 20.1 | 25.4 | 20.9 |

> **Key insight:** At 10k samples, SFT wins 3 of 4 datasets. At 50k samples, **KD wins all 4** — with the largest margins on cross-domain dialogue tasks (+4.5 ROUGE-1 on SAMSum and DialogSum). KD scales faster because teacher soft targets encode richer distributional information that compounds over more training examples.

**📄 Full analysis:** [`Knowledge distillation vs Finetuning/RESULTS.md`](Knowledge%20distillation%20vs%20Finetuning/RESULTS.md)

---

### Experiment 3 — GRPO Math Reasoning (Qwen3-4B on ROCm)

**Research question:** *Can a 4B base model learn structured chain-of-thought math reasoning purely from verifiable reward signals — without human preference labels?*

<table>
<tr><td><b>Model</b></td><td>Qwen3-4B-Base (unsloth/Qwen3-4B-Base)</td></tr>
<tr><td><b>Method</b></td><td>GRPO with 4 reward functions (format + correctness)</td></tr>
<tr><td><b>LoRA</b></td><td>r=32, α=64, targeting all attention + MLP projections (1.62% params)</td></tr>
<tr><td><b>Dataset</b></td><td>DAPO-Math-17k-Processed (12,709 examples after filtering)</td></tr>
<tr><td><b>SFT Warm-up</b></td><td>OpenMathReasoning-mini: 59 examples × 2 epochs (118 steps)</td></tr>
<tr><td><b>Hardware</b></td><td>AMD ROCm 190 GB cluster — 191.69 GB VRAM, ROCm 7.0, vLLM v1</td></tr>
<tr><td><b>Inference Engine</b></td><td>vLLM with CUDA graphs, prefix caching, LoRA hot-swap</td></tr>
</table>

#### Training Outcome (Selected Steps)

| Step | Total Reward | Format Match | Correct Answer |
|---|:---:|:---:|:---:|
| 1 | −7.50 | 0.00 | −2.00 |
| 5 | **+13.00** | +3.00 | **+5.00** |
| 15 | **+13.00** | +3.00 | **+5.00** |
| 22 | +11.00 | +3.00 | +4.25 |

> **Key insight:** The model rapidly learns to produce the correct output format (reward = +3.0) AND solve the problem correctly (reward = +5.0) within just 5 GRPO steps. The two-phase SFT → GRPO approach prevents format collapse in early training.

**📄 Full details:** [`GRPO/README.md`](GRPO/README.md)

---

### Experiment 2 — Direct Preference Optimization (DPO)

**Research question:** *Can DPO eliminate robotic "As an AI language model…" deflections and produce naturally conversational responses?*

<table>
<tr><td><b>Model</b></td><td>Qwen2.5-1.5B-Instruct</td></tr>
<tr><td><b>Dataset</b></td><td>HumanLLMs/Human-Like-DPO-Dataset (5k examples)</td></tr>
<tr><td><b>Method</b></td><td>DPO with LoRA r=32, 8-bit quantization, β=0.05</td></tr>
<tr><td><b>Training</b></td><td>1 epoch, ~80 min, final loss 0.0115</td></tr>
</table>

#### Qualitative Result

| | Response to *"Do you have a go-to karaoke jam?"* |
|---|---|
| **Before (base)** | *"As an AI language model, I don't have personal preferences or experiences. However, I can provide information and suggestions…"* |
| **After (DPO)** | *"I'd be happy to help you find your next favorite karaoke song! 😊 What's on your playlist right now? 🎶✨"* |

> The fine-tuned model engages conversationally — asking follow-up questions and using natural language instead of defaulting to disclaimers.

**📄 Full details:** [`DPO Finetuning/README.md`](DPO%20Finetuning/README.md)

---

## 🤗 Published Models

All trained models are published on Hugging Face Hub for direct inference or further fine-tuning:

| Model | Method | Training Data | Link |
|---|---|---|---|
| `qwen2.5-1.5B-dpo-finetuned` | DPO + LoRA | 5k preference pairs | [🤗 Hub](https://huggingface.co/Harsha901/qwen2.5-1.5B-dpo-finetuned) |
| `qwen2.5-0.5b-kd-merged-cnndm-50k` | Knowledge Distillation + LoRA | 50k teacher outputs | [🤗 Hub](https://huggingface.co/Harsha901/qwen2.5-0.5b-kd-merged-cnndm-50k) |
| `qwen2.5-0.5b-sft-merged-cnndm-50k` | SFT + LoRA | 50k gold summaries | [🤗 Hub](https://huggingface.co/Harsha901/qwen2.5-0.5b-sft-merged-cnndm-50k) |

**Quick inference:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Harsha901/qwen2.5-0.5b-kd-merged-cnndm-50k"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "Summarize the following article:\n\n<your article here>"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
outputs = model.generate(inputs.to(model.device), max_new_tokens=160)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

---

## ⚙️ Tech Stack & Dependencies

| Category | Libraries |
|---|---|
| **Training** | `transformers`, `trl`, `peft`, `bitsandbytes` |
| **Data** | `datasets`, `huggingface_hub` |
| **Evaluation** | `rouge-score`, `bert-score`, `evaluate` |
| **Compute** | `torch`, `accelerate` |
| **Visualization** | `matplotlib`, `seaborn`, `pandas` |

### Installation

```bash
# Core training & evaluation
pip install -U transformers datasets trl peft bitsandbytes accelerate

# Evaluation metrics
pip install rouge-score bert-score evaluate

# Visualization (optional, for plot generation)
pip install matplotlib seaborn pandas
```

### Hardware Requirements

| Experiment | Minimum GPU | Training Time |
|---|---|---|
| KD/SFT (0.5B student) | 1× A100 40GB (or T4 with bf16) | ~2–4 hrs / run |
| DPO (1.5B model, 8-bit) | 1× T4 16GB | ~80 min |

All notebooks are designed to run on **Google Colab** (free or Pro tier) or any CUDA-capable environment.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/HarshaVardhanMannem/llm-post-training-techniques.git
cd llm-post-training-techniques
```

### 2. Choose an experiment

Each experiment directory is self-contained. Navigate to the relevant folder and follow the numbered notebooks in order:

```bash
# Knowledge Distillation vs Fine-Tuning (10k run)
cd "Knowledge distillation vs Finetuning/kd_vs_sft_10k_cnn_dailymail/notebooks"
# Run notebooks 01 → 06 in order

# DPO Alignment
cd "DPO Finetuning"
# Run dpo_finetuning_qwen2.5_1.5B.ipynb, then dpo_comparison_base_vs_finetuned.ipynb
```

### 3. Reproduce comparison plots

```bash
cd "Knowledge distillation vs Finetuning/results"
python compare_10k_50k_eval.py    # Aggregates results across 10k and 50k runs
python generate_plots.py          # Generates all publication-ready plots
```

---

## 🗺️ Roadmap

- [x] **Knowledge Distillation vs SFT** — 10k and 50k scale comparison
- [x] **DPO fine-tuning** — human-like response alignment
- [x] **Cross-domain evaluation** — XSum, SAMSum, DialogSum generalization
- [x] **BERTScore evaluation** — semantic similarity metrics beyond ROUGE
- [x] **Model publishing** — merged weights on Hugging Face Hub
- [x] **GRPO Math Reasoning** — Qwen3-4B trained with RL on AMD ROCm 190 GB cluster
- [ ] **25k crossover experiment** — pinpoint the exact scale at which KD overtakes SFT
- [ ] **Soft-label KD** — KL-divergence loss on token-level teacher distributions
- [ ] **Extended training** — 6–8 epochs to test if KD benefits more from longer schedules
- [ ] **Larger student** — Qwen2.5-1.5B as student to test scaling behavior
- [ ] **GRPO full run** — Complete 12,709 steps and publish merged model to Hugging Face
- [ ] **GRPO evaluation** — MATH benchmark (AIME, AMC) evaluation of fine-tuned model
- [ ] **Multi-GPU GRPO** — Scale to multi-GPU sampling for faster training throughput

---

## 🧠 Key Takeaways from Research

1. **KD needs sufficient data to pay off.** At 10k samples, SFT is the safer default. At 50k, KD is clearly superior — especially for cross-domain generalization (+3–5 ROUGE-1 points).

2. **Cross-domain generalization is where KD earns its advantage.** In-domain ROUGE is nearly identical between KD and SFT at both scales; the real separation happens on unseen domains.

3. **The scale threshold lies between 10k and 50k.** A follow-up 25k experiment would identify the exact crossover point.

4. **DPO is effective for style alignment.** A single epoch of DPO training with only 5k preference pairs is sufficient to eliminate robotic disclaimers and produce naturally conversational outputs.

5. **LoRA makes all of this accessible.** Training only 2–3% of model parameters achieves substantial performance gains, enabling experimentation on consumer GPUs.

6. **GRPO learns to reason from scratch using only reward signals.** With a 2-phase SFT→GRPO pipeline, a 4B base model can learn complex output formatting and math problem-solving simultaneously — without any human preference labels. The model achieves max reward (+13.0) by step 5 on some problems.

7. **Two-phase training prevents early reward collapse.** Starting with a 59-example SFT warm-up before GRPO is critical: it gives the model just enough format priming so that early GRPO completions aren't all clipped at max sequence length.

8. **Large VRAM accelerates GRPO throughput.** The 191 GB ROCm cluster allows vLLM to cache ~1.17M tokens and run 574× concurrency — enabling the high-throughput multi-completion sampling that GRPO requires.

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@misc{mannem2026posttraining,
  author       = {Harsha Vardhan Mannem},
  title        = {LLM Post-Training Techniques: Experiments in Knowledge Distillation, Fine-Tuning, and Alignment},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/HarshaVardhanMannem/llm-post-training-techniques}
}
```

---

## 📝 License

This project is open-sourced under the [MIT License](LICENSE).

---

<div align="center">

**Built with 🔬 curiosity and ☕ caffeine**

*Contributions, issues, and discussions are welcome.*

</div>
