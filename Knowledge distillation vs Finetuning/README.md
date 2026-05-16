# Knowledge Distillation vs Fine-Tuning with LoRA

## What we're testing

When you have a small model and want it to improve at a task, two approaches compete:

- **Fine-tune on gold labels** — standard supervised fine-tuning (SFT)
- **Distill from a larger teacher** — train on the teacher's outputs (KD)

Both use **LoRA** (parameter-efficient training). The central question: **does distillation add value when you're already using LoRA, or is standard fine-tuning just as good?**

---

## Models

| Role | Model |
|---|---|
| Teacher | `Qwen/Qwen2.5-7B-Instruct` |
| Student | `Qwen/Qwen2.5-0.5B-Instruct` |

---

## Dataset

**CNN/DailyMail** (`cnn_dailymail`, config `3.0.0`) — news articles paired with bullet-point summaries.

- 1,000 examples for evaluation (test split, seed 42)
- 10,000 examples for training (train split, seed 42)

---

## Experiment Phases

### Phase 1 — Baseline Evaluation
Evaluate teacher (Qwen2.5-7B-Instruct) and student (Qwen2.5-0.5B-Instruct) zero-shot to establish the performance gap.

**Output:** ROUGE-1/2/L for both models → `baseline_results/`

| Model | ROUGE-1 |
|---|---|
| Teacher (7B) | ~36.4 |
| Student (0.5B) | ~32.7 |

---

### Phase 2 — Train and Compare

Each step lives in its own notebook under `notebooks/`.

| Step | Notebook | What it does | Output |
|---|---|---|---|
| 2a | `02_Qwen2.5-7B-teacher_generate_kd_targets.ipynb` | Teacher generates summaries on 10k train articles | `kd_teacher_data/teacher_generations.jsonl` |
| 2b | `03_kd_lora_train_Qwen2.5-0.5B-student.ipynb` | Train student via LoRA on **teacher generations** (KD) | `model_outputs/student_kd_lora/` |
| 2c | `04_sft_lora_finetune_Qwen2.5-0.5B-student.ipynb` | Train student via LoRA on **gold summaries** (SFT) | `model_outputs/student_sft_lora/` |
| 2d | `05_kd_vs_sft_eval_Qwen2.5-0.5B_indomain_crossdomain.ipynb` | Evaluate all four models in-domain + cross-domain | `kd_vs_sft_eval_results/` |

> **Critical:** Steps 2b and 2c are identical in every way except the target labels — same articles, same LoRA config, same hyperparameters, same seed. The only variable is whether the student learns from teacher outputs (KD) or gold summaries (SFT).

---

### Phase 3 — Cross-Domain (included in Phase 2d notebook)
Re-evaluate trained students on XSum and SAMSum to test generalization. Already integrated into `05_kd_vs_sft_eval_Qwen2.5-0.5B_indomain_crossdomain.ipynb`.

### Phase 4 (Planned) — Scale Up
If 10k shows real signal, repeat 2a–2d with 50k examples for final paper numbers.

---

## Training Config (identical for 2b and 2c)

| Setting | Value |
|---|---|
| LoRA rank | r=32, alpha=64, dropout=0.1 |
| LoRA targets | All attention + MLP projections |
| Effective batch size | 16 (per-device 4 × grad accum 4) |
| Epochs | 4 |
| Learning rate | 2e-4, cosine schedule, 3% warmup |
| Max sequence length | 1536 |
| Article pre-truncation | 6000 chars, left-truncated |
| Precision | bf16 + gradient checkpointing |
| Optimizer | AdamW fused |
| Loss | Completion-only (summary tokens only) |

---

## Evaluation Config (identical for Phase 1 and 2d)

| Setting | Value |
|---|---|
| Test examples | 1000, seed 42 |
| Decoding | Greedy (`do_sample=False`) |
| Max new tokens | 160 |
| Max input tokens | 3000 |
| Metrics | ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum |
| Post-processing | Strip preambles ("Here is a summary:", etc.) |

---

## Interpreting Results

After Phase 2d, the headline number is **KD-LoRA ROUGE-1 vs SFT-LoRA ROUGE-1**:

| Outcome | Interpretation |
|---|---|
| KD wins by 1+ points | Distillation adds value. Scale up and write the paper. |
| Within 0.5 points | No meaningful difference. Publishable as a negative result. |
| SFT wins | Teacher outputs are noisier than gold. Investigate why. |
| Neither beats baseline by much | Training didn't work. Debug first. |

---

## Directory Layout

```
.
├── README.md
├── notebooks/
│   ├── 01_baseline_eval_Qwen2.5-7B-teacher_vs_Qwen2.5-0.5B-student.ipynb
│   ├── 02_Qwen2.5-7B-teacher_generate_kd_targets.ipynb
│   ├── 03_kd_lora_train_Qwen2.5-0.5B-student.ipynb
│   ├── 04_sft_lora_finetune_Qwen2.5-0.5B-student.ipynb
│   └── 05_kd_vs_sft_eval_Qwen2.5-0.5B_indomain_crossdomain.ipynb
├── baseline_results/
│   ├── baseline_rouge_qwen.csv
│   ├── baseline_rouge_qwen.json
│   ├── predictions_Qwen2.5-0.5B-Instruct.json
│   └── predictions_Qwen2.5-7B-Instruct.json
├── kd_teacher_data/
│   └── teacher_generations.jsonl
├── model_outputs/
│   ├── student_kd_lora/       # LoRA adapter weights (after notebook 03)
│   └── student_sft_lora/      # LoRA adapter weights (after notebook 04)
└── kd_vs_sft_eval_results/
    └── ...                    # CSVs and JSONs from notebook 05
```

---

## Current Status

- [x] Phase 1 — Baseline evaluation complete
- [x] Phase 2a — Teacher generations complete (`kd_teacher_data/teacher_generations.jsonl`)
- [ ] Phase 2b — KD-LoRA training (Qwen2.5-0.5B student on teacher outputs)
- [ ] Phase 2c — SFT-LoRA training (Qwen2.5-0.5B student on gold summaries)
- [ ] Phase 2d/3 — Evaluation and comparison (in-domain + cross-domain)
- [ ] Phase 4 — Scale up to 50k
