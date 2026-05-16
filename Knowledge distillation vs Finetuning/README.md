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
Evaluate teacher and student zero-shot to establish the performance gap.

**Output:** ROUGE-1/2/L for both models → `phase1_results/`

| Model | ROUGE-1 |
|---|---|
| Teacher (7B) | ~36.4 |
| Student (0.5B) | ~32.7 |

---

### Phase 2 — Train and Compare

Each step lives in its own notebook under `notebooks/`.

| Step | Notebook | What it does | Output |
|---|---|---|---|
| 2a | `phase2a_teacher_generate.ipynb` | Teacher generates summaries on 10k train articles | `phase2_data/teacher_generations.jsonl` |
| 2b | `phase2b_kd_lora_train.ipynb` | Train student via LoRA on **teacher generations** (KD) | `phase2_outputs/student_kd_lora/` |
| 2c | `phase2c_sft_lora_train.ipynb` | Train student via LoRA on **gold summaries** (SFT) | `phase2_outputs/student_sft_lora/` |
| 2d | `phase2d_eval_compare.ipynb` | Evaluate all four models on the same test set | `phase2_eval_results/phase2_comparison.csv` |

> **Critical:** Steps 2b and 2c are identical in every way except the target labels — same articles, same LoRA config, same hyperparameters, same seed. The only variable is whether the student learns from teacher outputs or gold summaries.

---

### Phase 3 (Planned) — Cross-Domain
Re-evaluate trained students on XSum and SAMSum to test generalization.

### Phase 4 (Planned) — Scale Up
If 10k shows real signal, repeat 2a–2d with 50k examples for final paper numbers.

---

## Training Config (identical for 2b and 2c)

| Setting | Value |
|---|---|
| LoRA rank | r=16, alpha=32, dropout=0.05 |
| LoRA targets | All attention + MLP projections |
| Effective batch size | 16 (per-device 2 × grad accum 8) |
| Epochs | 3 |
| Learning rate | 3e-4, cosine schedule, 3% warmup |
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
│   ├── phase1_baseline_eval.ipynb
│   ├── phase2a_teacher_generate.ipynb
│   ├── phase2b_kd_lora_train.ipynb          # (to be created)
│   ├── phase2c_sft_lora_train.ipynb          # (to be created)
│   └── phase2d_eval_compare.ipynb            # (to be created)
├── phase1_results/
│   ├── baseline_rouge_qwen.csv
│   ├── baseline_rouge_qwen.json
│   ├── predictions_Qwen2.5-0.5B-Instruct.json
│   └── predictions_Qwen2.5-7B-Instruct.json
├── phase2_data/
│   └── teacher_generations.jsonl
├── phase2_outputs/
│   ├── student_kd_lora/                      # LoRA adapter weights (after 2b)
│   └── student_sft_lora/                     # LoRA adapter weights (after 2c)
└── phase2_eval_results/
    └── phase2_comparison.csv                 # (after 2d)
```

---

## Current Status

- [x] Phase 1 — Baseline evaluation complete
- [x] Phase 2a — Teacher generations complete (`teacher_generations.jsonl`)
- [ ] Phase 2b — KD-LoRA training
- [ ] Phase 2c — SFT-LoRA training
- [ ] Phase 2d — Evaluation and comparison
- [ ] Phase 3 — Cross-domain generalization
- [ ] Phase 4 — Scale up to 50k
