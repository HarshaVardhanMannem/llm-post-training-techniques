#!/usr/bin/env python3
"""
Compare 10k vs 50k eval results from notebook 06 (BERTScore + gen-fix ROUGE).

Reads ``all_results.csv`` from each run's ``kd_vs_sft_eval_results/`` folder,
rebuilds the headline pivot tables from notebook 06, and writes side-by-side
comparisons plus KD-vs-SFT deltas.

Usage:
    python compare_10k_50k_eval.py
    python compare_10k_50k_eval.py              # tables + plots (default)
    python compare_10k_50k_eval.py --no-plots   # tables only
    python compare_10k_50k_eval.py --dir-10k path/to/10k/kd_vs_sft_eval_results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ── Defaults (relative to this file) ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_10K = ROOT / "kd_vs_sft_10k_cnn_dailymail" / "kd_vs_sft_eval_results"
DEFAULT_50K = ROOT / "kd_vs_sft_50k_cnn_dailymail" / "kd_vs_sft_eval_results"
OUT_DIR = Path(__file__).parent / "comparison_output"

MODEL_ORDER = [
    "teacher_7B",
    "student_baseline_0.5B",
    "student_SFT_LoRA",
    "student_KD_LoRA",
]
DATASET_ORDER = ["cnn_dailymail", "xsum", "samsum", "dialogsum"]
DATASET_LABELS = {
    "cnn_dailymail": "CNN/DM (in-domain)",
    "xsum": "XSum (cross)",
    "samsum": "SAMSum (cross)",
    "dialogsum": "DialogSum (cross)",
}

HEADLINE_METRICS = [
    "rouge1_clean",
    "rouge2_clean",
    "rougeL_clean",
    "bertscore_f1",
    "avg_pred_words",
    "preamble_rate",
]
PIVOT_METRICS = [
    "rouge1_clean",
    "rouge2_clean",
    "rougeL_clean",
    "bertscore_f1",
    "avg_pred_words",
]


def load_results(results_dir: Path, run_label: str) -> pd.DataFrame:
    csv_path = results_dir / "all_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    df = pd.read_csv(csv_path)
    df["run"] = run_label
    return df


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Model × dataset table (same logic as notebook 06 headline tables)."""
    p = df.pivot_table(
        index="model_label",
        columns="dataset",
        values=metric,
        aggfunc="first",
    )
    p = p.reindex([m for m in MODEL_ORDER if m in p.index])
    cols = [c for c in DATASET_ORDER if c in p.columns]
    return p[cols]


def kd_vs_sft_delta(pivot: pd.DataFrame) -> pd.Series:
    """KD − SFT per dataset; positive means KD wins."""
    if "student_KD_LoRA" not in pivot.index or "student_SFT_LoRA" not in pivot.index:
        return pd.Series(dtype=float)
    return pivot.loc["student_KD_LoRA"] - pivot.loc["student_SFT_LoRA"]


def compare_metric(df_10k: pd.DataFrame, df_50k: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Long table: model, dataset, 10k, 50k, delta (50k − 10k)."""
    p10 = pivot_metric(df_10k, metric)
    p50 = pivot_metric(df_50k, metric)
    rows = []
    for model in p10.index:
        for ds in p10.columns:
            v10 = p10.loc[model, ds]
            v50 = p50.loc[model, ds] if model in p50.index and ds in p50.columns else float("nan")
            rows.append(
                {
                    "metric": metric,
                    "model_label": model,
                    "dataset": ds,
                    "value_10k": v10,
                    "value_50k": v50,
                    "delta_50k_minus_10k": round(v50 - v10, 3) if pd.notna(v50) and pd.notna(v10) else None,
                }
            )
    return pd.DataFrame(rows)


def training_lift(pivot: pd.DataFrame, metric: str) -> pd.DataFrame:
    """LoRA gain over untrained baseline (max of KD/SFT minus baseline)."""
    if "student_baseline_0.5B" not in pivot.index:
        return pd.DataFrame()
    base = pivot.loc["student_baseline_0.5B"]
    kd = pivot.loc["student_KD_LoRA"] if "student_KD_LoRA" in pivot.index else None
    sft = pivot.loc["student_SFT_LoRA"] if "student_SFT_LoRA" in pivot.index else None
    rows = []
    for ds in pivot.columns:
        b = base[ds]
        row = {"dataset": ds, "baseline": b}
        if kd is not None:
            row["KD_lift"] = round(kd[ds] - b, 3)
        if sft is not None:
            row["SFT_lift"] = round(sft[ds] - b, 3)
        if kd is not None and sft is not None:
            row["best_method"] = "KD" if kd[ds] >= sft[ds] else "SFT"
        rows.append(row)
    return pd.DataFrame(rows)


def print_section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def print_pivot(title: str, pivot: pd.DataFrame) -> None:
    print_section(title)
    display = pivot.rename(columns=DATASET_LABELS)
    print(display.to_string(float_format=lambda x: f"{x:.3f}"))


def print_kd_sft_block(run_label: str, pivot_r1: pd.DataFrame, pivot_bs: pd.DataFrame) -> None:
    print_section(f"KD vs SFT - {run_label} (positive = KD wins)")
    delta_r1 = kd_vs_sft_delta(pivot_r1)
    delta_bs = kd_vs_sft_delta(pivot_bs)
    summary = pd.DataFrame(
        {
            "dataset": [DATASET_LABELS.get(d, d) for d in delta_r1.index],
            "rouge1_KD_minus_SFT": delta_r1.values,
            "bertscore_KD_minus_SFT": delta_bs.values,
        }
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    wins = (delta_r1 > 0).sum()
    print(f"\nROUGE-1: KD wins {wins}/{len(delta_r1)} datasets")


def save_outputs(
    df_10k: pd.DataFrame,
    df_50k: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat([df_10k, df_50k], ignore_index=True)
    combined.to_csv(out_dir / "all_results_combined.csv", index=False)

    for metric in HEADLINE_METRICS:
        cmp = compare_metric(df_10k, df_50k, metric)
        cmp.to_csv(out_dir / f"compare_{metric}.csv", index=False)

    # Headline pivots per run
    for run_label, df in [("10k", df_10k), ("50k", df_50k)]:
        pivot_metric(df, "rouge1_clean").to_csv(out_dir / f"headline_rouge1_{run_label}.csv")
        pivot_metric(df, "bertscore_f1").to_csv(out_dir / f"headline_bertscore_{run_label}.csv")

    # KD vs SFT delta across scales
    kd_sft_rows = []
    for run_label, df in [("10k", df_10k), ("50k", df_50k)]:
        for metric in ("rouge1_clean", "bertscore_f1"):
            delta = kd_vs_sft_delta(pivot_metric(df, metric))
            for ds, val in delta.items():
                kd_sft_rows.append(
                    {"run": run_label, "metric": metric, "dataset": ds, "KD_minus_SFT": val}
                )
    pd.DataFrame(kd_sft_rows).to_csv(out_dir / "kd_vs_sft_delta_by_run.csv", index=False)

    # Scale effect for student models only
    scale_rows = []
    for model in ("student_KD_LoRA", "student_SFT_LoRA", "student_baseline_0.5B"):
        for metric in ("rouge1_clean", "bertscore_f1"):
            cmp = compare_metric(df_10k, df_50k, metric)
            sub = cmp[cmp["model_label"] == model]
            scale_rows.append(sub)
    pd.concat(scale_rows, ignore_index=True).to_csv(out_dir / "scale_effect_students.csv", index=False)


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.4,
            "grid.linestyle": "--",
            "legend.fontsize": 10,
            "figure.dpi": 120,
        }
    )
    return plt


def make_plots(df_10k: pd.DataFrame, df_50k: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.patches as mpatches
        import numpy as np

        plt = _setup_matplotlib()
    except ImportError:
        print("matplotlib not installed — skip plots (pip install matplotlib)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    DPI = 200

    DS_SHORT = ["CNN/DM\n(in-domain)", "XSum\n(cross)", "SAMSum\n(cross)", "DialogSum\n(cross)"]
    DS_TICK = ["CNN/DM", "XSum", "SAMSum", "DialogSum"]

    MODEL_KEYS = list(MODEL_ORDER)
    MODEL_NAMES = {
        "teacher_7B": "Teacher 7B",
        "student_baseline_0.5B": "Base 0.5B",
        "student_KD_LoRA": "KD-LoRA",
        "student_SFT_LoRA": "SFT-LoRA",
    }
    COLORS = {
        "teacher_7B": "#2c5282",
        "student_baseline_0.5B": "#a0aec0",
        "student_KD_LoRA": "#dd6b20",
        "student_SFT_LoRA": "#c53030",
    }
    KD_COLOR = "#dd6b20"
    SFT_COLOR = "#c53030"
    C10K = "#718096"
    C50K = "#2d3748"

    def vals(df: pd.DataFrame, metric: str, model: str) -> list[float]:
        p = pivot_metric(df, metric)
        return [float(p.loc[model, ds]) for ds in DATASET_ORDER]

    def save(fig, name: str) -> None:
        path = out_dir / name
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved {path.name}")

    def annotate_bars(ax, bars, fmt="{:.1f}", ypad=0.4, fontsize=8):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + ypad,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                fontweight="bold",
            )

    # ── 1. All models — ROUGE-1 grouped bars (10k & 50k) ───────────────────
    for run_label, df in [("10k", df_10k), ("50k", df_50k)]:
        n_m = len(MODEL_KEYS)
        x = np.arange(len(DS_SHORT))
        w = 0.18
        offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * w

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, mkey in enumerate(MODEL_KEYS):
            v = vals(df, "rouge1_clean", mkey)
            bars = ax.bar(
                x + offsets[i],
                v,
                w,
                label=MODEL_NAMES[mkey],
                color=COLORS[mkey],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            annotate_bars(ax, bars)

        ax.set_xticks(x)
        ax.set_xticklabels(DS_SHORT, fontsize=10)
        ax.set_ylabel("ROUGE-1 (clean)")
        ax.set_ylim(0, 48)
        ax.set_title(f"All models — ROUGE-1 [{run_label} training samples]", fontweight="bold", pad=14)
        ax.legend(loc="upper right", ncol=2, framealpha=0.95)
        fig.tight_layout()
        save(fig, f"01_rouge1_all_models_{run_label}.png")

    # ── 2. KD vs SFT only — ROUGE-1 side-by-side 10k vs 50k ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    x = np.arange(4)
    w = 0.35
    for ax, (run_label, df) in zip(axes, [("10k", df_10k), ("50k", df_50k)]):
        kd = vals(df, "rouge1_clean", "student_KD_LoRA")
        sft = vals(df, "rouge1_clean", "student_SFT_LoRA")
        b1 = ax.bar(x - w / 2, kd, w, label="KD-LoRA", color=KD_COLOR, edgecolor="white", zorder=3)
        b2 = ax.bar(x + w / 2, sft, w, label="SFT-LoRA", color=SFT_COLOR, edgecolor="white", zorder=3)
        annotate_bars(ax, b1)
        annotate_bars(ax, b2)
        ax.set_xticks(x)
        ax.set_xticklabels(DS_TICK, fontsize=10)
        ax.set_title(f"{run_label} training", fontweight="bold")
        ax.set_ylim(0, 42)
        ax.legend(loc="upper right")
    axes[0].set_ylabel("ROUGE-1 (clean)")
    fig.suptitle("KD-LoRA vs SFT-LoRA — ROUGE-1 by dataset", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "02_rouge1_kd_vs_sft_10k_50k.png")

    # ── 3. KD − SFT delta (who wins?) ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (run_label, df) in zip(axes, [("10k", df_10k), ("50k", df_50k)]):
        delta = kd_vs_sft_delta(pivot_metric(df, "rouge1_clean")).values
        colors_bar = [KD_COLOR if d > 0 else SFT_COLOR for d in delta]
        bars = ax.barh(np.arange(4), delta, color=colors_bar, height=0.55, zorder=3)
        for bar, d in zip(bars, delta):
            ax.text(
                d + (0.15 if d >= 0 else -0.15),
                bar.get_y() + bar.get_height() / 2,
                f"{d:+.2f}",
                va="center",
                ha="left" if d >= 0 else "right",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_yticks(np.arange(4))
        ax.set_yticklabels(DS_TICK, fontsize=10)
        ax.axvline(0, color="#1a202c", linewidth=1.2)
        ax.set_xlabel("ROUGE-1: KD − SFT  (+ = KD wins)")
        ax.set_title(f"{run_label} training", fontweight="bold")
    kd_patch = mpatches.Patch(color=KD_COLOR, label="KD wins")
    sft_patch = mpatches.Patch(color=SFT_COLOR, label="SFT wins")
    fig.legend(handles=[kd_patch, sft_patch], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("KD vs SFT advantage per dataset", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, "03_kd_minus_sft_rouge1.png")

    # ── 4. Scale effect — 10k → 50k lines (KD & SFT, 4 panels) ─────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    teacher_r1 = vals(df_50k, "rouge1_clean", "teacher_7B")
    base_r1 = vals(df_10k, "rouge1_clean", "student_baseline_0.5B")
    x_line = [0, 1]
    x_labels = ["10k", "50k"]

    for ax, ds_i, title in zip(axes.flat, range(4), DS_TICK):
        kd_line = [
            vals(df_10k, "rouge1_clean", "student_KD_LoRA")[ds_i],
            vals(df_50k, "rouge1_clean", "student_KD_LoRA")[ds_i],
        ]
        sft_line = [
            vals(df_10k, "rouge1_clean", "student_SFT_LoRA")[ds_i],
            vals(df_50k, "rouge1_clean", "student_SFT_LoRA")[ds_i],
        ]
        ax.plot(x_line, kd_line, "o-", color=KD_COLOR, linewidth=2.5, markersize=9, label="KD-LoRA")
        ax.plot(x_line, sft_line, "s-", color=SFT_COLOR, linewidth=2.5, markersize=9, label="SFT-LoRA")
        ax.axhline(base_r1[ds_i], color="#a0aec0", linestyle="--", linewidth=1.5, label="Base 0.5B")
        ax.axhline(teacher_r1[ds_i], color="#2c5282", linestyle=":", linewidth=1.5, label="Teacher 7B")
        for xv, yv in zip(x_line, kd_line):
            ax.text(xv, yv + 0.6, f"{yv:.1f}", ha="center", fontsize=9, color=KD_COLOR, fontweight="bold")
        for xv, yv in zip(x_line, sft_line):
            ax.text(xv, yv - 1.4, f"{yv:.1f}", ha="center", fontsize=9, color=SFT_COLOR, fontweight="bold")
        ax.set_xticks(x_line)
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.set_title(title, fontweight="bold")
        ymin = min(kd_line + sft_line + [base_r1[ds_i]]) - 4
        ymax = max(kd_line + sft_line + [teacher_r1[ds_i]]) + 4
        ax.set_ylim(max(0, ymin), ymax)

    axes[0, 0].set_ylabel("ROUGE-1 (clean)")
    axes[1, 0].set_ylabel("ROUGE-1 (clean)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Scale effect: 10k → 50k training samples", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    save(fig, "04_scale_effect_rouge1.png")

    # ── 5. Scale gain bars (50k − 10k) for KD & SFT ──────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(4)
    w = 0.35
    kd_gain = [
        vals(df_50k, "rouge1_clean", "student_KD_LoRA")[i]
        - vals(df_10k, "rouge1_clean", "student_KD_LoRA")[i]
        for i in range(4)
    ]
    sft_gain = [
        vals(df_50k, "rouge1_clean", "student_SFT_LoRA")[i]
        - vals(df_10k, "rouge1_clean", "student_SFT_LoRA")[i]
        for i in range(4)
    ]
    b1 = ax.bar(x - w / 2, kd_gain, w, label="KD-LoRA gain", color=KD_COLOR, edgecolor="white", zorder=3)
    b2 = ax.bar(x + w / 2, sft_gain, w, label="SFT-LoRA gain", color=SFT_COLOR, edgecolor="white", zorder=3)
    annotate_bars(ax, b1, fmt="{:.2f}", ypad=0.15)
    annotate_bars(ax, b2, fmt="{:.2f}", ypad=0.15)
    ax.axhline(0, color="#1a202c", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(DS_TICK, fontsize=10)
    ax.set_ylabel("ROUGE-1 change (50k − 10k)")
    ax.set_title("How much does scaling training data help?", fontweight="bold", pad=12)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "05_scale_gain_rouge1.png")

    # ── 6. BERTScore F1 — KD vs SFT (10k & 50k) ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, (run_label, df) in zip(axes, [("10k", df_10k), ("50k", df_50k)]):
        kd = vals(df, "bertscore_f1", "student_KD_LoRA")
        sft = vals(df, "bertscore_f1", "student_SFT_LoRA")
        b1 = ax.bar(x - w / 2, kd, w, label="KD-LoRA", color=KD_COLOR, edgecolor="white", zorder=3)
        b2 = ax.bar(x + w / 2, sft, w, label="SFT-LoRA", color=SFT_COLOR, edgecolor="white", zorder=3)
        annotate_bars(ax, b1)
        annotate_bars(ax, b2)
        ax.set_xticks(x)
        ax.set_xticklabels(DS_TICK, fontsize=10)
        ax.set_title(f"{run_label} training", fontweight="bold")
        ax.axhline(0, color="#cbd5e0", linewidth=0.8)
    axes[0].set_ylabel("BERTScore F1 (rescaled)")
    fig.suptitle("KD-LoRA vs SFT-LoRA — BERTScore F1", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "06_bertscore_kd_vs_sft.png")

    # ── 7. Heatmaps — ROUGE-1 for 10k and 50k ────────────────────────────────
    models_hm = ["teacher_7B", "student_baseline_0.5B", "student_KD_LoRA", "student_SFT_LoRA"]
    hm_labels = [MODEL_NAMES[m] for m in models_hm]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    vmin, vmax = 5, 40
    for ax, (run_label, df) in zip(axes, [("10k", df_10k), ("50k", df_50k)]):
        matrix = np.array([vals(df, "rouge1_clean", m) for m in models_hm])
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(4))
        ax.set_xticklabels(DS_TICK, fontsize=10)
        ax.set_yticks(range(4))
        ax.set_yticklabels(hm_labels, fontsize=9)
        for i in range(4):
            for j in range(4):
                v = matrix[i, j]
                txt_color = "white" if v > 28 else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=10, fontweight="bold", color=txt_color)
        ax.set_title(f"{run_label} training", fontweight="bold", pad=10)
        fig.colorbar(im, ax=ax, shrink=0.9, label="ROUGE-1")
    fig.suptitle("ROUGE-1 heatmap — all models", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "07_heatmap_rouge1.png")

    # ── 8. Summary dashboard (2×2) ───────────────────────────────────────────
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    # (a) 50k all models compact
    ax_a = fig.add_subplot(gs[0, 0])
    n_m = 4
    x_a = np.arange(4)
    offsets_a = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * 0.18
    for i, mkey in enumerate(MODEL_KEYS):
        v = vals(df_50k, "rouge1_clean", mkey)
        ax_a.bar(x_a + offsets_a[i], v, 0.18, label=MODEL_NAMES[mkey], color=COLORS[mkey], edgecolor="white")
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(DS_TICK, fontsize=9)
    ax_a.set_ylabel("ROUGE-1")
    ax_a.set_title("50k — all models", fontweight="bold")
    ax_a.legend(fontsize=8, loc="upper right")
    ax_a.set_ylim(0, 42)

    # (b) KD−SFT 10k vs 50k on same axes
    ax_b = fig.add_subplot(gs[0, 1])
    d10 = kd_vs_sft_delta(pivot_metric(df_10k, "rouge1_clean")).values
    d50 = kd_vs_sft_delta(pivot_metric(df_50k, "rouge1_clean")).values
    x_b = np.arange(4)
    ax_b.bar(x_b - 0.2, d10, 0.38, label="10k", color=C10K, edgecolor="white")
    ax_b.bar(x_b + 0.2, d50, 0.38, label="50k", color=C50K, edgecolor="white")
    ax_b.axhline(0, color="black", linewidth=1)
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(DS_TICK, fontsize=9)
    ax_b.set_ylabel("KD − SFT (ROUGE-1)")
    ax_b.set_title("KD advantage shifts with scale", fontweight="bold")
    ax_b.legend()

    # (c) scale gain
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.bar(x_b - 0.2, kd_gain, 0.38, label="KD gain", color=KD_COLOR, edgecolor="white")
    ax_c.bar(x_b + 0.2, sft_gain, 0.38, label="SFT gain", color=SFT_COLOR, edgecolor="white")
    ax_c.axhline(0, color="black", linewidth=1)
    ax_c.set_xticks(x_b)
    ax_c.set_xticklabels(DS_TICK, fontsize=9)
    ax_c.set_ylabel("50k − 10k ROUGE-1")
    ax_c.set_title("Scaling benefit per method", fontweight="bold")
    ax_c.legend()

    # (d) BERTScore 50k KD vs SFT
    ax_d = fig.add_subplot(gs[1, 1])
    kd_bs = vals(df_50k, "bertscore_f1", "student_KD_LoRA")
    sft_bs = vals(df_50k, "bertscore_f1", "student_SFT_LoRA")
    ax_d.bar(x_b - 0.2, kd_bs, 0.38, label="KD-LoRA", color=KD_COLOR, edgecolor="white")
    ax_d.bar(x_b + 0.2, sft_bs, 0.38, label="SFT-LoRA", color=SFT_COLOR, edgecolor="white")
    ax_d.set_xticks(x_b)
    ax_d.set_xticklabels(DS_TICK, fontsize=9)
    ax_d.set_ylabel("BERTScore F1")
    ax_d.set_title("50k — BERTScore (2nd metric)", fontweight="bold")
    ax_d.legend()

    fig.suptitle("KD vs SFT @ 10k vs 50k — summary dashboard", fontsize=15, fontweight="bold", y=1.01)
    save(fig, "08_summary_dashboard.png")

    print(f"\nAll plots written to: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 10k vs 50k notebook-06 eval results.")
    parser.add_argument("--dir-10k", type=Path, default=DEFAULT_10K, help="10k kd_vs_sft_eval_results/")
    parser.add_argument("--dir-50k", type=Path, default=DEFAULT_50K, help="50k kd_vs_sft_eval_results/")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="Output directory for CSVs/plots")
    parser.add_argument("--plots", action="store_true", default=True, help="Generate plots (default: on)")
    parser.add_argument("--no-plots", dest="plots", action="store_false", help="Skip plot generation")
    args = parser.parse_args()

    df_10k = load_results(args.dir_10k, "10k")
    df_50k = load_results(args.dir_50k, "50k")

    print_section("Loaded eval runs (notebook 06 - rouge1_clean + bertscore_f1)")
    print(f"  10k: {args.dir_10k}  ({len(df_10k)} rows)")
    print(f"  50k: {args.dir_50k}  ({len(df_50k)} rows)")

    for run_label, df in [("10k", df_10k), ("50k", df_50k)]:
        print_pivot(f"ROUGE-1 (clean) - {run_label}", pivot_metric(df, "rouge1_clean"))
        print_pivot(f"BERTScore F1 - {run_label}", pivot_metric(df, "bertscore_f1"))
        print_kd_sft_block(run_label, pivot_metric(df, "rouge1_clean"), pivot_metric(df, "bertscore_f1"))

        lift = training_lift(pivot_metric(df, "rouge1_clean"), "rouge1_clean")
        print_section(f"Training lift over baseline (ROUGE-1) - {run_label}")
        print(lift.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print_section("Scale effect: 50k - 10k (student models)")
    for model in ("student_KD_LoRA", "student_SFT_LoRA"):
        cmp = compare_metric(df_10k, df_50k, "rouge1_clean")
        sub = cmp[cmp["model_label"] == model][
            ["dataset", "value_10k", "value_50k", "delta_50k_minus_10k"]
        ].copy()
        sub["dataset"] = sub["dataset"].map(lambda d: DATASET_LABELS.get(d, d))
        print(f"\n{model}:")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print_section("KD vs SFT winner flip? (ROUGE-1)")
    d10 = kd_vs_sft_delta(pivot_metric(df_10k, "rouge1_clean"))
    d50 = kd_vs_sft_delta(pivot_metric(df_50k, "rouge1_clean"))
    flip = pd.DataFrame(
        {
            "dataset": [DATASET_LABELS.get(d, d) for d in d10.index],
            "KD_minus_SFT_10k": d10.values,
            "KD_minus_SFT_50k": d50.values,
        }
    )
    print(flip.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    save_outputs(df_10k, df_50k, args.out)
    print(f"\nWrote comparison CSVs to: {args.out.resolve()}")

    if args.plots:
        make_plots(df_10k, df_50k, args.out)
    else:
        print("Plots skipped (--no-plots).")


if __name__ == "__main__":
    main()
