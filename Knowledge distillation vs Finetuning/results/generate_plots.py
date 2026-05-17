import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent

# ── Raw data ──────────────────────────────────────────────────────────────────
DATASETS = ["CNN/DailyMail\n(in-domain)", "XSum\n(cross-domain)", "SAMSum\n(cross-domain)", "DialogSum\n(cross-domain)"]
DS_KEYS  = ["cnn_dailymail", "xsum", "samsum", "dialogsum"]

DATA = {
    "10k": {
        "Teacher 7B":       [37.405, 28.476, 41.662, 34.026],
        "Student Base 0.5B":[25.175, 14.814, 24.656, 20.456],
        "KD-LoRA 0.5B":     [31.017, 22.749, 25.554, 21.413],
        "SFT-LoRA 0.5B":    [31.765, 20.354, 26.865, 23.679],
    },
    "50k": {
        "Teacher 7B":       [37.462, 28.467, 41.662, 33.915],
        "Student Base 0.5B":[24.830, 14.737, 24.656, 20.507],
        "KD-LoRA 0.5B":     [32.889, 22.964, 29.955, 25.474],
        "SFT-LoRA 0.5B":    [32.204, 20.102, 25.435, 20.942],
    },
}

ROUGE2 = {
    "10k": {
        "Teacher 7B":       [12.481, 7.688, 15.674, 11.287],
        "Student Base 0.5B":[ 9.980, 1.410,  6.834,  4.622],
        "KD-LoRA 0.5B":     [10.275, 4.735,  7.719,  6.555],
        "SFT-LoRA 0.5B":    [13.343, 3.330,  8.311,  6.690],
    },
    "50k": {
        "Teacher 7B":       [12.647, 7.719, 15.674, 11.083],
        "Student Base 0.5B":[ 9.699, 1.443,  6.834,  4.589],
        "KD-LoRA 0.5B":     [10.749, 4.785,  8.656,  7.409],
        "SFT-LoRA 0.5B":    [13.803, 3.186,  7.679,  5.894],
    },
}

COLORS = {
    "Teacher 7B":        "#4e79a7",
    "Student Base 0.5B": "#aec7e8",
    "KD-LoRA 0.5B":      "#f28e2b",
    "SFT-LoRA 0.5B":     "#e15759",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# ── 1. Grouped bar: ROUGE-1 per experiment ────────────────────────────────────
def grouped_bar(scale: str, filename: str):
    models = list(DATA[scale].keys())
    n_ds   = len(DATASETS)
    n_m    = len(models)
    x      = np.arange(n_ds)
    w      = 0.18
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * w

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, model in enumerate(models):
        vals = DATA[scale][model]
        bars = ax.bar(x + offsets[i], vals, w, color=COLORS[model],
                      label=model, zorder=3, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=10)
    ax.set_ylabel("ROUGE-1", fontsize=11)
    ax.set_ylim(0, 50)
    ax.set_title(f"ROUGE-1 — KD vs SFT vs Baselines  [{scale} training samples]",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")

grouped_bar("10k", "rouge1_grouped_10k.png")
grouped_bar("50k", "rouge1_grouped_50k.png")

# ── 2. KD vs SFT delta heatmap (10k vs 50k) ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
ds_labels = ["CNN/DM\n(in-domain)", "XSum\n(cross)", "SAMSum\n(cross)", "DialogSum\n(cross)"]

for ax, scale in zip(axes, ["10k", "50k"]):
    kd  = np.array(DATA[scale]["KD-LoRA 0.5B"])
    sft = np.array(DATA[scale]["SFT-LoRA 0.5B"])
    delta = kd - sft  # positive = KD wins

    colors = ["#f28e2b" if d > 0 else "#e15759" for d in delta]
    bars = ax.barh(np.arange(len(ds_labels)), delta, color=colors, zorder=3,
                   edgecolor="white", linewidth=0.5)
    for bar, d in zip(bars, delta):
        ax.text(d + (0.08 if d >= 0 else -0.08), bar.get_y() + bar.get_height() / 2,
                f"{d:+.2f}", va="center", ha="left" if d >= 0 else "right",
                fontsize=9, fontweight="bold")

    ax.set_yticks(np.arange(len(ds_labels)))
    ax.set_yticklabels(ds_labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlim(-4, 6)
    ax.set_xlabel("ROUGE-1  (KD − SFT)", fontsize=10)
    ax.set_title(f"{scale} samples", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.35, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

kd_patch  = mpatches.Patch(color="#f28e2b", label="KD wins")
sft_patch = mpatches.Patch(color="#e15759", label="SFT wins")
fig.legend(handles=[kd_patch, sft_patch], loc="lower center",
           ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.04), framealpha=0.9)
fig.suptitle("KD-LoRA vs SFT-LoRA Advantage  (ROUGE-1 delta)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(OUT / "kd_vs_sft_delta.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved kd_vs_sft_delta.png")

# ── 3. Scale effect: 10k → 50k for KD and SFT ───────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=False)
ds_short = ["CNN/DM\n(in-domain)", "XSum\n(cross)", "SAMSum\n(cross)", "DialogSum\n(cross)"]
scales   = ["10k", "50k"]
x_pos    = [0, 1]

for ax, ds_i, label in zip(axes, range(4), ds_short):
    kd_vals  = [DATA[s]["KD-LoRA 0.5B"][ds_i]  for s in scales]
    sft_vals = [DATA[s]["SFT-LoRA 0.5B"][ds_i] for s in scales]
    base     = DATA["10k"]["Student Base 0.5B"][ds_i]
    teacher  = DATA["10k"]["Teacher 7B"][ds_i]

    ax.plot(x_pos, kd_vals,  "o-", color="#f28e2b", linewidth=2.2, markersize=8, label="KD-LoRA", zorder=4)
    ax.plot(x_pos, sft_vals, "s-", color="#e15759", linewidth=2.2, markersize=8, label="SFT-LoRA", zorder=4)
    ax.axhline(base,    color="#aec7e8", linewidth=1.4, linestyle="--", label="Base 0.5B")
    ax.axhline(teacher, color="#4e79a7", linewidth=1.4, linestyle=":",  label="Teacher 7B")

    for v, x in zip(kd_vals, x_pos):
        ax.text(x, v + 0.35, f"{v:.1f}", ha="center", fontsize=8, color="#f28e2b", fontweight="bold")
    for v, x in zip(sft_vals, x_pos):
        ax.text(x, v - 0.9, f"{v:.1f}", ha="center", fontsize=8, color="#e15759", fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["10k", "50k"], fontsize=10)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_ylabel("ROUGE-1" if ds_i == 0 else "", fontsize=10)
    ymin = min(kd_vals + sft_vals) - 3
    ymax = max(kd_vals + sft_vals + [teacher]) + 3
    ax.set_ylim(ymin, ymax)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4,
           fontsize=9, bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
fig.suptitle("Scale Effect: ROUGE-1 at 10k vs 50k Training Samples", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(OUT / "scale_effect.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved scale_effect.png")

# ── 4. Full ROUGE heatmap (50k, all metrics) ─────────────────────────────────
import matplotlib.colors as mcolors

METRICS = {
    "ROUGE-1 (50k)": DATA["50k"],
    "ROUGE-2 (50k)": ROUGE2["50k"],
}
models_order = ["Teacher 7B", "Student Base 0.5B", "KD-LoRA 0.5B", "SFT-LoRA 0.5B"]
ds_short4    = ["CNN/DM", "XSum", "SAMSum", "DialogSum"]

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
for ax, (title, metric_data) in zip(axes, METRICS.items()):
    matrix = np.array([[metric_data[m][i] for i in range(4)] for m in models_order])
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                   vmin=matrix.min() - 1, vmax=matrix.max() + 1)
    ax.set_xticks(range(4)); ax.set_xticklabels(ds_short4, fontsize=10)
    ax.set_yticks(range(4)); ax.set_yticklabels(models_order, fontsize=9)
    for i in range(4):
        for j in range(4):
            v = matrix[i, j]
            color = "white" if v > (matrix.max() * 0.72) else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=color)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Score")

fig.suptitle("Score Heatmap — 50k Experiment", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "heatmap_50k.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved heatmap_50k.png")

print("\nAll plots generated.")
