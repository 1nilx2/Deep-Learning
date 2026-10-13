#!/usr/bin/env python3
"""Compact GitHub spotlight charts from the public benchmark CSV."""

import csv
import os
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("SPOTLIGHT_OUTPUT_DIR", ROOT))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ROWS = list(csv.DictReader((ROOT / "results.csv").open(newline="")))
SIZES = ("2k", "4k", "7k")
LABELS = ("~2k", "~4k", "~6.5k")
BG = "#0d1117"
FG = "#e6edf3"
MUTED = "#8b949e"
GRID = "#30363d"
FP16 = "#58a6ff"
TURBO = "#3fbba9"
MODEL = "#485360"
COMPUTE = "#718092"


def median(size, arm, field):
    return statistics.median(float(row[field]) for row in ROWS if row["size"] == size and row["arm"] == arm)


def axis_style(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name, description):
    path = OUTPUT_DIR / name
    fig.savefig(path, format="svg", facecolor=BG, metadata={"Title": description})
    print(path)
    plt.close(fig)


x = (0, 1, 2)
baseline_kv = [median(size, "fp16", "kv_mib") for size in SIZES]
turbo_kv = [median(size, "turbo4", "kv_mib") for size in SIZES]
baseline_total = [median(size, "fp16", "metal_self_mib") for size in SIZES]
turbo_total = [median(size, "turbo4", "metal_self_mib") for size in SIZES]


fig, ax = plt.subplots(figsize=(12.4, 3.6), dpi=160)
fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.23, top=0.67)
axis_style(ax)
fig.text(0.07, 0.90, "TurboQuant: KV cache allocation", color=FG,
         fontsize=19, fontweight="bold")
fig.text(0.07, 0.79, "M1 Max · Qwen2.5-1.5B · median of 4 prompts per length",
         color=MUTED, fontsize=10)
ax.plot(x, baseline_kv, color=FP16, linewidth=3, marker="o", markersize=7,
        label="FP16 K/V")
ax.plot(x, turbo_kv, color=TURBO, linewidth=3, marker="o", markersize=7,
        label="Q8 K + TurboQuant 4-bit V")
ax.fill_between(x, turbo_kv, baseline_kv, color=TURBO, alpha=0.08)
ax.set_xlim(-0.12, 2.12)
ax.set_ylim(0, 252)
ax.set_xticks(x, LABELS)
ax.set_ylabel("MiB", color=MUTED, fontsize=10)
for i, (base, turbo) in enumerate(zip(baseline_kv, turbo_kv)):
    ax.annotate(f"{base:.0f}", (i, base), xytext=(0, 8), textcoords="offset points",
                ha="center", color=FP16, fontweight="bold", fontsize=10)
    ax.annotate(f"{turbo:.0f}", (i, turbo), xytext=(0, -16), textcoords="offset points",
                ha="center", color=TURBO, fontweight="bold", fontsize=10)
fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=2,
           frameon=False, labelcolor=FG, fontsize=10)
save(fig, "turboquant-kv-cache.svg", "KV cache allocation by prompt length: FP16 versus TurboQuant")


fig, ax = plt.subplots(figsize=(12.4, 4.0), dpi=160)
fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.25, top=0.70)
axis_style(ax)
fig.text(0.07, 0.91, "What changes in the full model footprint?", color=FG,
         fontsize=19, fontweight="bold")
fig.text(0.07, 0.81, "Metal allocation = model weights + compute buffers + KV cache",
         color=MUTED, fontsize=10)
base_model = 1801  # Backend-reported model allocation; same in all 48 runs.
compute = [round(baseline_total[i] - base_model - baseline_kv[i]) for i in x]
centers = (0, 1.35, 2.70)
width = 0.33
for i, center in enumerate(centers):
    for xpos, total, kv, color in (
        (center - 0.20, baseline_total[i], baseline_kv[i], FP16),
        (center + 0.20, turbo_total[i], turbo_kv[i], TURBO),
    ):
        ax.bar(xpos, base_model, width, color=MODEL)
        ax.bar(xpos, compute[i], width, bottom=base_model, color=COMPUTE)
        ax.bar(xpos, kv, width, bottom=base_model + compute[i], color=color)
        ax.annotate(f"{total:,.0f}", (xpos, total), xytext=(0, 4),
                    textcoords="offset points", ha="center", color=color,
                    fontweight="bold", fontsize=10)
    saving = baseline_total[i] - turbo_total[i]
    ax.text(center, 2770, f"−{saving:.0f} MiB / {saving / baseline_total[i] * 100:.1f}%",
            ha="center", color=TURBO, fontweight="bold", fontsize=10)
ax.set_xlim(-0.60, 3.30)
ax.set_ylim(0, 2900)
ax.set_xticks(centers, LABELS)
ax.set_ylabel("MiB", color=MUTED, fontsize=10)
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
fig.legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, color=MODEL, label="Model weights"),
        plt.Rectangle((0, 0), 1, 1, color=COMPUTE, label="Compute buffers"),
        plt.Rectangle((0, 0), 1, 1, color=FP16, label="FP16 KV"),
        plt.Rectangle((0, 0), 1, 1, color=TURBO, label="TurboQuant KV"),
    ],
    loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=4,
    frameon=False, labelcolor=FG, fontsize=9,
)
save(fig, "turboquant-total-metal.svg", "Total Metal model allocation by prompt length")
