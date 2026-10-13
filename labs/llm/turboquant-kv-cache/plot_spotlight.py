#!/usr/bin/env python3
"""Render the three TurboQuant GitHub spotlight charts from published CSVs."""

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
with (ROOT / "results.csv").open(newline="") as file:
    SHORT_ROWS = list(csv.DictReader(file))
with (ROOT / "results_long.csv").open(newline="") as file:
    LONG_ROWS = list(csv.DictReader(file))

SIZES = ("7k", "16k", "30k")
LABELS = ("6.5k", "16.0k", "30.3k")
CAPACITIES = (8192, 18432, 32768)
BG = "#0d1117"
FG = "#e6edf3"
MUTED = "#8b949e"
GRID = "#30363d"
FP16 = "#58a6ff"
TURBO = "#3fbba9"
MODEL = "#485360"
COMPUTE = "#718092"


def median(size, arm, field):
    rows = SHORT_ROWS if size == "7k" else LONG_ROWS
    matching = [float(row[field]) for row in rows
                if row["size"] == size and row["arm"] == arm and row["returncode"] == "0"]
    if not matching:
        raise ValueError(f"Missing successful result: {size} {arm} {field}")
    return statistics.median(matching)


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
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    preview_dir = os.environ.get("SPOTLIGHT_PREVIEW_DIR")
    if preview_dir:
        Path(preview_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(preview_dir) / name.replace(".svg", ".png"),
                    format="png", dpi=160, facecolor=BG)
    print(path)
    plt.close(fig)


input_k = tuple(median(size, "fp16", "prompt_tokens") / 1000 for size in SIZES)
baseline_kv = [median(size, "fp16", "kv_mib") for size in SIZES]
turbo_kv = [median(size, "turbo4", "kv_mib") for size in SIZES]
baseline_total = [median(size, "fp16", "metal_self_mib") for size in SIZES]
turbo_total = [median(size, "turbo4", "metal_self_mib") for size in SIZES]


# 1. Measured KV allocation through the longest supported context.
fig, ax = plt.subplots(figsize=(12.4, 3.9), dpi=160)
fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.29, top=0.68)
axis_style(ax)
fig.text(0.07, 0.91, "Measured KV cache allocation", color=FG,
         fontsize=19, fontweight="bold")
fig.text(0.07, 0.80, "M1 Max · Qwen2.5-1.5B · x-axis: actual input tokens",
         color=MUTED, fontsize=10)
ax.plot(input_k, baseline_kv, color=FP16, linewidth=3, marker="o", markersize=7,
        label="FP16 K/V")
ax.plot(input_k, turbo_kv, color=TURBO, linewidth=3, marker="o", markersize=7,
        label="Q8 K + TurboQuant 4-bit V")
ax.fill_between(input_k, turbo_kv, baseline_kv, color=TURBO, alpha=0.08)
ax.set_xlim(3.5, 33.5)
ax.set_ylim(0, 1050)
ax.set_xticks(input_k, LABELS)
ax.set_ylabel("MiB", color=MUTED, fontsize=10)
for i, (base, turbo) in enumerate(zip(baseline_kv, turbo_kv)):
    ax.annotate(f"{base:.0f}", (input_k[i], base), xytext=(0, 8), textcoords="offset points",
                ha="center", color=FP16, fontweight="bold", fontsize=10)
    ax.annotate(f"{turbo:.0f}", (input_k[i], turbo), xytext=(0, -10), textcoords="offset points",
                ha="center", color=TURBO, fontweight="bold", fontsize=10)
fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2,
           frameon=False, labelcolor=FG, fontsize=10)
save(fig, "turboquant-kv-cache.svg", "Measured KV cache allocation through 30.3k tokens")


# 2. A storage-only projection for a capacity the model cannot run.
projected_fp16_gib = baseline_kv[-1] * (1_000_000 / CAPACITIES[-1]) / 1024
projected_turbo_gib = turbo_kv[-1] * (1_000_000 / CAPACITIES[-1]) / 1024
fig, ax = plt.subplots(figsize=(12.4, 3.9), dpi=160)
fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.23, top=0.68)
axis_style(ax)
fig.text(0.07, 0.91, "If KV storage scaled to 1M tokens", color=FG,
         fontsize=19, fontweight="bold")
fig.text(0.07, 0.80, "Projection from measured bytes/token · KV only · not a 1M model run",
         color=MUTED, fontsize=10)
bars = ax.bar((0.4, 1.4), (projected_fp16_gib, projected_turbo_gib),
              width=0.48, color=(FP16, TURBO), hatch="//", edgecolor=BG)
for bar, value in zip(bars, (projected_fp16_gib, projected_turbo_gib)):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.1f} GiB",
            ha="center", color=FG, fontsize=13, fontweight="bold")
ax.set_xlim(-0.08, 1.88)
ax.set_ylim(0, 31)
ax.set_xticks((0.4, 1.4), ("FP16 K/V", "Q8 K + TurboQuant V"))
ax.set_ylabel("GiB", color=MUTED, fontsize=10)
fig.text(0.50, 0.045, "Projected saving: 16.1 GiB per sequence  ·  model native limit: 32,768 tokens",
         ha="center", color=MUTED, fontsize=10)
save(fig, "turboquant-1m-projection.svg", "Hypothetical KV storage at one million tokens")


# 3. The measured KV saving in relation to the rest of the model allocation.
fig, ax = plt.subplots(figsize=(12.4, 4.4), dpi=160)
fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.28, top=0.67)
axis_style(ax)
fig.text(0.07, 0.92, "What changes in the full model footprint?", color=FG,
         fontsize=19, fontweight="bold")
fig.text(0.07, 0.82, "Numbers above bars = total Metal allocation (MiB); colored tops = KV cache",
         color=MUTED, fontsize=10)
fig.text(0.07, 0.75, "At 30.3k input: TurboQuant KV 357 / total 3,829 MiB = 9.3%",
         color=MUTED, fontsize=10)
base_model = 1801  # Backend-reported model allocation; unchanged across these runs.
width = 2.4
offset = 1.35
for i, center in enumerate(input_k):
    for xpos, total, kv, color in (
        (center - offset, baseline_total[i], baseline_kv[i], FP16),
        (center + offset, turbo_total[i], turbo_kv[i], TURBO),
    ):
        compute = total - base_model - kv
        ax.bar(xpos, base_model, width, color=MODEL)
        ax.bar(xpos, compute, width, bottom=base_model, color=COMPUTE)
        ax.bar(xpos, kv, width, bottom=base_model + compute, color=color)
        if i == 2:
            ax.text(xpos, total - kv / 2, f"KV {kv:.0f}", ha="center", va="center",
                    color=BG, fontsize=8, fontweight="bold")
        ax.annotate(f"Total {total:,.0f}", (xpos, total), xytext=(0, 4),
                    textcoords="offset points", ha="center", color=color,
                    fontweight="bold", fontsize=10)
    saving = baseline_total[i] - turbo_total[i]
    ax.text(center, 4820, f"−{saving:.0f} MiB / {saving / baseline_total[i] * 100:.1f}%",
            ha="center", color=TURBO, fontweight="bold", fontsize=10)
ax.set_xlim(3.5, 33.5)
ax.set_ylim(0, 5100)
ax.set_xticks(input_k, LABELS)
ax.set_ylabel("MiB", color=MUTED, fontsize=10)
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
fig.legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, color=MODEL, label="Model weights"),
        plt.Rectangle((0, 0), 1, 1, color=COMPUTE, label="Compute buffers"),
        plt.Rectangle((0, 0), 1, 1, color=FP16, label="FP16 KV"),
        plt.Rectangle((0, 0), 1, 1, color=TURBO, label="TurboQuant KV"),
    ],
    loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=4,
    frameon=False, labelcolor=FG, fontsize=9,
)
save(fig, "turboquant-total-metal.svg", "Measured total Metal allocation through 30.3k tokens")
