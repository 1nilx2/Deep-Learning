"""Build a checked Markdown result page from untouched-test JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path(__file__).parent / "runs")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "RESULTS.md")
    args = parser.parse_args()
    names = [
        ("Laya, original", "laya_base_test.json"),
        ("Laya, fine-tuned", "laya_test.json"),
        ("CrossEncoder, fine-tuned", "cross_encoder_test.json"),
    ]
    reports = [(name, json.loads((args.runs / filename).read_text())) for name, filename in names]
    manifest = reports[0][1]["data_manifest"]
    for name, report in reports:
        if report["data_manifest"] != manifest:
            raise ValueError(f"{name} used a different dataset manifest")
        if report["n"] != manifest["total_rows"]["test"] or report["exploratory_run"]:
            raise ValueError(f"{name} is not a complete official-test run")
    devices = {report["device"] for _, report in reports}
    batch_sizes = {report["batch_size"] for _, report in reports}
    if len(devices) != 1 or len(batch_sizes) != 1:
        raise ValueError("Runtime comparisons require the same device and batch size")
    device = next(iter(devices))
    batch_size = next(iter(batch_sizes))
    labels = reports[0][1]["label_order"]
    lines = [
        "# BANKING77 payment intent comparison",
        "",
        "The [experiment protocol](README.md) fixes eight related payment intents, 50 training and 20 validation requests per intent, and all 40 official test requests per intent. This is an eight-intent subset of BANKING77, not the full 77-intent task.",
        "",
        f"Dataset source commit: `{manifest['source_commit']}`. Test examples: {manifest['total_rows']['test']}.",
        "",
        "| Model | Accuracy | Macro-F1 | Log loss ↓ | Brier ↓ | ECE ↓ | ms/request ↓ | Parameters |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, report in reports:
        lines.append(
            f"| {name} | {report['accuracy']:.3f} | {report['macro_f1']:.3f} | "
            f"{report['log_loss']:.3f} | {report['brier']:.3f} | "
            f"{report['ece_10_bins']:.3f} | {report['milliseconds_per_request']:.1f} | "
            f"{report.get('parameters', 0):,} |"
        )
    lines += ["", "## F1 by intent", "", "| Intent | Laya original | Laya fine-tuned | CrossEncoder |",
              "| --- | ---: | ---: | ---: |"]
    for label in labels:
        scores = [report["per_class_f1"][label] for _, report in reports]
        lines.append(f"| `{label}` | {scores[0]:.3f} | {scores[1]:.3f} | {scores[2]:.3f} |")
    original = reports[0][1]
    fine_tuned = reports[1][1]
    cross_encoder = reports[2][1]
    original_correct = round(original["accuracy"] * original["n"])
    laya_correct = round(fine_tuned["accuracy"] * fine_tuned["n"])
    cross_correct = round(cross_encoder["accuracy"] * cross_encoder["n"])
    lines += [
        "", "## Observations", "",
        f"Laya classified {original_correct}/{original['n']} test requests correctly before fine-tuning and {laya_correct}/{fine_tuned['n']} afterward, an increase of {laya_correct - original_correct} correct requests. CrossEncoder classified {cross_correct}/{cross_encoder['n']} correctly. The {abs(laya_correct - cross_correct)}-request gap between the fine-tuned models is small on this test set and does not establish a reliable ranking.",
        "",
        f"CrossEncoder used {cross_encoder['parameters']:,} parameters and {cross_encoder['milliseconds_per_request']:.1f} ms/request; fine-tuned Laya used {fine_tuned['parameters']:,} parameters and {fine_tuned['milliseconds_per_request']:.1f} ms/request. These are single local CPU timing runs, not a hardware-independent throughput claim.",
        "",
    ]
    for name, report in reports[1:]:
        matrix = report["confusion_matrix"]
        mistakes = sorted(
            ((matrix[i][j], labels[i], labels[j])
             for i in range(len(labels)) for j in range(len(labels)) if i != j),
            reverse=True,
        )
        count, truth, prediction = mistakes[0]
        lines.append(f"The most frequent {name} confusion was `{truth}` → `{prediction}` ({count} requests).")
    lines += [
        "", "## Reading these results", "",
        f"All three runs used `{device}`, batch size {batch_size}, the same question wording, option descriptions, and official test rows. The CrossEncoder scores eight pairs per request; Laya scores one eight-option question. The starting checkpoints differ in size and training history, so this is a system comparison.",
        "",
        "Laya reserved 40 of its 400 allocated training requests for probability calibration, leaving 360 gradient-training items. CrossEncoder used all 400 for gradient training and fitted temperature on validation logits. Compare both quality and probability metrics with this protocol difference in mind. The benchmark does not measure out-of-scope detection or generalization to all 77 BANKING77 intents.",
        "",
        "The model checkpoints and raw data are excluded from Git; rerun the commands in [README.md](README.md) to regenerate them.",
        "",
    ]
    args.out.write_text("\n".join(lines))
    print(args.out)


if __name__ == "__main__":
    main()
