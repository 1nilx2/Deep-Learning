#!/usr/bin/env python3
"""Extend the local TurboQuant retrieval benchmark to longer contexts."""

import csv
import json
from pathlib import Path

import run_benchmark as bench


def main():
    if not bench.BINARY.is_file() or not bench.MODEL.is_file():
        raise SystemExit("Benchmark binary or model is missing")

    bench.OUT = Path(__file__).resolve().parent / "long-extension"
    bench.PROMPTS = bench.OUT / "prompts"
    bench.LOGS = bench.OUT / "logs"
    bench.RESULTS = bench.OUT / "results.csv"
    bench.PROMPTS.mkdir(parents=True, exist_ok=True)
    bench.LOGS.mkdir(parents=True, exist_ok=True)

    cases = []
    for size, records, context in (("16k", 700, 18432), ("30k", 1300, 32768)):
        for position, fraction in (("early", 0.15), ("late", 0.85)):
            name = f"{size}_{position}_s11"
            case = {
                "case": name,
                "seed": 11,
                "position": position,
                "size": size,
                "records": records,
                "context": context,
                "budget": 0,
                "gold": bench.TARGET_CODES[11],
            }
            (bench.PROMPTS / f"{name}.txt").write_text(
                bench.make_prompt(records, fraction, 11)
            )
            cases.append(case)
    (bench.OUT / "cases.json").write_text(json.dumps(cases, indent=2))

    with bench.RESULTS.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=bench.FIELDS, lineterminator="\n")
        writer.writeheader()
        for case in cases:
            for arm in ("fp16", "turbo4"):
                row = bench.run_one(case, arm)
                writer.writerow(row)
                file.flush()
                print(
                    f"{case['case']:16} {arm:7} "
                    f"tokens={row['prompt_tokens']} "
                    f"kv={row['kv_mib']} MiB "
                    f"metal={row['metal_self_mib']} MiB "
                    f"prefill={row['prefill_ms']} ms "
                    f"correct={row['first_code_correct']} "
                    f"exit={row['returncode']} error={row['error']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
