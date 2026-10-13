#!/usr/bin/env python3
"""Small, reproducible Mac smoke benchmark for the experimental KV cache branch."""

import csv
import json
import os
import random
import re
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BINARY = Path(os.environ.get("TURBOQUANT_BINARY", ROOT / "llama-cpp-turboquant/build/bin/llama-completion")).expanduser().resolve()
MODEL = Path(os.environ.get("TURBOQUANT_MODEL", ROOT / "models/qwen2.5-1.5b-instruct-q8_0.gguf")).expanduser().resolve()
OUT = Path(os.environ.get("TURBOQUANT_OUTPUT_DIR", ROOT / "quick-benchmark")).expanduser().resolve()
PROMPTS = OUT / "prompts"
LOGS = OUT / "logs"
RESULTS = OUT / "results.csv"
SEEDS = (11, 29)
SIZES = (("2k", 85, 3072), ("4k", 175, 5120), ("7k", 285, 8192))
POSITIONS = (("early", 0.15), ("late", 0.85))
ARMS = ("fp16", "turbo4", "tri", "turbo4_tri")
TARGET_CODES = {11: "LIME-742", 29: "NOVA-615"}
DISTRACTOR_PROJECTS = ("MAPLE", "RIVER", "SOLAR", "CLOUD", "DELTA")
FIELDS = (
    "case", "arm", "seed", "position", "size", "records", "context", "budget",
    "prompt_tokens", "evicted_tokens", "answer", "first_code_correct", "exact_match",
    "kv_mib", "metal_self_mib", "peak_rss_mib", "prefill_ms", "decode_ms",
    "wall_ms", "returncode", "error",
)


def make_prompt(n_records: int, fraction: float, seed: int) -> str:
    rng = random.Random(seed)
    target_idx = round(n_records * fraction)
    lines = ["Read these synthetic records. Answer the question using the matching project only."]
    for i in range(1, n_records + 1):
        if i == target_idx:
            line = f"Record {i:03d}: The audit code for project ORBIT is {TARGET_CODES[seed]}."
        elif i % 13 == 0:
            project = rng.choice(DISTRACTOR_PROJECTS)
            code = f"{rng.choice(('PINE', 'MINT', 'GOLD', 'SNOW'))}-{rng.randrange(100, 1000)}"
            line = f"Record {i:03d}: The audit code for project {project} is {code}."
        else:
            line = (
                f"Record {i:03d}: The local archive contains routine sample entry "
                f"number {i:03d} with no audit code."
            )
        lines.append(line)
    lines += ["", "Question: What is the audit code for project ORBIT? Answer with the code only.", "Answer:"]
    return "\n".join(lines)


def extract(pattern: str, data: str, cast=float):
    match = re.search(pattern, data)
    return cast(match.group(1)) if match else None


def run_one(case: dict, arm: str) -> dict:
    prompt_path = PROMPTS / f"{case['case']}.txt"
    command = [
        "/usr/bin/time", "-l", str(BINARY),
        "-m", str(MODEL), "-f", str(prompt_path),
        "-c", str(case["context"]), "-n", "12",
        "-ngl", "99", "-fa", "on", "-fit", "off",
        "-ctk", "q8_0" if "turbo4" in arm else "f16",
        "-ctv", "turbo4" if "turbo4" in arm else "f16",
        "-s", "42", "--no-conversation", "--no-display-prompt", "--temp", "0",
    ]
    if "tri" in arm:
        command += [
            "--triatt-budget", str(case["budget"]),
            "--triatt-hybrid", "2", "--triatt-prefix", "128",
        ]
    env = dict(os.environ, DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer")
    begin = time.perf_counter()
    proc = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    error = ""
    try:
        stdout, stderr = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        error = "timeout after 60 s"
    wall_ms = round((time.perf_counter() - begin) * 1000, 1)
    log_stem = f"{case['case']}__{arm}"
    (LOGS / f"{log_stem}.stdout.txt").write_text(stdout)
    (LOGS / f"{log_stem}.stderr.txt").write_text(stderr)
    answer = stdout.strip().replace("\n", " ")
    first_code = re.search(r"[A-Z]{3,6}-\d{3}", answer)
    first_code_correct = bool(first_code and first_code.group(0) == case["gold"])
    exact_match = answer.rstrip(" .") == case["gold"]
    if proc.returncode != 0 and not error:
        error = f"exit {proc.returncode}"
    return {
        **{k: case[k] for k in ("case", "seed", "position", "size", "records", "context", "budget")},
        "arm": arm,
        "prompt_tokens": extract(r"prompt eval time\s*=\s*[\d.]+ ms /\s*(\d+) tokens", stderr, int),
        "evicted_tokens": sum(int(x) for x in re.findall(r"evict: evicted (\d+)/", stderr)) or None,
        "answer": answer,
        "first_code_correct": int(first_code_correct),
        "exact_match": int(exact_match),
        "kv_mib": extract(r"KV buffer size =\s*([\d.]+) MiB", stderr),
        "metal_self_mib": extract(r"MTL0 \(Apple[^)]*\).*?\(\s*([\d.]+) =", stderr),
        "peak_rss_mib": round(extract(r"(\d+)\s+maximum resident set size", stderr, int) / 1048576, 2)
        if re.search(r"(\d+)\s+maximum resident set size", stderr) else None,
        "prefill_ms": extract(r"prompt eval time\s*=\s*([\d.]+) ms", stderr),
        "decode_ms": extract(r"(?<!prompt )eval time\s*=\s*([\d.]+) ms", stderr),
        "wall_ms": wall_ms,
        "returncode": proc.returncode,
        "error": error,
    }


def main():
    if not BINARY.is_file() or not MODEL.is_file():
        raise SystemExit(
            "Set TURBOQUANT_BINARY and TURBOQUANT_MODEL to the built llama-completion "
            "binary and downloaded GGUF model."
        )
    PROMPTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    cases = []
    for size, records, context in SIZES:
        for position, fraction in POSITIONS:
            for seed in SEEDS:
                name = f"{size}_{position}_s{seed}"
                case = {
                    "case": name, "seed": seed, "position": position,
                    "size": size, "records": records, "context": context,
                    "budget": round(records * 17), "gold": TARGET_CODES[seed],
                }
                (PROMPTS / f"{name}.txt").write_text(make_prompt(records, fraction, seed))
                cases.append(case)
    (OUT / "cases.json").write_text(json.dumps(cases, indent=2))

    with RESULTS.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for case in cases:
            for arm in ARMS:
                row = run_one(case, arm)
                writer.writerow(row)
                file.flush()
                print(
                    f"{case['case']:16} {arm:10} "
                    f"tokens={str(row['prompt_tokens']):>4} "
                    f"kv={str(row['kv_mib']):>6} MiB "
                    f"prefill={str(row['prefill_ms']):>7} ms "
                    f"correct={row['first_code_correct']} exit={row['returncode']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
