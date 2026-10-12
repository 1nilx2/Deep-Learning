"""Evaluate both systems on the same untouched test split."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import candidate_texts, load_question, load_rows, metrics


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def cross_encoder_logits(model, tokenizer, rows: list[dict], options: list[str],
                         batch_size: int, max_length: int, device: str) -> tuple[np.ndarray, float]:
    scores = []
    sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            encoded = tokenizer(
                [row["text"] for row in batch for _ in options],
                options * len(batch), padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            scores.append(model(**encoded).logits.reshape(len(batch), len(options)).float().cpu().numpy())
    sync(device)
    return np.concatenate(scores), time.perf_counter() - started


def fitted_temperature(logits: np.ndarray, rows: list[dict], labels: list[str]) -> float:
    truth = np.array([labels.index(row["label"]) for row in rows])
    def loss(log_temperature: float) -> float:
        scaled = logits / np.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(truth)), truth]))
    return float(np.exp(minimize_scalar(loss, bounds=(-3, 3), method="bounded").x))


def evaluate_cross_encoder(args, question: dict, labels: list[str], rows: list[dict]) -> tuple[np.ndarray, float, dict]:
    from train_cross_encoder import device_choice

    device = device_choice(args.device)
    training = json.loads((args.checkpoint / "training.json").read_text())
    model_path = args.checkpoint / "model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    options = candidate_texts(question)
    max_length = training["max_length"]
    # The validation split fits only a scalar temperature; test labels stay sealed.
    validation = load_rows(args.data / "validation.jsonl")
    cross_encoder_logits(model, tokenizer, validation[:1], options, 1, max_length, device)
    val_logits, _ = cross_encoder_logits(model, tokenizer, validation, options,
                                         args.batch_size, max_length, device)
    temperature = fitted_temperature(val_logits, validation, labels)
    logits, seconds = cross_encoder_logits(model, tokenizer, rows, options,
                                            args.batch_size, max_length, device)
    return softmax(logits / temperature, axis=1), seconds, {
        "device": device, "model": training["model"], "parameters": training["parameters"],
        "temperature_fitted_on_validation": temperature,
        "train_seconds": training["train_seconds"],
    }


def evaluate_laya(args, questions: dict, labels: list[str], rows: list[dict]) -> tuple[np.ndarray, float, dict]:
    import laya

    from train_cross_encoder import device_choice

    device = device_choice(args.device)
    agent = laya.load(str(args.checkpoint), device=device)
    agent.predict(rows[0]["text"], questions)  # Warm-up outside the timed region.
    sync(device)
    started = time.perf_counter()
    answers = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        answers.extend(agent.predict_batch([row["text"] for row in batch], questions,
                                           batch_size=args.batch_size))
    sync(device)
    seconds = time.perf_counter() - started
    probabilities = []
    for result in answers:
        usage = result.get("usage", {})
        if usage.get("truncated") or usage.get("options"):
            raise ValueError(f"Laya input was truncated or options collapsed: {usage}")
        distribution = result["answers"]["category"]["probabilities"]
        probabilities.append([distribution[label] for label in labels])
    return np.array(probabilities, dtype=float), seconds, {
        "device": device,
        "parameters": sum(parameter.numel() for parameter in agent.model.parameters()),
        "laya_version": laya.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("cross-encoder", "laya"))
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, help="Pilot only; never report as the full test result")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    questions, labels = load_question(args.data)
    rows = load_rows(args.data / "test.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    if args.model == "cross-encoder":
        probabilities, seconds, details = evaluate_cross_encoder(
            args, questions["category"], labels, rows)
    else:
        probabilities, seconds, details = evaluate_laya(args, questions, labels, rows)
    report = metrics(rows, probabilities, labels)
    manifest = json.loads((args.data / "manifest.json").read_text())
    report.update({"model_type": args.model, "inference_seconds": seconds,
                   "milliseconds_per_request": 1000 * seconds / len(rows),
                   "batch_size": args.batch_size, "pilot_limit": args.limit,
                   "exploratory_run": args.limit is not None or args.data.name == "pilot",
                   "python_version": platform.python_version(),
                   "platform": platform.platform(),
                   "torch_version": torch.__version__,
                   "transformers_version": transformers.__version__,
                   "data_manifest": manifest,
                   **details})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "n", "accuracy", "macro_f1", "log_loss", "brier", "ece_10_bins",
        "milliseconds_per_request", "pilot_limit",
    )}, indent=2))


if __name__ == "__main__":
    main()
