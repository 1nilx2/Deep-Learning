"""Train one scalar score for each (request, candidate) pair with listwise CE."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import candidate_texts, load_question, load_rows, metrics

DEFAULT_MODEL_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"


def device_choice(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def collator(tokenizer, options: list[str], max_length: int):
    def collate(rows: list[dict]) -> tuple[dict, torch.Tensor]:
        states = [row["text"] for row in rows for _ in options]
        choices = options * len(rows)
        encoded = tokenizer(states, choices, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
        targets = torch.tensor([row["label_index"] for row in rows], dtype=torch.long)
        return encoded, targets
    return collate


@torch.no_grad()
def predict(model, loader, device: str, k: int) -> np.ndarray:
    model.eval()
    all_probs = []
    for encoded, _ in loader:
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded).logits.reshape(-1, k)
        all_probs.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(all_probs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "runs/cross_encoder")
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4, help="Requests per step; each expands to K pairs")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_choice(args.device)
    questions, labels = load_question(args.data)
    options = candidate_texts(questions["category"])
    train = load_rows(args.data / "train.jsonl")
    validation = load_rows(args.data / "validation.jsonl")
    for row in train + validation:
        row["label_index"] = labels.index(row["label"])
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.model_revision, num_labels=1).to(device)
    collate = collator(tokenizer, options, args.max_length)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_updates = args.epochs * math.ceil(len(train_loader) / args.grad_accum)
    updates = 0
    best = -1.0
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for step, (encoded, targets) in enumerate(train_loader, 1):
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.reshape(-1, len(labels))
            loss = F.cross_entropy(logits.float(), targets.to(device))
            group_start = ((step - 1) // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(train_loader) - group_start)
            (loss / group_size).backward()
            epoch_loss += loss.item()
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
        probabilities = predict(model, val_loader, device, len(labels))
        report = metrics(validation, probabilities, labels)
        history.append({"epoch": epoch, "train_loss": epoch_loss / len(train_loader),
                        "validation_accuracy": report["accuracy"],
                        "validation_macro_f1": report["macro_f1"]})
        print(json.dumps(history[-1]), flush=True)
        if report["macro_f1"] > best:
            best = report["macro_f1"]
            model.save_pretrained(args.out / "model")
            tokenizer.save_pretrained(args.out / "model")
    summary = {
        "model": args.model, "model_revision": args.model_revision,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": device, "epochs": args.epochs, "batch_size": args.batch_size,
        "grad_accum": args.grad_accum, "max_length": args.max_length,
        "learning_rate": args.lr, "seed": args.seed, "optimizer_updates": updates,
        "planned_optimizer_updates": total_updates,
        "train_seconds": time.perf_counter() - started, "best_validation_macro_f1": best,
        "history": history,
    }
    (args.out / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
