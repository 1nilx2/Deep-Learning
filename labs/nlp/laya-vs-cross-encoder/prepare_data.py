"""Make a fixed eight-intent benchmark from BANKING77's official split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

SOURCE_COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    f"{SOURCE_COMMIT}/banking_data"
)
QUESTION_ID = "category"
INSTRUCTIONS = "Which payment issue best describes this banking request?"

# Eight related BANKING77 intents, fixed before evaluating the official test set.
CRITERIA = {
    "card_payment_fee_charged": "a fee charged for a card payment",
    "card_payment_not_recognised": "a card payment the customer does not recognise",
    "card_payment_wrong_exchange_rate": "the exchange rate on a card payment seems wrong",
    "declined_card_payment": "a card payment was declined",
    "pending_card_payment": "a card payment is pending",
    "reverted_card_payment?": "a card payment was reversed or reverted",
    "transaction_charged_twice": "the same transaction was charged twice",
    "direct_debit_payment_not_recognised": "a direct debit payment the customer does not recognise",
}
QUESTION = {QUESTION_ID: {
    "type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA,
}}


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def stable_key(seed: int, label: str, text: str) -> str:
    return hashlib.sha256(f"{seed}\0{label}\0{text}".encode()).hexdigest()


def download_csv(name: str) -> tuple[list[dict], str]:
    with urllib.request.urlopen(f"{SOURCE_URL}/{name}", timeout=30) as response:
        raw = response.read()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    if not rows or set(rows[0]) != {"text", "category"}:
        raise ValueError(f"Unexpected columns in {name}")
    return rows, hashlib.sha256(raw).hexdigest()


def deduplicate(rows: list[dict], forbidden: set[str]) -> tuple[dict[str, list[dict]], int]:
    result: dict[str, list[dict]] = {label: [] for label in CRITERIA}
    seen = set(forbidden)
    skipped = 0
    for row in rows:
        label = row["category"]
        if label not in CRITERIA:
            continue
        text = row["text"].strip()
        key = normalized(text)
        if not key or key in seen:
            skipped += 1
            continue
        seen.add(key)
        result[label].append({
            "id": hashlib.sha256(f"{label}\0{key}".encode()).hexdigest()[:16],
            "text": text, "label": label,
        })
    return result, skipped


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-per-class", type=int, default=20)
    parser.add_argument("--train-per-class", type=int, default=50,
                        help="Fixed low-data training size per class (default: 50)")
    args = parser.parse_args()

    source_train, train_sha = download_csv("train.csv")
    source_test, test_sha = download_csv("test.csv")
    test_by_label, duplicate_test = deduplicate(source_test, set())
    test_texts = {normalized(item["text"]) for values in test_by_label.values() for item in values}
    train_by_label, duplicate_train = deduplicate(source_train, test_texts)
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for label in CRITERIA:
        candidates = sorted(train_by_label[label], key=lambda row: stable_key(
            args.seed, label, normalized(row["text"])))
        if len(candidates) <= args.validation_per_class:
            raise ValueError(f"Too few training rows for {label}")
        splits["validation"].extend(candidates[:args.validation_per_class])
        remaining = candidates[args.validation_per_class:]
        if len(remaining) < args.train_per_class:
            raise ValueError(f"Too few rows after validation for {label}")
        splits["train"].extend(remaining[:args.train_per_class])
        splits["test"].extend(test_by_label[label])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "questions.json").write_text(json.dumps(QUESTION, indent=2) + "\n")
    for split, records in splits.items():
        records.sort(key=lambda row: stable_key(args.seed, row["label"], row["text"]))
        write_jsonl(args.out / f"{split}.jsonl", records)
        write_jsonl(args.out / f"laya_{split}.jsonl", [{
            "state": row["text"], "questions": QUESTION,
            "expected": {QUESTION_ID: row["label"]},
        } for row in records])
    manifest = {
        "dataset": "BANKING77 (eight preselected payment intents)",
        "source": "https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data",
        "source_commit": SOURCE_COMMIT,
        "source_sha256": {"train.csv": train_sha, "test.csv": test_sha},
        "license": "CC-BY-4.0",
        "seed": args.seed,
        "validation_per_class": args.validation_per_class,
        "train_per_class": args.train_per_class,
        "exact_duplicates_removed": {"train": duplicate_train, "test": duplicate_test},
        "test_from_official_split": True,
        "total_rows": {split: len(rows) for split, rows in splits.items()},
        "class_counts": {
            split: dict(sorted(Counter(row["label"] for row in rows).items()))
            for split, rows in splits.items()
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
