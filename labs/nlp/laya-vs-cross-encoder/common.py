"""Shared benchmark data and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def load_rows(path: Path) -> list[dict]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def load_question(data_dir: Path) -> tuple[dict, list[str]]:
    questions = json.loads((data_dir / "questions.json").read_text())
    labels = list(questions["category"]["criteria"])
    return questions, labels


def candidate_texts(question: dict) -> list[str]:
    return [
        f'{question["instructions"]} {label}: {description}'
        for label, description in question["criteria"].items()
    ]


def metrics(rows: list[dict], probabilities: np.ndarray, labels: list[str]) -> dict:
    if probabilities.shape != (len(rows), len(labels)):
        raise ValueError(f"Unexpected probability shape: {probabilities.shape}")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Invalid probabilities")
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    truth = np.array([labels.index(row["label"]) for row in rows])
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predicted == truth).astype(float)
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        in_bin = (confidence >= low) & (confidence <= high if high == 1 else confidence < high)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    one_hot = np.eye(len(labels))[truth]
    return {
        "n": len(rows),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=range(len(labels)), average="macro")),
        "log_loss": float(-np.log(np.clip(probabilities[np.arange(len(rows)), truth], 1e-8, 1)).mean()),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece_10_bins": float(ece),
        "per_class_f1": dict(zip(labels, map(float, f1_score(
            truth, predicted, labels=range(len(labels)), average=None, zero_division=0
        )))),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=range(len(labels))).tolist(),
        "label_order": labels,
    }
