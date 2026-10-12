# Banking intent classification: Laya vs CrossEncoder

This lab trains two decision systems on the same public banking requests. It is an executable companion to the [Laya study note](../../../notes/llm/laya.md).

## Task and data

Input: the `text` field of [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data). Target: one of eight related payment intents, selected and described in `prepare_data.py` before evaluating the test set. BANKING77 contains customer service queries and an official train/test split. The script downloads both CSV files at pinned commit `57ec275d8078af65b7731c2a98be812d844a6d6b`; the files are licensed CC BY 4.0. The benchmark is deliberately an **eight-intent subset**, so its numbers are not comparable to published 77-intent scores.

Both models see the same eight candidate labels, descriptions, and question wording in `prepare_data.py`:

- **Laya:** one `choice` question with all candidates; fine-tuned through the public `laya-train` CLI with its default RLCD objective.
- **CrossEncoder:** one scalar score for every `(request, candidate)` pair from `answerdotai/ModernBERT-base`, trained with cross-entropy across the eight scores.

These are end-to-end systems with different starting checkpoints and parameter counts. The comparison does not isolate the effect of the decision head alone. Candidate wording is fixed before any test evaluation. The dataset gives one correct label per item, so Laya receives `expected`; no teacher probability distribution is fabricated.

## Reproduce

Run from the repository root with Python 3.12. `HF_HOME` keeps downloaded public artifacts inside this ignored repository cache; model checkpoints, raw data, and runs are also ignored by Git.
The run used Laya English checkpoint revision `7b928d828b7b0e022f929d9bd2e44165aa270148` and ModernBERT-base revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`.

```bash
export UV_CACHE_DIR=.cache/uv
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r labs/nlp/laya-vs-cross-encoder/requirements.txt
export HF_HOME=.cache/huggingface
LAYA_BASE=$(.venv/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("convaiinnovations/laya", revision="7b928d828b7b0e022f929d9bd2e44165aa270148", allow_patterns=["model.safetensors", "rl_agent_config.json", "config.json", "encoder/config.json", "tokenizer/*"]))')

.venv/bin/python labs/nlp/laya-vs-cross-encoder/prepare_data.py
.venv/bin/laya-train \
  --data labs/nlp/laya-vs-cross-encoder/data/laya_train.jsonl \
  --eval-data labs/nlp/laya-vs-cross-encoder/data/laya_validation.jsonl \
  --base "$LAYA_BASE" --out labs/nlp/laya-vs-cross-encoder/runs/laya \
  --loss rlcd --shuffle-options --epochs 3 --micro-batch 4 --grad-accum 4 \
  --max-len 256 --head-max-len 192 --seed 42
.venv/bin/python labs/nlp/laya-vs-cross-encoder/train_cross_encoder.py \
  --out labs/nlp/laya-vs-cross-encoder/runs/cross_encoder

.venv/bin/python labs/nlp/laya-vs-cross-encoder/evaluate.py laya \
  --checkpoint labs/nlp/laya-vs-cross-encoder/runs/laya \
  --out labs/nlp/laya-vs-cross-encoder/runs/laya_test.json
.venv/bin/python labs/nlp/laya-vs-cross-encoder/evaluate.py laya \
  --checkpoint "$LAYA_BASE" \
  --out labs/nlp/laya-vs-cross-encoder/runs/laya_base_test.json
.venv/bin/python labs/nlp/laya-vs-cross-encoder/evaluate.py cross-encoder \
  --checkpoint labs/nlp/laya-vs-cross-encoder/runs/cross_encoder \
  --out labs/nlp/laya-vs-cross-encoder/runs/cross_encoder_test.json
.venv/bin/python labs/nlp/laya-vs-cross-encoder/summarize.py
```

This is a fixed low-data experiment: 50 allocated training and 20 validation requests per class, drawn from the official training split. All 40 official test requests per class are retained, for 400 allocated train / 160 validation / 320 test. Exact duplicate texts are removed from training if they occur in test, and split membership is recorded under `data/`. `laya-train` internally reserves 40 of the 400 training requests for calibration, leaving 360 gradient-training items. CrossEncoder uses all 400 for gradient training and fits one temperature on validation logits. The separate validation file is also used for Laya's training report; the test file is used only by `evaluate.py`.

Test reports contain accuracy, macro-F1, per-class F1, log loss, Brier score, 10-bin ECE, confusion matrix, and request-level inference time. Timing uses batches of four requests on the same device. The CrossEncoder scores all eight candidates for each request; Laya answers one question per request. Start-up and one warm-up request are excluded from timing.

The untouched Laya English checkpoint is included to show what fine-tuning changed.

The summary script checks that all three reports use the same manifest, full test set, device, and batch size before it writes [RESULTS.md](RESULTS.md).

For a fast training check, use `--train-per-class 5 --out .../data/pilot` with `prepare_data.py` and pass `--data .../data/pilot` to training. The pilot training items are a subset of the 400-row run. Do not use the official test split to tune the pipeline. The final benchmark evaluates the official test split once.

## Interpretation limits

- Eight related intents make a focused diagnostic task, but omit 69 BANKING77 intents. A deployed classifier would need an out-of-scope option or a full 77-intent evaluation.
- The two starting checkpoints have different histories and sizes. Model parameters and inference time are reported alongside quality metrics.
- No check was made for BANKING77 examples in the pretraining of either base checkpoint, so this is not a decontaminated benchmark.
- Laya's native training calibration and the CrossEncoder's validation temperature use different calibration protocols. Accuracy and macro-F1 remain the primary comparison; probability metrics should be interpreted with that difference in mind.
- The public dataset, public model packages, and code in this lab are independent of any employer repository or private customer data.

## Sources

- [Laya public fine-tuning guide](https://nandhakishorm.github.io/laya/finetune/)
- [Sentence Transformers CrossEncoder overview](https://www.sbert.net/docs/cross_encoder/training_overview.html)
- [BANKING77 source and citation](https://github.com/PolyAI-LDN/task-specific-datasets)
- [Casanueva et al., Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807)
