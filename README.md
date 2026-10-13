# Machine Learning Notes & Labs

Study notes and hands-on notebooks across machine learning, NLP, computer vision, and LLMs. I use this repository to connect model architecture, training objectives, evaluation, and implementation. The indexes below describe the scope of each item.

## Start here

- [Laya vs CrossEncoder classification lab](labs/nlp/laya-vs-cross-encoder/README.md) — reproducible training on eight BANKING77 intents, with [held-out results](labs/nlp/laya-vs-cross-encoder/RESULTS.md).
- [Laya: typed decision model](notes/llm/laya.md) — a bilingual walkthrough from a customer request and typed questions to option probabilities, model architecture, and loss functions.
- [Training and optimization](notes/foundations/training-and-optimization.md) — working notes on gradients, backpropagation, cross-entropy, calibration, and training efficiency.
- [Segmentation metrics](notes/cv/segmentation-metrics.md) — notes on pixel accuracy, Dice, IoU, and mAP.

## Browse

| Section | Contents |
| --- | --- |
| [Notes](notes/README.md) | Concept explanations, model analyses, and research reading notes |
| [Labs](labs/README.md) | Learning notebooks and small code experiments, grouped by domain |
| [References](references/README.md) | PDFs retained as background reading |

The notebooks document exploration and learning. Their titles do not imply benchmark results or production readiness. Each lab is listed separately so its purpose is clear before opening it.

## Repository layout

```text
notes/       Written explanations and research notes
labs/        Notebooks and code used while learning or experimenting
assets/      Images used by the notes
references/  Background PDFs
```

## Experiment spotlight

Selected results from completed labs. Click a chart for the full protocol, metrics, and interpretation limits.

### Laya (Jev) vs Cross Encoder

[![Laya (Jev) vs Cross Encoder on BANKING77: Laya base 77.5% accuracy at 98.6 ms per request; Laya fine-tuned 90.6% at 98.8 ms; CrossEncoder fine-tuned with ModernBERT-large 93.4% at 208.0 ms.](assets/spotlight/laya-banking77-model-comparison.svg)](labs/nlp/laya-vs-cross-encoder/RESULTS.md)

Eight related payment intents, 320 held-out requests. The [lab](labs/nlp/laya-vs-cross-encoder/README.md) includes the training code and reproduction steps; the linked results page explains the remaining differences between the systems.

### TurboQuant: measured KV memory and a 1M-token projection

[![Measured KV cache allocation at 6.5k, 16k, and 30.3k input tokens, comparing FP16 with Q8 K and TurboQuant 4-bit V.](assets/spotlight/turboquant-kv-cache.svg)](labs/llm/turboquant-kv-cache/RESULTS.md)

[![Projected KV cache allocation at one million tokens: 26.7 GiB with FP16 versus 10.6 GiB with Q8 K and TurboQuant 4-bit V. This is a storage estimate, not a model run.](assets/spotlight/turboquant-1m-projection.svg)](labs/llm/turboquant-kv-cache/RESULTS.md)

[![Measured total Metal allocation at 6.5k, 16k, and 30.3k input tokens, broken into model weights, compute buffers, and KV cache.](assets/spotlight/turboquant-total-metal.svg)](labs/llm/turboquant-kv-cache/RESULTS.md)

On an M1 Max with Qwen2.5-1.5B, the 30.3k-token runs cut KV allocation from 896 to 357 MiB and total Metal allocation from 4,368 to 3,829 MiB. The 1M chart extends **KV storage only**; this model cannot run a 1M-token context. The [results](labs/llm/turboquant-kv-cache/RESULTS.md) include timing, retrieval outcomes, and limits.
