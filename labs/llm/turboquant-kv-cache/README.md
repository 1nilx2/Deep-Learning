# TurboQuant KV-cache experiment on Apple Silicon

This small lab asks a practical question: when a KV cache is compressed, how much does the **whole model allocation** change? It also checks prefill time and answer retrieval as context grows. The [results page](RESULTS.md) contains the charts and interpretation.

## Protocol

- **Model:** the official [Qwen2.5-1.5B-Instruct Q8_0 GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF).
- **Runtime:** [`TheTom/llama-cpp-turboquant`, `experiment/triattention-integration`](https://github.com/TheTom/llama-cpp-turboquant/tree/experiment/triattention-integration), pinned to `8ebbace3f33f63457f39479d8c2e4f8b1b4fc202`, with Metal on an Apple M1 Max (64 GB unified memory).
- **Data:** 12 deterministic, fully synthetic retrieval prompts: three lengths (~1.97k, ~4.03k, ~6.54k tokens) × target near 15% or 85% of the records × two generation seeds. Each prompt asks for one project's audit code among routine records and codes for other projects. No company data is used.
- **Conditions:** FP16 K/V; Q8 K + TurboQuant 4-bit V; FP16 K/V + TriAttention; and the combination. The TriAttention budget retains roughly 75% of prompt tokens. All runs use the same model, prompt, Metal backend, temperature 0, sampler seed 42, and 12-token output cap.
- **Measurements:** backend-reported KV allocation and Metal model/compute/KV allocation; macOS `/usr/bin/time -l` peak process RSS; prompt evaluation time; and whether the first code in the response matches the target. Each case-condition pair was run once (48 executions total).

## Reproduce

Build the pinned experimental branch with Metal support and download the Q8_0 GGUF from the links above. Then provide absolute paths to the binary and model:

```bash
TURBOQUANT_BINARY=/absolute/path/to/llama-completion \
TURBOQUANT_MODEL=/absolute/path/to/qwen2.5-1.5b-instruct-q8_0.gguf \
python3 run_benchmark.py
```

The script regenerates the synthetic prompts and writes `quick-benchmark/results.csv`, case definitions, and per-run stdout/stderr logs. It targets macOS because it uses Metal and `/usr/bin/time -l`. To redraw the two spotlight charts from the published [`results.csv`](results.csv), install Matplotlib and run:

```bash
SPOTLIGHT_OUTPUT_DIR=../../../assets/spotlight python3 plot_spotlight.py
```

The published [CSV](results.csv) and [case definitions](cases.json) are included so the plots and counts can be checked without downloading a model.
