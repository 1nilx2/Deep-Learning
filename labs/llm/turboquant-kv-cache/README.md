# TurboQuant KV-cache experiment on Apple Silicon

This small lab asks a practical question: when a KV cache is compressed, how much does the **whole model allocation** change? It also checks prefill time and answer retrieval as context grows. The [results page](RESULTS.md) contains the charts and interpretation.

## Protocol

- **Model:** the official [Qwen2.5-1.5B-Instruct Q8_0 GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF).
- **Runtime:** [`TheTom/llama-cpp-turboquant`, `experiment/triattention-integration`](https://github.com/TheTom/llama-cpp-turboquant/tree/experiment/triattention-integration), pinned to `8ebbace3f33f63457f39479d8c2e4f8b1b4fc202`, with Metal on an Apple M1 Max (64 GB unified memory).
- **Data:** 12 short prompts at ~1.97k, ~4.03k, and ~6.54k tokens, plus four long prompts at 16,014 and 30,290 tokens. Each length places the target near 15% or 85% of synthetic records. The short set uses two generation seeds; the long set uses one. Each prompt asks for one project's audit code among routine records and codes for other projects. No company data is used.
- **Conditions:** The short set compares FP16 K/V; Q8 K + TurboQuant 4-bit V; FP16 K/V + TriAttention; and the combination. The TriAttention budget retains roughly 75% of prompt tokens. The long set compares FP16 K/V and Q8 K + TurboQuant 4-bit V only. All runs use the same model, Metal backend, temperature 0, sampler seed 42, and 12-token output cap.
- **Measurements:** backend-reported KV allocation and Metal model/compute/KV allocation; macOS `/usr/bin/time -l` peak process RSS; prompt evaluation time; and whether the first code in the response matches the target. Each case-condition pair was run once (48 short and eight long executions).

## Reproduce

Build the pinned experimental branch with Metal support and download the Q8_0 GGUF from the links above. Then provide absolute paths to the binary and model:

```bash
TURBOQUANT_BINARY=/absolute/path/to/llama-completion \
TURBOQUANT_MODEL=/absolute/path/to/qwen2.5-1.5b-instruct-q8_0.gguf \
python3 run_benchmark.py
```

The script regenerates the synthetic prompts and writes `quick-benchmark/results.csv`, case definitions, and per-run stdout/stderr logs. It targets macOS because it uses Metal and `/usr/bin/time -l`.

The longer comparison uses the same binary and model, with a two-minute per-run cap:

```bash
TURBOQUANT_BINARY=/absolute/path/to/llama-completion \
TURBOQUANT_MODEL=/absolute/path/to/qwen2.5-1.5b-instruct-q8_0.gguf \
TURBOQUANT_TIMEOUT_S=120 python3 run_long_extension.py
```

It writes `long-extension/results.csv`, cases, and per-run logs. The published measurements are [`results_long.csv`](results_long.csv) and [`cases_long.json`](cases_long.json). To redraw the three spotlight charts from both published CSVs, install Matplotlib and run:

```bash
SPOTLIGHT_OUTPUT_DIR=../../../assets/spotlight python3 plot_spotlight.py
```

The published CSVs and case definitions let readers inspect plotted values without downloading a model. The [projection and serving note](PROJECTION_AND_SERVING.md) separates the 1M KV storage calculation from actual model inference.
