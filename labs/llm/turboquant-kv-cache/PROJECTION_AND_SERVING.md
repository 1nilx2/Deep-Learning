# KV cache at 1M tokens: arithmetic projection and serving practice

## What the local experiment can project

The local Qwen2.5-1.5B-Instruct Q8_0 runs allocated their KV cache for configured context capacities of 18,432 and 32,768 tokens. The allocation was exactly linear over these two capacities. The [long-context CSV](results_long.csv) contains the measurements:

| Format used in this experiment | Measured bytes per allocated token | At 1,000,000 allocated tokens |
| --- | ---: | ---: |
| FP16 keys + FP16 values | 28,672 bytes (28 KiB) | 26.70 GiB |
| Q8_0 keys + TurboQuant 4-bit values | 11,424 bytes (11.16 KiB) | 10.64 GiB |

The projected KV difference is **16.06 GiB per sequence** at 1M tokens. This is a storage calculation for this exact model architecture and cache format, not a measured 1M run. The model's [official configuration](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json) has a 32,768-token native position limit. The projection says nothing about whether the model would understand 1M tokens, or whether its attention kernels, temporary buffers, latency, and total memory would be practical there. Concurrent requests would add more live KV state unless blocks can be shared.

The experimental local configuration uses **Q8_0 for keys and TurboQuant 4-bit for values**. Its compression ratio and Mac latency should not be read as a reproduction of [Google's TurboQuant paper results](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/), which evaluate different configurations and hardware.

![Projected KV allocation at one million tokens](../../../assets/spotlight/turboquant-1m-projection.svg)

The model weights alone occupied 1,801 MiB in the local Metal memory breakdown. Weight + projected KV would therefore be at least about 28.46 GiB for FP16 or 12.40 GiB for this TurboQuant configuration, before compute buffers and other allocations. We do **not** extrapolate the observed total Metal allocation or prefill time linearly.

## How publicly documented long-context systems manage it

These techniques address different costs and are often combined:

1. **Fewer bytes for each retained token.** Model architecture such as grouped-query attention reduces the number of KV heads. KV quantization reduces precision; for example, [vLLM documents FP8 KV storage](https://docs.vllm.ai/en/v0.25.0/features/quantization/quantized_kvcache/), while [NVIDIA documents active KV quantization and newer compression paths](https://nvidia.github.io/TensorRT-LLM/features/kv-cache-compression.html). The trade-off is possible quality loss or extra conversion work, as seen in the local 30k test.
2. **Allocate and reuse blocks efficiently.** [vLLM's paged and prefix cache](https://docs.vllm.ai/en/stable/design/prefix_caching/) allocates KV in blocks and reuses a shared prefix across requests. This reduces fragmentation and repeat prefill; it does not remove the need to retain the KV for a unique active 1M-token request.
3. **Move less-used KV to another tier.** [vLLM's offloading connector](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/) moves completed blocks to host memory and loads hits back on demand. [TensorRT-LLM's KV connector](https://nvidia.github.io/TensorRT-LLM/features/kv-cache-connector.html) describes GPU/CPU/SSD tiers. This increases effective retained capacity but introduces transfer and scheduling costs.
4. **Reduce long-context processing work.** Qwen's [1M model and custom vLLM deployment](https://qwenlm.github.io/blog/qwen2.5-1m/) use sparse attention and chunked prefill. Chunking primarily cuts temporary activation memory; sparse attention primarily cuts attention work, and some sparse methods can also reduce retained KV. Qwen [lists at least 120 GB of GPU VRAM](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M) across devices for its 7B 1M model. This shows why a 1M service is a model and serving-system design problem, beyond KV quantization alone.

Public API features such as Google's [Gemini context caching](https://ai.google.dev/gemini-api/docs/long-context) expose reuse of repeated input to users. They do not disclose a provider's complete internal KV layout or memory policy, so those internals should not be inferred from the API feature.

The most useful distinction for our result is: **TurboQuant reduces bytes per retained token; paging, reuse, offloading, and sparse attention manage different parts of serving a long request.**
