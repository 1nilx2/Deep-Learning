# TurboQuant KV-cache: quick M1 Max results

This is an exploratory, single-run comparison on 12 synthetic retrieval prompts. The [protocol and reproduction steps](README.md), [CSV](results.csv), and [case definitions](cases.json) are available in this lab.

## KV cache allocation

![FP16 KV cache allocation versus Q8 K plus TurboQuant 4-bit V at three prompt lengths](../../../assets/spotlight/turboquant-kv-cache.svg)

The reserved KV cache is about **60% smaller** in the TurboQuant condition at each length: 84 → 33 MiB, 140 → 56 MiB, and 224 → 89 MiB.

## Whole model allocation

![Metal backend model, compute, and KV allocations with and without TurboQuant](../../../assets/spotlight/turboquant-total-metal.svg)

The model weights and compute buffers remain. The **total Metal backend allocation** falls by 50 MiB (2.3%), 84 MiB (3.8%), and 135 MiB (5.5%) across the same three lengths. The separately measured process peak RSS falls by approximately 50, 84, and 135 MiB, respectively. Backend allocation and RSS are different views of memory and must not be added together; neither is a system-wide physical-memory peak.

| Prompt length | FP16 KV | TurboQuant KV | FP16 total Metal | TurboQuant total Metal | FP16 peak RSS | TurboQuant peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ~1.97k tokens | 84 MiB | 33 MiB | 2,184 MiB | 2,134 MiB | 2,018 MiB | 1,968 MiB |
| ~4.03k tokens | 140 MiB | 56 MiB | 2,240 MiB | 2,156 MiB | 2,082 MiB | 1,998 MiB |
| ~6.54k tokens | 224 MiB | 89 MiB | 2,454 MiB | 2,319 MiB | 2,177 MiB | 2,041 MiB |

Values are medians over four prompts at each length. Different `-c` context capacities were allocated for the three length bands (3,072 / 5,120 / 8,192 tokens); the allocation graph therefore shows memory for those capacities, not a dynamically growing cache within a fixed-capacity process.

## Retrieval and timing

| Condition | Code retrieval | Median prefill at ~1.97k | ~4.03k | ~6.54k |
| --- | ---: | ---: | ---: | ---: |
| FP16 K/V | 12/12 | 1.01 s | 2.22 s | 4.12 s |
| Q8 K + TurboQuant 4-bit V | 12/12 | 1.17 s | 2.86 s | 5.75 s |
| FP16 K/V + TriAttention | 9/12 | 1.02 s | 2.27 s | 4.15 s |
| Q8 K + TurboQuant 4-bit V + TriAttention | 6/12 | 1.17 s | 2.89 s | 5.79 s |

The code metric checks the **first code detected in the generated response**. It accepts extra text after a correct code; a missing hyphen counts as wrong. TurboQuant alone kept this retrieval result but increased prefill time. With roughly 25% of prompt tokens evicted, TriAttention lost answers, especially at the longest length. That observation is specific to this budget and task, not a general assessment of TriAttention.

The 12 prompts reuse two target codes and represent one synthetic task. One run per condition is insufficient for confidence intervals or a general accuracy or latency claim. These measurements show the practical memory tradeoff for this model and input range; a broader evaluation needs more independent queries, repeats, and longer contexts.
