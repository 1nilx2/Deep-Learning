# SPD: Single Pass Decoding for Generative Reranking

[Interactive study source](index.html) · [Website preview and publishing](../../website/README.md) · [Original Korean study notes](source/original-summary.ko.md)

**Authors:** Emil Laftchiev et al., Meta Platforms  
**Version:** arXiv:2609.01807v2, September 4, 2026; under review  
**Paper:** https://arxiv.org/abs/2609.01807v2

## 1. The idea: one LLM pass, two paths

An autoregressive ranker generates a ranking token by token. SPD instead extracts contextual candidate representations from one LLM forward pass and learns an item–position score matrix. At inference, an assignment algorithm converts that matrix into a complete ranking.

The task is **reranking**: an upstream system has already retrieved the candidates.

Shared computation:

```text
User context + candidate descriptions
  → LLM, one prefill pass
  → candidate vectors H
  → additional self-attention + linear projection
  → item–position scores M
```

After M, training and inference diverge:

```text
Training:  M → Sinkhorn → S → cross-entropy against teacher target P
Inference: M → Hungarian algorithm → final ranking
```

“Single pass” refers to the LLM forward-pass count. It does not mean that total computation is independent of input length or candidate count.

## 2. H: Read the user. Read the candidates.

The input is one prompt containing user context and N candidate descriptions. SPD reads the hidden state at the final token of each candidate description:

\[
H=[h_1;\ldots;h_N]\in\mathbb R^{N\times D}.
\]

These vectors are contextual, rather than independently cached item embeddings. The backbone is a causal decoder, so earlier candidates cannot attend to later candidates during the backbone pass. Additional self-attention subsequently allows all candidate vectors to interact.

The paper does not disclose a detailed prompt template or exact user-information fields.

## 3. M: A score for every rank

Two additional self-attention layers compare candidate vectors. A linear projection produces M:

\[
H\;(N\times D)\rightarrow\widetilde H\;(N\times d)\rightarrow M\;(N\times K).
\]

For the full-ranking case discussed here, K=N.

\[
M_{ij}=\text{score for placing item }i\text{ at rank }j.
\]

Each row is an item; each column is a rank. For example, M(A,2)=8 means that assigning A to rank 2 earns a score of 8. This is a learned compatibility score, not a probability or expected reward.

The item–position score matrix M is different from the item–item attention matrices inside the attention layers.

## 4. S: Soft assignments for training

Hungarian assignment is discrete and non-differentiable. During training, SPD instead uses a differentiable Sinkhorn relaxation:

\[
S=\operatorname{Sinkhorn}(\exp(M/\tau)).
\]

Here Sinkhorn denotes alternating normalization of a positive matrix's rows and columns. The temperature controls sharpness. Finite iterations approximately enforce row and column sums of 1.

S is a doubly stochastic soft assignment matrix. It is a continuous relaxation of a permutation matrix, not an explicit probability distribution over complete rankings.

Unlike independent row-wise softmax, Sinkhorn also constrains column sums, coupling the assignments of different items.

## 5. Learn the teacher's ranking

The autoregressive teacher generates a ranking for each training slate once, offline, using direct-output greedy decoding without a reasoning trace. The resulting fixed permutation is the student's target.

A training example is conceptually:

```text
(user context, candidate descriptions, teacher permutation P)
```

For the target ranking [B, A, C], with rows A/B/C and columns ranks 1/2/3:

\[
P=\begin{pmatrix}0&1&0\\1&0&0\\0&0&1\end{pmatrix}.
\]

Each row is one-hot, and each column also contains exactly one 1. The loss is cross-entropy:

\[
\mathcal L=-\sum_{i,j}P_{ij}\log S_{ij}
=-\log S_{A,2}-\log S_{B,1}-\log S_{C,3}.
\]

This is ordinary cross-entropy over target assignments. The distinction is the preceding Sinkhorn normalization, rather than a new kind of CE. The paper's expression is for a slate; batch aggregation and normalization are separate implementation details.

Gradients pass through Sinkhorn, the scoring layers, and the backbone's LoRA parameters. Hungarian is not part of backpropagation.

The interactive page's slider blends S with P to illustrate decreasing CE. It does not simulate an actual optimizer step.

## 6. Decode the best assignment

At inference, SPD bypasses Sinkhorn and applies Hungarian/LAPJV assignment directly to M:

\[
\pi^*=\arg\max_{\pi\in\Pi_N}\sum_i M_{i,\pi(i)}.
\]

Here pi(i) is the rank assigned to item i. Every item receives one rank, and every rank receives one item.

| Item | Rank 1 | Rank 2 | Rank 3 |
|---|---:|---:|---:|
| A | 9 | 8 | 1 |
| B | 10 | 2 | 0 |
| C | 1 | 3 | 9 |

Independent row maxima would place both A and B first. The optimal joint assignment is B → A → C, with total score 10+8+9=27. Moving A from rank 1 to rank 2 costs only 1 point; moving B would cost 8.

The guarantee is **optimal assignment for the learned M**, not optimal real-world user satisfaction. Output validity is guaranteed by the one-to-one constraints.

The page's three-item demonstration enumerates all six permutations. The paper uses a Hungarian/LAPJV solver.

## 7. 28 ms: read the comparison

Internal-dataset results on NVIDIA A100:

| Model | Latency | AUC | Recall@1 | NDCG@1 |
|---|---:|---:|---:|---:|
| Teacher, with reasoning | 1,807 ms | 0.5911 | 0.1634 | 0.1776 |
| Teacher, ranking only | 88 ms | 0.5912 | 0.1635 | 0.1779 |
| SPD | 28 ms | 0.5907 | 0.1652 | 0.1791 |

SPD is approximately 64.5× faster than the reasoning teacher and 3.1× faster than the ranking-only teacher. The first comparison includes the benefit of removing reasoning output; the second more directly isolates replacing sequential ranking decoding.

Both internal teacher and student use a 0.6B backbone. These are measured configurations, not universal latency guarantees.

On Amazon Beauty, SPD 0.6B is approximately 44.9× faster than the Qwen3-32B teacher, but some quality metrics are lower. Near-lossless quality is not universal across all reported settings.

Hungarian has O(N³) worst-case complexity. At small slate sizes, the paper reports that the solver cost is tiny relative to LLM prefill. One forward pass does not imply constant total computational cost.

## 8. A ranking backbone, not a text generator

SPD uses the LLM to produce contextual representations and learn assignment scores. It does not generate ranking text at inference.

| Aspect | GenRec | SPD |
|---|---|---|
| Main setting | Catalog or candidate-set ranking | Reranking a supplied candidate slate |
| Readout | User/context vector h | Per-candidate vectors h_i |
| Item representation | Learned item-specific embeddings e_i | Candidate descriptions encoded in the prompt |
| Output scores | Scores per item | Item–position matrix M |
| Final ranking | Ranking from item scores | One-to-one assignment |

Both adapt the backbone for recommendation rather than merely using a frozen embedding extractor.

### New candidates

SPD can encode a new candidate from its description without requiring a dedicated learned embedding for that item. This lowers a structural barrier to new-item support, but does not establish strong zero-shot ranking quality on unseen items.

Candidate-count flexibility is also limited by context length, computation, and the position projection's output size K.

### Why not an embedding model or cross-encoder?

The paper does not present a matched comparison showing that a decoder-only backbone is better than an encoder.

A user–item cross-encoder can model the interaction within each pair. An encoder that jointly receives the user and all candidates can also model cross-candidate relationships and could feed a similar scoring/assignment architecture.

The frozen-versus-LoRA ablation supports adapting the backbone under the tested setup; it does not establish superiority over these encoder alternatives.

### A reporting inconsistency

Table 3 and the surrounding interpretation disagree: the caption and Section 6.1 claim that teacher-ranking targets hurt a frozen backbone relative to click labels, but the listed AUC values are 0.5573 versus 0.5187, respectively. That table does not support the claimed harmful effect. The listed improvement from LoRA under teacher-ranking supervision is consistent with the table.

## Takeaway

SPD combines contextual candidate representations, teacher-ranking distillation, differentiable assignment training, and exact one-to-one decoding. It shows an efficient alternative to autoregressive ranking in the reported experiments. Unseen-item generalization and matched encoder comparisons remain open questions.

## Source map

- Section 3.2: ranking as optimal assignment
- Section 3.3: Sinkhorn relaxation and cross-entropy
- Section 4.1: backbone, attention layers, projection, and decoder
- Sections 4.3–4.4: offline distillation and LoRA
- Section 5.2 / Table 2: quality and latency
- Sections 5.3–5.4: backbone and scoring-layer ablations
- Section 5.5: solver overhead

Numerical matrix examples are illustrative. Generalization and alternative-backbone discussions are architectural interpretations, not additional experimental results reported by the paper.
