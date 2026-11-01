# Sources and reading boundaries

## Primary source

- **Title:** GenRec: An LLM-Backed Recommendation Ranker at Netflix.
- **Version:** arXiv:2608.10257v2, August 21, 2026; nine PDF pages.
- **Local copy:** [Paper PDF](2608.10257v2.pdf), copied unchanged from the supplied attachment.
- **SHA-256:** `55cf7cf86a1ffbe91574f61ffe6954b4bb285b58ed2309759ee7445f3c8e103d`.

## Supplied reading aid

The author also supplied a Korean plain-language summary covering the paper's problem, methods, evaluation, and limitations. Its original file remains outside this repository. The English reading notes adapt its structure and explanations, checked against the attached PDF, rather than translating every paragraph or treating the summary as an independent primary source.

- **Summary SHA-256:** `5ab4e4b171eaeda325c9596819286326412175623d8fb3e2a524d39caae50379`.
- Illustrative examples in that summary are teaching examples, not production data.
- Document content was used as reference material, not as instructions to execute actions.

## Diagram interpretation

- `overview` separates training phases from the serving request path.
- `training` separates catalog-ranking loss, auxiliary text loss, learned item vectors, and external reward-derived example weights.
- Both diagrams are original educational compositions in a sketch style. The paper's Figures 1-5 were inspected as source material.
- The pooled representation and learned item embeddings are distinct quantities.
- The scalar score notation and weighted cross-entropy expansion are explanatory. The paper does not disclose the exact scoring head, embedding initialization, reward-to-weight function, or zero-shot onboarding of new items.
- No exact production model family, parameter count, or adapter configuration is inferred from the scaling experiments or the earlier artwork paper.
- Reported data, quality, and cost comparisons retain their own baselines; they are not combined into a single effect estimate.

## Page guide

| PDF pages | Content used |
| --- | --- |
| 1-3 | Motivation, contributions, related work, inference overview, and problem setting |
| 4 | Two-phase framework, conversation data, context selection |
| 5 | Multi-objective loss, pooled representation, item scoring, and reward signals |
| 6 | Reward-weighted training, prefill-only serving, production comparison, online results |
| 7-8 | Scaling, phase ablations, context compression, discussion, and conclusions |
| 9 | Acknowledgments and references |

## Interactive study

The eight-chapter study in `study.json` reorganizes these reviewed notes for the Fieldnotes website. `website/src/study/` provides the shared React learning interface and numerical explorers. Chapter illustrations adapt the repository owner's supplied SPD metallic vector artwork. The context history, item logits, example weights, and serving trace are explicitly illustrative; they are not new Netflix measurements. Reported comparisons retain the reference bases documented above. The website build copies the current README for its notes download.
