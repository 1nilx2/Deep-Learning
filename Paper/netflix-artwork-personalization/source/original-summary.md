# Netflix Artwork Personalization via LLM Post-training: Supplied Summary

This is an English translation of the supplied Korean draft. It preserves the draft's explanations and assumptions for comparison. For corrections based on the paper, see [the reading notes](../README.md#2-clarifications-to-the-supplied-summary). The original Korean file remains at the location supplied by the author outside this repository.

The draft describes how Netflix uses LLM post-training to optimize personalized artwork (thumbnail) recommendations through LoRA-SFT and LoRA-DPO.

## 1. Overview and pipeline architecture

Traditional recommendation systems focus on which title to recommend. This work addresses a more granular personalization problem: **once a title has been selected, which visual aspects should its artwork emphasize to maximize the user's likelihood of watching or clicking?**

The draft presents a two-stage post-training pipeline using **LoRA (Low-Rank Adaptation)**, a parameter-efficient fine-tuning method that freezes the weights of an 8B open-source LLM, such as Llama 3.1 8B, and trains only adapter parameters.

```text
[Base LLM: frozen]
        ↓ + LoRA adapter
[Stage 1: LoRA-SFT]
Supervised learning of CoT reasoning and artwork prediction
        ↓ Copy / branch weights
[Stage 2: LoRA-DPO]
Reference: frozen LoRA-SFT model
Policy: trainable LoRA-DPO model
```

## 2. Stage 1: LoRA-SFT (Supervised Fine-Tuning)

This stage teaches the base language model to compare user preferences with visual descriptions and generate a recommendation through chain-of-thought (CoT) reasoning.

### 2.1 Input composition, $X$

For computational efficiency, the prompt uses abstracted text captions from a vision-language model (VLM), rather than image pixels.

1. **User information:** User metadata, viewing history, and genre or mood preferences.
2. **Target-title metadata:** Title, genres, detailed synopsis, and major themes.
3. **Candidate artwork captions:** Detailed descriptions of candidate thumbnails, such as `[Artwork 1]`, `[Artwork 2]`, and so on. These may describe solo or ensemble composition, character emotions and expressions, lighting and color, motion, or suspenseful framing.
4. **Task instructions:** Compare the user's preferences with the candidates, select the best artwork, and explain the decision.

### 2.2 Output structure, $Y$

The draft combines a CoT explanation with the answer rather than outputting only an index.

- **Reasoning:** Compare the user's past preferences with visual elements in each artwork and explain why a candidate should be selected.
- **Selected artwork:** Output the final index in a tagged format, such as `[Selected Artwork] Artwork 1`.

### 2.3 Obtaining reasoning targets

Production logs record behavior labels such as “the user clicked Artwork 1,” but do not record the user's internal reason for the choice.

The draft proposes **teacher-model distillation** using a capable reasoning LLM, such as a QwQ-32B model:

- **Teacher input:** User information, target metadata, candidate captions, and the label of the artwork actually clicked.
- **Teacher task:** Generate a retrospective rationalization of why the user selected that artwork.
- **Training target:** Combine the generated reasoning with the ground-truth index to form the LoRA-SFT target $Y$.

### 2.4 Loss function and backpropagation scope

Apply the standard causal language modeling **cross-entropy / next-token prediction loss**:

$$
\mathcal{L}_{\text{SFT}} = - \sum_{t=1}^{T} \log P_\theta(y_t \mid X, y_{<t})
$$

The draft describes the loss scope as follows:

- **Prompt $X$:** Excluded from loss through `label = -100` masking.
- **Target $Y$:** Includes the entire sequence of `[reasoning tokens] + [selected artwork tokens]`.

### 2.5 Why include both reasoning and selection in the loss?

1. **Reasoning distillation:** The draft argues that masking reasoning and training only on the index removes the incentive to generate useful intermediate reasoning, potentially weakening contextual generation.
2. **Autoregressive conditioning:** The final decision follows $P(\text{Decision} \mid \text{Input}, \text{Reasoning})$. The draft therefore argues that learning coherent preceding reasoning can support more stable decisions and generalization.

## 3. Stage 2: LoRA-DPO (Direct Preference Optimization)

### 3.1 Training objective

The draft describes the objective as follows:

> Align the policy to prefer the artwork the user saw and clicked ($w$) over an artwork that was also shown but skipped ($l$).

It argues that SFT, which uses positive examples to increase the probability of the correct answer, has difficulty separating subtle candidate preferences. It describes DPO as using exposed-but-ignored negative feedback to establish a clearer preference margin.

### 3.2 Data collection through existing exploration infrastructure

The draft assumes that Netflix's production infrastructure already uses contextual bandits and online A/B testing to expose different candidate artworks randomly or probabilistically. It therefore assumes that offline logs can directly supply clicked and exposed-but-unclicked artwork pairs under the same context.

### 3.3 DPO data, $(x, y_w, y_l)$

- **Prompt $x$:** The same structure as Stage 1: user history, target metadata, candidate captions, and instructions.
- **Winning response $y_w$ (chosen):** A response for artwork $w$, which the user watched or clicked: `[Reasoning for w] + [Selected Artwork: Artwork w]`.
- **Losing response $y_l$ (rejected):** A response for artwork $l$, which the draft describes as shown alongside the chosen artwork but skipped: `[Reasoning for l] + [Selected Artwork: Artwork l]`.

### 3.4 Obtaining reasoning for $y_l$

The draft describes two possible collection methods:

1. **SFT self-generation / rollouts, presented as the main method:** Give $x$ to the Stage 1 model, $\pi_{\text{ref}}$. If it incorrectly selects an ignored artwork $l$, retain the generated response as $y_l$. The draft argues that this targets the policy's actual error patterns and its on-policy distribution.
2. **Teacher-conditioned generation:** Ask a teacher to select a specified incorrect artwork $l$ and generate a plausible explanation, then use that response as $y_l$.

### 3.5 DPO loss

Optimize the LoRA adapters directly with a closed-form objective, without training a separate reward model:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

- **Reference model $\pi_{\text{ref}}$:** The frozen model obtained after LoRA-SFT in Stage 1.
- **Policy model $\pi_\theta$:** Initialized from the same weights, with continued updates to the LoRA adapters.
- **Intuition in the draft:** Increase the relative probability of clicked responses and reduce that of ignored responses compared with the SFT reference, shifting the policy toward observed user preferences.

## 4. Comparison in the supplied draft

| Aspect | Stage 1: LoRA-SFT | Stage 2: LoRA-DPO |
| --- | --- | --- |
| Main purpose | Learn viewing-history and caption understanding, plus CoT reasoning | Increase preference margins using user-response logs |
| Training unit | A single sequence $(x, y)$ | A contrastive pair $(x, y_w, y_l)$ |
| Negative logs | Not used; positive click logs only | Described as required exposed-but-ignored examples |
| Loss | Causal cross-entropy | DPO margin loss with implicit reward |
| Updated parameters | Trainable LoRA adapters | Trainable LoRA adapters; reference model frozen |
