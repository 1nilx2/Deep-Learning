# Laya — Typed Decision Model

Study note · 2026-10-12

## English

### In one sentence

Laya takes a state and typed questions, then returns a probability distribution over each question's options without generating text.

### Input and output: one situation, two questions

A customer says, **“I was charged twice. Please refund the second charge.”** This sentence is the state. We want to know two things:

1. **Is this a refund request?** → <code>true</code> or <code>false</code>.
2. **Which team should handle it?** → <code>billing</code>, <code>technical</code>, or <code>other</code>.

This is one customer request with **two different questions about the same state**.

#### What does a training case look like?

The following illustrates **one record** in an RLCD training JSONL file. <code>state</code> is the customer's message, <code>questions</code> defines the questions and their options, and <code>gold</code> gives the target probability distribution for each question. The numbers are illustrative targets, not predictions from a model run. [Fine-tuning data format](https://nandhakishorm.github.io/laya/finetune/)

~~~json
{
  "state": "I was charged twice. Please refund the second charge.",
  "questions": {
    "refund_requested": {
      "type": "noul",
      "instructions": "Is the customer asking for a refund?",
      "criteria": {"false": "no refund requested", "true": "refund requested"}
    },
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "payments or refunds",
        "technical": "product errors",
        "other": "anything else"
      }
    }
  },
  "gold": {
    "refund_requested": {"probabilities": {"false": 0.02, "true": 0.98}},
    "department": {"probabilities": {"billing": 0.90, "technical": 0.05, "other": 0.05}}
  }
}
~~~

A training record does not always contain a probability distribution. It may instead provide one answer per question under <code>expected</code>, which the training pipeline converts to a target distribution. This example uses <code>gold</code> to connect the data to the loss function below. [Training data processing](https://nandhakishorm.github.io/laya/reference/train/)

#### What input does Laya read?

Laya builds **one input row per question** from <code>state</code> and <code>questions</code>. This record therefore produces two input rows containing the same customer message. The <code>gold</code> distributions are training targets used to compare with each row's prediction.

~~~text
one training record
  ├─ state + refund question → input row 1 → P(false), P(true)
  └─ state + team question   → input row 2 → P(billing), P(technical), P(other)
     gold distributions are compared with the corresponding predictions
~~~

Here are the two serialized input rows in a readable form. <code>noul question:</code> and <code>choice question:</code> are ordinary text that expresses the question type. The tokenizer may split the words into smaller tokens. [Sequence builder](https://github.com/NandhaKishorM/laya/blob/main/laya/common.py#L2334-L2518)

~~~text
row 1 — refund requested
[CLS] noul question: Is the customer asking for a refund? [SEP]
[MASK] false: no refund requested
[MASK] true: refund requested [SEP]
I was charged twice. Please refund the second charge. [SEP]

row 2 — department
[CLS] choice question: Which team should handle this? [SEP]
[MASK] billing: payments or refunds
[MASK] technical: product errors
[MASK] other: anything else [SEP]
I was charged twice. Please refund the second charge. [SEP]
~~~

Processing both rows in one batch produces a separate prediction distribution for each question. For <code>noul</code>, the answer reports the probability of <code>true</code>; for <code>choice</code>, it reports the option probabilities and the highest-probability option. The <code>gold</code> values above are illustrative targets, not measured predictions. [Prediction interface](https://github.com/NandhaKishorM/laya#quickstart)

### Architecture: ModernBERT + Decision Head

This walkthrough follows the **English checkpoint**, whose encoder is ModernBERT-large. The multilingual checkpoint uses mmBERT-base, while the overall path from input to option probabilities is similar. [Checkpoint overview](https://huggingface.co/convaiinnovations/laya/blob/main/rl_agent_config.json)

The two questions pass through the model as shown below. ModernBERT makes a **contextual vector for every token**; the Decision Head reads the **option-marker vectors** and turns them into probabilities.

~~~mermaid
flowchart LR
    R1["row 1<br/>state + refund question (noul)"] --> T["batch: 2 input rows<br/>token IDs · 2×L"]
    R2["row 2<br/>state + team question (choice)"] --> T
    T --> B["ModernBERT<br/>contextual token vectors · 2×L×D"]
    B --> H["Decision Head<br/>type embedding + 2 Transformer layers<br/>[MASK] vectors → scorer → softmax<br/>2×L×D → 2×3×D → 2×3"]
    H --> O1["refund requested<br/>P(false), P(true)"]
    H --> O2["department<br/>P(billing), P(technical), P(other)"]
~~~

B is the batch size (the number of input rows processed together), L is the padded token length, D is the vector dimension, and K is the maximum number of options in the batch. Here, **B=2 and K=3**. The refund row has only two options, so its third, empty option slot is masked before probabilities are computed.

The Decision Head proceeds in five steps:

1. Add an embedding for each row's question type to every token vector in that row.
2. Pass all token vectors through two Transformer layers, keeping the B×L×D shape.
3. Gather the hidden vector at each option's <code>[MASK]</code> position. This marker is a **readout location for the option representation**, rather than a word prediction site.
4. Apply the same scorer (LayerNorm → Linear → GELU → Linear) to each vector, producing one logit per option.
5. Mask nonexistent option slots, apply the calibrated temperature and softmax, and read out the typed answer.

A separate act/escalate branch uses the <code>[CLS]</code> vector and summary features to estimate whether to act or escalate. It does not replace the option probabilities. A high raw probability is not automatically a calibrated confidence. [Model implementation](https://github.com/NandhaKishorM/laya/blob/main/laya/common.py#L3015-L3151), [Calibration guidance](https://nandhakishorm.github.io/laya/benchmarks/)

### Loss function

```math
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{RLCD}}
+
\mathcal{L}_{\mathrm{CE}}.
```

#### CE

```math
\mathcal{L}_{\mathrm{CE}}
=
-\frac{1}{B}\sum_{b=1}^{B}\sum_{i=1}^{K_b}
t_{b,i}\log p_{\theta,b,i}.
```

Here, $b$ indexes input rows within a batch, and $B$ is the **batch size**. Each question becomes one input row, so a batch containing both questions from the customer example has $B=2$. $K_b$ is the **number of options** in row $b$. Because each option has one <code>[MASK]</code> marker, it is also the number of <code>[MASK]</code> markers in that row.

$t_b$ is the **ground-truth target distribution** provided under <code>gold</code>, and $p_{\theta,b}$ is the model's predicted distribution. For a separate three-option example:

| Option | A | B | C |
| --- | ---: | ---: | ---: |
| Target $t_b$ | 0.1 | 0.7 | 0.2 |
| Model prediction $p_{\theta,b}$ | 0.2 | 0.5 | 0.3 |

**Meaning:** CE matches **these two probability distributions**. The target assigns option B a probability of 0.7, while the model assigns it 0.5. Substituting the numbers gives $-0.1\log 0.2-0.7\log 0.5-0.2\log 0.3$. Minimizing CE moves the model's probability allocation toward the target distribution.

#### RLCD

RLCD averages the following term across rows in a batch and samples for each row. The core of each sample's contribution is **advantage × log probability**; the minus sign appears because this is a loss to minimize.

```math
\mathcal{L}_{\mathrm{RLCD}}
=
-\frac{1}{BG}\sum_{b=1}^{B}\sum_{g=1}^{G}
A_b^{(g)}\log\pi_\theta\!\left(z_b^{(g)}\mid x_b\right).
```

**$A_b^{(g)}$ — advantage:** How much better the $g$-th candidate is than the other candidates for input row $b$. Its reward comes from comparison with the target distribution $t_b$. It is positive when the reward is above that row's sample average and negative when below. In the loss, it **weights** the candidate's log probability.

**$\log\pi_\theta(z_b^{(g)}\mid x_b)$ — log probability:** The value the current model assigns to candidate $z_b^{(g)}$ given input $x_b$. It plays the same role as $\log\pi_\theta(y\mid x)$ in a familiar RL expression, but the candidate here is an **entire noisy logit vector**, rather than one class. Because the vector is continuous, this is a log *probability density*. The predicted probability of class $i$ is $p_{\theta,b,i}$ in the CE expression above.

Thus, training raises the log density of candidates with $A_b^{(g)}>0$ and lowers it for candidates with $A_b^{(g)}<0$.

**Details:** $G$ is the number of candidates sampled for each row. Laya adds noise to the model's original logits to form $z_b^{(g)}$, then applies softmax to get a candidate probability distribution. A proper scoring rule compares that distribution with $t_b$. Subtracting the mean reward of the row's candidates and normalizing it yields $A_b^{(g)}$. The implementation uses $G=4$ and computes log density from the distance between the candidate logit vector and the current model logit vector. [Reward function](https://github.com/NandhaKishorM/laya/blob/main/laya/common.py#L3254-L3302), [Loss implementation](https://github.com/NandhaKishorM/laya/blob/main/laya/train.py#L3004-L3037)

## 한국어

### 한 문장으로

Laya는 하나의 상태(state)와 유형이 정해진 질문들을 받아, 텍스트를 생성하지 않고 질문마다 선택지의 확률분포를 반환하는 decision model이다.

### 입력과 출력: 하나의 상황, 두 가지 질문

고객이 **“I was charged twice. Please refund the second charge.”**라고 말했다. 이 문장이 하나의 상황, 즉 <code>state</code>다. 여기서 알고 싶은 정보는 두 가지다.

1. **환불 관련 요청인가?** → 답은 <code>true</code> 또는 <code>false</code>.
2. **어느 팀이 처리해야 하는가?** → 답은 <code>billing</code>, <code>technical</code>, <code>other</code> 중 하나.

사람에게는 하나의 고객 문의지만, 모델에는 **같은 상황에 대한 서로 다른 두 질문**이다.

#### 학습 데이터에서는 어떻게 보일까?

아래는 이 사례를 RLCD 학습용 JSONL의 **한 레코드**로 표현한 예시다. <code>state</code>는 고객 문장, <code>questions</code>는 질문과 선택지, <code>gold</code>는 각 질문에 대한 정답 확률분포다. 숫자는 설명을 위해 만든 예시이며, 모델의 예측 결과가 아니다. [Fine-tuning 데이터 형식](https://nandhakishorm.github.io/laya/finetune/)

~~~json
{
  "state": "I was charged twice. Please refund the second charge.",
  "questions": {
    "refund_requested": {
      "type": "noul",
      "instructions": "Is the customer asking for a refund?",
      "criteria": {"false": "no refund requested", "true": "refund requested"}
    },
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "payments or refunds",
        "technical": "product errors",
        "other": "anything else"
      }
    }
  },
  "gold": {
    "refund_requested": {"probabilities": {"false": 0.02, "true": 0.98}},
    "department": {"probabilities": {"billing": 0.90, "technical": 0.05, "other": 0.05}}
  }
}
~~~

정답 확률분포가 **항상** 데이터에 들어가는 것은 아니다. 단일 정답을 <code>expected</code>로 적는 형식도 있고, 학습 과정에서 이를 목표 분포로 바꾼다. 위 예시는 뒤의 손실 함수를 이해하기 쉽도록 <code>gold</code> 형식을 사용했다. [학습 데이터 처리](https://nandhakishorm.github.io/laya/reference/train/)

#### Laya가 실제로 읽는 입력은?

<code>gold</code>는 학습 때 예측과 비교할 **정답**이므로 모델 입력 문장에 넣지 않는다. Laya는 <code>state</code>와 <code>questions</code>로 **질문당 입력 row 하나**를 만든다. 따라서 이 레코드 하나에서, 같은 고객 문장을 포함한 입력 row 두 개가 나온다.

~~~text
학습 레코드 1개
  ├─ state + 환불 여부 질문 → 입력 row 1 → P(false), P(true)
  └─ state + 담당 팀 질문   → 입력 row 2 → P(billing), P(technical), P(other)
     gold 확률분포는 학습 때 각 row의 예측과 비교
~~~

입력 row를 읽기 쉽게 펼치면 다음과 같다. <code>noul question:</code>과 <code>choice question:</code>은 질문 유형을 나타내는 **일반 텍스트**다. 실제 토크나이저는 이 텍스트를 더 작은 토큰으로 나눌 수 있다. [입력 구성 코드](https://github.com/NandhaKishorM/laya/blob/main/laya/common.py#L2334-L2518)

~~~text
row 1 — 환불 여부
[CLS] noul question: Is the customer asking for a refund? [SEP]
[MASK] false: no refund requested
[MASK] true: refund requested [SEP]
I was charged twice. Please refund the second charge. [SEP]

row 2 — 담당 팀
[CLS] choice question: Which team should handle this? [SEP]
[MASK] billing: payments or refunds
[MASK] technical: product errors
[MASK] other: anything else [SEP]
I was charged twice. Please refund the second charge. [SEP]
~~~

두 개의 row를 한 배치로 처리하면 질문별 예측 분포가 각각 나온다. <code>noul</code>은 <code>true</code>의 확률로, <code>choice</code>는 선택지별 확률과 가장 높은 선택지로 답을 읽는다. 위의 <code>gold</code> 수치는 정답 예시일 뿐, 실제 모델의 예측값은 아니다. [예측 인터페이스](https://github.com/NandhaKishorM/laya#quickstart)

### 구조: ModernBERT + Decision Head

여기서는 **영어 체크포인트**를 기준으로 본다. 이 체크포인트의 인코더는 ModernBERT-large다. 다국어 체크포인트는 mmBERT-base를 사용하지만, 입력에서 선택지별 결정으로 이어지는 큰 흐름은 같다. [체크포인트 개요](https://huggingface.co/convaiinnovations/laya/blob/main/rl_agent_config.json)

앞의 두 질문이 모델을 통과하는 경로는 다음과 같다. ModernBERT는 **모든 토큰의 문맥 벡터**를 만들고, Decision Head는 그중 **선택지 marker의 벡터**를 읽어 확률로 바꾼다.

~~~mermaid
flowchart LR
    R1["row 1<br/>state + 환불 여부 (noul)"] --> T["배치: 입력 row 2개<br/>토큰 ID · 2×L"]
    R2["row 2<br/>state + 담당 팀 (choice)"] --> T
    T --> B["ModernBERT<br/>토큰별 문맥 벡터 · 2×L×D"]
    B --> H["Decision Head<br/>유형 임베딩 + Transformer 2층<br/>[MASK] 벡터 → scorer → softmax<br/>2×L×D → 2×3×D → 2×3"]
    H --> O1["환불 여부<br/>P(false), P(true)"]
    H --> O2["담당 팀<br/>P(billing), P(technical), P(other)"]
~~~

B는 batch size(한 번에 처리하는 입력 row 수), L은 패딩 후 토큰 길이, D는 벡터의 차원, K는 배치에서 가장 많은 선택지 수다. 이 예시에서는 **B=2, K=3**이다. 환불 여부 row에는 선택지가 두 개이므로, 세 번째 빈 자리는 확률 계산 전에 가린다.

그림의 Decision Head가 하는 일을 순서대로 적으면 다음과 같다.

1. 각 row의 질문 유형에 해당하는 임베딩을 모든 토큰 벡터에 더한다.
2. 토큰 벡터 전체를 Transformer layer 두 개에 통과시킨다. shape는 B×L×D로 유지된다.
3. 선택지마다 <code>[MASK]</code> 위치의 hidden vector를 추출한다. 이 marker는 **단어 예측 위치가 아니라 선택지 표현을 읽는 위치**다.
4. 같은 scorer(LayerNorm → Linear → GELU → Linear)를 각 벡터에 적용해 선택지별 logit을 하나씩 만든다.
5. 존재하지 않는 선택지 자리를 가리고, 보정된 temperature와 softmax를 적용해 질문 유형에 맞는 답을 읽는다.

별도의 act/escalate 경로는 <code>[CLS]</code> 벡터와 요약 특징을 사용해 행동할지 사람에게 넘길지의 확률을 추정한다. 이 경로가 선택지별 확률을 대신하지는 않는다. 높은 원시 확률을 곧바로 보정된 신뢰도로 해석하지 않는다. [모델 구현](https://github.com/NandhaKishorM/laya/blob/main/laya/common.py#L3015-L3151), [보정 안내](https://nandhakishorm.github.io/laya/benchmarks/)

### Loss function

```math
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{RLCD}}
+
\mathcal{L}_{\mathrm{CE}}.
```

#### CE

```math
\mathcal{L}_{\mathrm{CE}}
=
-\frac{1}{B}\sum_{b=1}^{B}\sum_{i=1}^{K_b}
t_{b,i}\log p_{\theta,b,i}.
```

여기서 $b$는 배치 안의 입력 row 번호이고, $B$는 **batch size**다. 이 모델에서는 질문마다 입력 row 하나를 만들므로, $B$는 한 배치에 들어간 질문의 수다. 앞의 고객 문의에서 두 질문을 함께 처리하면 $B=2$다. $K_b$는 row $b$의 **선택지 개수**이며, 선택지마다 <code>[MASK]</code>가 하나씩 있으므로 그 row의 <code>[MASK]</code> 개수와 같다.

$t_b$는 <code>gold</code>로 주어진 **정답(ground truth) 확률분포**, $p_{\theta,b}$는 모델이 예측한 확률분포다. 선택지가 A, B, C인 별도 예시를 보면:

| 선택지 | A | B | C |
| --- | ---: | ---: | ---: |
| 정답 분포 $t_b$ | 0.1 | 0.7 | 0.2 |
| 모델 예측 $p_{\theta,b}$ | 0.2 | 0.5 | 0.3 |

**해석:** CE는 **이 두 확률분포를 맞추는 손실**이다. 위 row에서 정답 분포는 B에 0.7을 주지만 모델은 0.5를 주었다. 식에 대입하면 $-0.1\log 0.2-0.7\log 0.5-0.2\log 0.3$이 된다. CE를 줄이는 방향으로 학습하면 모델의 확률 배분이 정답 분포에 가까워진다.

#### RLCD

RLCD 손실은 batch의 row와 각 row에서 만든 샘플에 대해 다음 항을 평균한다. 각 샘플의 핵심은 **advantage × log probability**이며, 손실을 최소화하는 식이므로 앞에 마이너스가 붙는다.

```math
\mathcal{L}_{\mathrm{RLCD}}
=
-\frac{1}{BG}\sum_{b=1}^{B}\sum_{g=1}^{G}
A_b^{(g)}\log\pi_\theta\!\left(z_b^{(g)}\mid x_b\right).
```

**$A_b^{(g)}$ — advantage:** $b$번째 입력 row에서 $g$번째 후보가 **같은 row의 다른 후보들보다 얼마나 좋은지** 나타낸다. 정답 분포 $t_b$와 비교해 받은 보상이 평균보다 높으면 양수, 낮으면 음수다. 따라서 수식에서 각 후보의 log probability에 곱하는 **가중치** 역할을 한다.

**$\log\pi_\theta(z_b^{(g)}\mid x_b)$ — log probability:** 입력 $x_b$가 주어졌을 때 현재 모델이 후보 $z_b^{(g)}$에 부여하는 값이다. 일반적인 RL의 $\log\pi_\theta(y\mid x)$와 같은 역할이지만, 여기서 후보는 **클래스 하나가 아니라 noisy logit 벡터 전체**다. 연속적인 벡터이므로 정확히는 log *probability density*다. 클래스 $i$의 예측 확률은 앞의 CE 식에 나온 $p_{\theta,b,i}$다.

식 전체로 보면 $A_b^{(g)}>0$인 후보의 log density는 높이고, $A_b^{(g)}<0$인 후보의 log density는 낮추는 방향으로 학습한다.

**세부 계산:** $G$는 row마다 만든 후보 수다. 모델의 원래 logit에 노이즈를 더해 $z_b^{(g)}$를 만들고, softmax를 적용해 후보 확률분포를 얻는다. 이를 $t_b$와 proper scoring rule로 비교해 보상을 계산한 뒤, 같은 row의 평균 보상을 빼고 정규화해 $A_b^{(g)}$를 만든다. 구현에서 $G=4$이며, log density는 후보 logit 벡터와 현재 모델 logit 벡터 사이의 거리로 계산한다. [보상 함수](https://github.com/NandhaKishorM/laya/blob/main/laya/common.py#L3254-L3302), [손실 구현](https://github.com/NandhaKishorM/laya/blob/main/laya/train.py#L3004-L3037)
