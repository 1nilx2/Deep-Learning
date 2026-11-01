# SPD 논문 정리

> **논문:** SPD: Single Pass Decoding for Generative Reranking  
> **저자:** Emil Laftchiev 외, Meta Platforms  
> **버전:** arXiv:2609.01807v2, 2026년 9월 4일, Under Review  
> **원문:** https://arxiv.org/abs/2609.01807v2

## 1. 핵심 질문과 아이디어

**이미 ranking을 잘하는 생성형 LLM이 있을 때, 순위를 토큰으로 하나씩 생성하는 과정을 없애면서 성능을 유지할 수 있을까?**

SPD는 LLM을 한 번 실행해 후보별 hidden state를 얻고, 이를 **후보–순위 점수 행렬**로 바꾼다. 추론에서는 Hungarian algorithm으로 점수 합이 가장 큰 일대일 배정을 찾아 최종 순위를 결정한다.

따라서 서비스 시에는 설명이나 순위를 autoregressive하게 생성하지 않는다. LLM은 사용자와 후보의 관계를 담은 representation을 만드는 backbone 역할을 한다.

이 논문은 전체 카탈로그에서 추천 대상을 찾는 retrieval보다, **상위 시스템이 가져온 후보 목록을 재정렬하는 reranking**을 다룬다.

## 2. 학습 모델을 세 단계로 보기

기호는 다음과 같다.

- \(N\): 입력 후보 수
- \(D\): LLM hidden dimension
- \(K\): 출력 순위 자리 수. 전체 후보의 순열을 만드는 경우 \(K=N\)
- \(d\): 추가 head에서 사용하는 표현 차원

이하에서는 이해를 위해 \(K=N\)인 전체 정렬을 기준으로 설명한다.

### 단계 1. Input: 사용자 context + 후보 설명들

사용자 context와 모든 후보의 설명을 하나의 prompt에 넣는다.

```text
User context: 사용자의 선호나 행동 등 요청 관련 정보
Candidates:
  A: 후보 A의 설명
  B: 후보 B의 설명
  C: 후보 C의 설명
```

입력의 사용자 정보 필드와 prompt의 정확한 형식은 논문에서 상세히 공개하지 않는다.

### 단계 2. Intermediate output: 후보별 벡터 → 점수 행렬 M

LLM의 single forward pass(prefill) 후, **각 후보 설명의 마지막 토큰 위치**에서 hidden state를 추출한다.

\[
H=[h_1;\ldots;h_N]\in\mathbb R^{N\times D}
\]

이 벡터들을 새로운 입력으로 사용해, 두 층의 작은 self-attention head를 통과시킨다. 이어 linear projection으로 각 후보의 순위별 점수를 계산한다.

\[
H\;(N\times D)
\xrightarrow{\text{self-attention head}}
\widetilde H\;(N\times d)
\xrightarrow{\text{linear projection}}
M\;(N\times N)
\]

**두 종류의 행렬을 구분해야 한다.**

- Attention 내부의 attention weight: 후보와 후보 사이의 관계
- 최종 출력 \(M\): **후보와 순위 자리 사이의 배치 적합도**

둘 다 \(N\times N\) 형태로 나타날 수 있지만 의미가 다르다. Head가 feature dimension을 유지하는 간략한 설명에서는 \(d=D\)로 생각할 수 있으나, 실제 head에서는 차원 변환이 들어갈 수 있다.

#### M의 의미

\[
M_{ij}=\text{후보 }i\text{를 }j\text{등에 배치하는 적합도 점수}
\]

예를 들어 A의 행이 다음과 같다면:

\[
M_A=[2,8,5]
\]

A를 1등에 놓는 점수는 2, 2등은 8, 3등은 5다. 이 행만 보면 2등 배치를 가장 선호한다.

이 값은 학습된 **score**이며, 그 자체가 확률이나 예상 클릭 수는 아니다. 일반적인 후보별 scalar scoring과 달리, 후보 하나에 대해 가능한 순위마다 점수를 하나씩 만든다.

### 단계 3. 학습용 output: 부드러운 배정 행렬 S

학습 시에는 \(M\)에 Sinkhorn 정규화를 적용해 \(S\)를 만든다.

\[
S=\operatorname{Sinkhorn}\bigl(\exp(M/\tau)\bigr)
\]

여기서 Sinkhorn은 양수 행렬의 행과 열을 번갈아 정규화하는 연산을 뜻한다. 구현에서는 수치 안정성을 위해 log-space 형태를 사용할 수 있다. \(\tau\)는 배정의 뾰족함을 조절하는 temperature다.

\(S\)는 각 행과 열의 합이 1에 가까워지는 **doubly stochastic matrix**다. 유한 번 반복하는 구현에서는 정규화가 근사적으로 성립한다.

예를 들어 정답 순위가 `[B, A, C]`라면:

\[
P=
\begin{pmatrix}
0&1&0\\
1&0&0\\
0&0&1
\end{pmatrix},
\qquad
S=
\begin{pmatrix}
0.15&0.75&0.10\\
0.80&0.15&0.05\\
0.05&0.10&0.85
\end{pmatrix}
\]

행은 A, B, C이고 열은 1등, 2등, 3등이다. 위 숫자는 설명용 예시다.

- \(P\): 정답 순열 행렬. 각 행이 one-hot이며, 각 열에도 1이 하나씩 있다.
- \(S\): 정답 순열 행렬의 부드러운 근사. 각 후보를 각 자리에 배정하는 정도를 나타낸다.

\(S\)는 순열 전체에 대한 명시적인 확률분포가 아니라, 부드러운 배정 행렬로 이해하는 것이 정확하다.

## 3. 정답 데이터와 loss: CE가 맞다

### 정답은 teacher가 만든 전체 순위

Autoregressive teacher가 학습용 후보 목록마다 순위를 생성한다. 이를 한 번 계산해 저장하고, student 학습의 고정된 정답으로 사용한다. 논문은 이 teacher-label 생성에 reasoning 없는 direct-output greedy decoding을 사용한다고 설명한다.

샘플은 개념적으로 다음 형태다.

\[
(\text{사용자 context},\;\text{후보 목록},\;\text{teacher 순열 }P)
\]

### Loss는 Sinkhorn cross-entropy

\[
\boxed{\mathcal L=-\sum_{i,j}P_{ij}\log S_{ij}}
\]

**이것은 CE와 비슷한 별도 loss가 아니라, cross-entropy 자체다.**

후보 \(i\)의 정답 순위를 \(r_i\)라고 하면:

\[
\mathcal L=-\sum_i\log S_{i,r_i}
\]

위 예시의 정답은 A→2등, B→1등, C→3등이므로:

\[
\mathcal L=-\log S_{A,2}-\log S_{B,1}-\log S_{C,3}
\]

**각 행의 one-hot 정답에 대한 CE를 합한 식**이다. 이 식은 논문의 한 후보 목록에 대한 목적함수이며, 배치 집계와 평균 방식은 별도의 구현 선택이다.

일반적인 분류와 다른 것은 CE 식보다 **그 앞의 정규화**다.

| 구분 | 일반 분류 | SPD |
|---|---|---|
| Score → 출력 | 행별 softmax | Sinkhorn |
| 정규화 제약 | 각 행의 합이 1 | 각 행과 각 열의 합이 1 |
| 정답 | 클래스 one-hot | 순위 one-hot들을 모은 순열 행렬 |
| Loss | CE | CE |

열 제약 때문에 각 행의 출력이 서로 연결된다. 각 후보를 서로 독립적으로 분류하는 것과는 다르다.

## 4. Backprop과 추론: Hungarian은 학습 경로에 없다

### 학습

```text
사용자 context + 후보들
  → LLM (LoRA로 조정)
  → 후보별 hidden states
  → 작은 self-attention head
  → linear projection
  → M
  → Sinkhorn
  → S
  → 정답 P와 CE
```

Sinkhorn은 미분 가능한 연산이므로 gradient가 head와 LoRA 파라미터까지 전달된다. 기본 구성에서는 backbone 전체 weight를 모두 업데이트하는 대신 LoRA를 학습한다.

### 추론

```text
사용자 context + 후보들
  → LLM
  → 후보별 hidden states
  → 작은 self-attention head
  → linear projection
  → M
  → Hungarian algorithm
  → 최종 순위
```

**Hungarian은 미분할 수 없지만, 추론에만 사용하므로 backprop이 필요 없다.** 자동으로 건너뛰는 것이 아니라, 학습과 추론의 마지막 연산을 다르게 설계한 것이다.

## 5. Hungarian algorithm은 무엇을 출력하나?

다음 점수표를 생각해 보자.

| 후보 | 1등 | 2등 | 3등 |
|---|---:|---:|---:|
| A | 9 | 8 | 1 |
| B | 10 | 2 | 0 |
| C | 1 | 3 | 9 |

후보마다 최고점을 독립적으로 고르면 A와 B가 모두 1등을 선택한다. Hungarian algorithm은 다음 제약 아래 점수 합을 최대화한다.

- 각 후보를 정확히 하나의 순위에 배정한다.
- 각 순위에 정확히 하나의 후보를 배정한다.

\[
\pi^*=\arg\max_{\pi\in\Pi_N}\sum_i M_{i,\pi(i)}
\]

여기서 \(\pi(i)\)는 후보 \(i\)에 배정한 순위다.

예시의 결과는:

```text
A → 2등
B → 1등
C → 3등
총점: 8 + 10 + 9 = 27
최종 목록: [B, A, C]
```

A는 1등에서 2등으로 내려가도 점수가 1만 줄지만, B는 8이 줄어든다. 전체 점수 합을 위해 A를 2등에 배치하는 것이다.

**최적이라는 말은 주어진 M에 대한 배정이 최적이라는 뜻이다.** 모델의 점수 자체가 부정확하면 실제 사용자에게 가장 좋은 순위라는 보장은 없다. 보장되는 것은 중복·누락 없는 배정의 유효성이다.

## 6. 왜 빠른가? 결과를 어떻게 읽어야 하나?

순위 토큰을 하나씩 생성하는 autoregressive decoding을 없애고, 한 번의 LLM prefill과 작은 head, 배정 solver로 끝낸다.

내부 데이터에서 보고한 결과는 다음과 같다.

| 모델 | 지연 시간 |
|---|---:|
| Teacher, reasoning 포함 | 1,807 ms |
| Teacher, reasoning 없음 | 88 ms |
| SPD | 28 ms |

- Reasoning teacher 대비 약 **64.5배** 빠르다.
- 순위만 생성하는 teacher 대비 약 **3.1배** 빠르다.
- 내부 데이터의 ranking 지표는 대체로 비슷한 수준이다.
- Amazon Beauty에서는 SPD 0.6B가 Qwen3-32B teacher 대비 약 44.9배 빠르지만, AUC 등 일부 지표는 낮다. 모든 데이터에서 완전히 손실이 없는 것은 아니다.

**64배 전체를 순수한 순위 decoding 방식의 효과로 해석하면 안 된다.** 그 비교에는 reasoning 출력 제거 효과도 들어 있다. 내부 실험의 teacher와 student backbone은 모두 0.6B 규모다.

또 **O(1)은 forward pass 횟수**에 대한 표현이다. 후보 수와 입력 길이가 커져도 전체 연산량이 일정하다는 뜻이 아니다. Hungarian의 최악 시간 복잡도는 O(N³)이며, 논문은 작은 후보 목록에서는 solver 비용이 매우 작다고 보고한다.

## 7. GenRec과 비교: 둘 다 LLM을 representation backbone으로 쓴다

| 구분 | GenRec | SPD |
|---|---|---|
| 주된 문제 | 전체 카탈로그 또는 후보 집합의 추천 ranking | 주어진 후보 목록의 reranking |
| LLM readout | 사용자 context를 요약한 벡터 h | 각 후보 위치의 벡터 h_i |
| 콘텐츠 표현 | 학습 가능한 콘텐츠별 e_i 사용 | 입력 후보 설명에서 h_i 계산 |
| 출력 점수 | 콘텐츠별 점수 | 후보 × 순위 점수 행렬 M |
| 추론의 최종 결정 | 점수를 이용한 ranking | Hungarian 일대일 배정 |
| 텍스트 autoregressive 생성 | 추천 추론에서 사용하지 않음 | 사용하지 않음 |

둘 다 LLM을 단순히 고정된 embedding 추출기로 사용하는 데 그치지 않고, ranking에 맞게 추가 학습한다.

### 새 후보에 대한 구조적 유연성

SPD는 새 후보의 설명을 입력하면 h_i를 계산할 수 있으므로, 후보마다 학습 가능한 전용 embedding을 마련해야 하는 의존성이 낮다.

하지만 다음은 구분해야 한다.

- **새 콘텐츠의 정체성:** 설명 기반으로 표현할 수 있으므로 구조적 장벽이 낮다.
- **새 콘텐츠의 추천 품질:** zero-shot으로 잘 정렬하는지는 별도 검증이 필요하다.
- **후보 수 변화:** 순위별 projection의 출력 크기 K와 입력 길이 한계가 있으므로 무제한으로 확장할 수 있다는 뜻은 아니다.

논문은 새로운 콘텐츠만 따로 분리한 평가를 통해 zero-shot item generalization을 명확히 입증하지 않는다.

## 8. 그럼 embedding 모델이나 cross-encoder로 만들면 안 되나?

**구조적으로 가능하다. 이 논문은 그것이 불가능하다고 보이지 않는다.**

SPD의 h_i는 후보 설명만 독립적으로 인코딩한 고정 벡터가 아니라, prompt context를 반영하는 표현이다. 다만 decoder는 causal attention이므로 앞쪽 후보가 뒤쪽 후보를 backbone 단계에서 보는 것은 아니다. 추가 self-attention head에서 모든 후보가 서로를 볼 수 있다.

그렇다고 contextual encoding이 decoder-only LLM만의 장점은 아니다.

- `(사용자, 후보 i)`를 함께 인코딩하는 일반적인 cross-encoder는 사용자–후보 관계를 반영한다.
- `(사용자, 후보 1, …, 후보 N)`을 함께 넣는 encoder는 후보 간 관계도 반영할 수 있다.
- 후자의 후보별 표현을 같은 attention head와 배정 loss에 연결하는 대안도 생각할 수 있다.

논문은 frozen backbone보다 LoRA로 조정한 backbone이 더 좋다는 실험을 제시한다. 그러나 이것은 **전용 embedding 모델이나 encoder보다 decoder-only LLM이 우월하다는 비교가 아니다.** 동일한 조건의 encoder 대체 실험은 제시하지 않는다.

따라서 이 논문은 **가장 좋은 backbone 종류를 규명하는 연구**보다, **이미 ranking을 잘하는 생성형 LLM을 한 번의 forward pass로 순위를 내는 모델로 전환하는 연구**로 읽는 것이 적절하다.

## 9. 의의와 읽을 때의 주의점

의의는 다음 요소의 결합에 있다.

1. LLM hidden state에서 후보–순위 점수 행렬을 만든다.
2. Teacher의 전체 순열을 Sinkhorn CE로 증류한다.
3. 추론에서는 Hungarian으로 유효한 순열을 보장한다.
4. 생성형 teacher와 비슷한 ranking 품질을 더 낮은 지연 시간으로 얻는다.

다만 다음 주장은 논문이 보여준 것보다 강하다.

- “Hungarian이 실제 추천 품질의 전역 최적해를 보장한다.” → M에 대한 최적 배정만 보장한다.
- “후보 수가 늘어도 비용은 O(1)이다.” → forward pass 횟수만 일정하다.
- “LLM이 cross-encoder보다 더 적합함을 입증했다.” → 해당 비교가 없다.
- “새 작품에 대한 zero-shot 일반화를 입증했다.” → 해당 평가가 명확히 제시되지 않는다.
- “기존 LLM에 설정만 바꾸면 적용된다.” → head 학습과 backbone adaptation이 필요하다.

추가로 Table 3에는 서술 불일치가 있다. Caption과 §6.1은 frozen backbone에서 teacher ranking 신호가 click label보다 나쁘다고 해석하지만, 표의 AUC는 0.5573 대 0.5187로 반대 방향이다. 따라서 이 표로 **풍부한 신호 자체가 frozen backbone에 해롭다**고 단정하면 안 된다. Teacher-ranking 조건에서 LoRA를 적용했을 때 성능이 개선된다는 비교는 표의 수치와 일치한다.

## 10. 기억할 한 줄

**SPD는 LLM으로 후보별 contextual representation을 만들고, 후보–순위 점수 행렬을 학습한 뒤, 추론에서 최대 합 일대일 배정으로 순위를 결정하는 모델이다. Loss는 CE이고, Sinkhorn은 학습용 미분 가능한 정규화, Hungarian은 추론용 최종 배정 알고리즘이다.**

---

### 원문 위치 안내

- §3.2: 순위의 최적 배정 문제 정식화
- §3.3: Sinkhorn과 CE loss
- §4.1: Backbone, self-attention head, assignment decoder
- §4.3–4.4: Offline teacher distillation과 LoRA
- §5.2 및 Table 2: 주요 성능·속도 결과
- §5.3–5.4: Backbone adaptation과 head ablation
- §5.5: Solver 비용

본 문서는 첨부된 v2 원문과 대화 내용을 바탕으로 작성했다. 수치 예시는 별도 표기가 없는 논문 결과와 구분했으며, 일반화와 encoder 대안에 대한 논의는 구조에 근거한 해석이다.
