# EmbeddingGemma 2 기반 멀티모달 표현 학습 및 크로스모달 갭(Cross-Modal Gap) 분석 실험 연구 디자인

## 1. 개요 (Executive Summary)
본 연구 디자인 문서에서는 **EmbeddingGemma 2**의 네이티브 멀티모달 공통 임베딩 공간(Single Vector Space)을 활용하여, 아동 언어·발달 도메인에서의 모달리티별 표현력(Representation), 강건성(Robustness), 그리고 크로스모달 갭(Cross-Modal Gap)의 유의미성을 검증하기 위한 4가지 핵심 실험을 정의합니다.

* **핵심 질문**: "동일한 상황/발화에 대해 표현 도메인(Text vs. Audio)이 다를 때, 임베딩 모델은 이를 얼마나 일관되게 표현하며, 그 간극(Gap)은 발달 특성을 진단하는 지표가 될 수 있는가?"
* **백본 모델**: Google EmbeddingGemma 2 (740M Parameter, Modality Encoders + Shared Transformer Backbone)
* **주요 데이터셋**: TalkBank / CHILDES (Paired Utterance Transcripts & Audio Files)

---

## 2. 데이터셋 및 파이프라인 전처리 구조

### 2.1 Paired Data Specification
* **Target Data**: TalkBank (CHILDES) 아동-보호자 대화 세션
* **Instance Structure**: $i$-번째 발화에 대한 페어 $(T_i, A_i, Y_i)$
  * $T_i$: 전사 텍스트 (Transcript / Ideal linguistic intent)
  * $A_i$: 발화 구간 원본 오디오 클립 (Audio Signal / Acoustic realization)
  * $Y_i$: 아동 메타데이터 (연령 $Age_i$, 문법적 복잡도 $MLU_i$, 발달 상태 $Diagnosis_i$)

### 2.2 Embedding Extraction Pipeline
```
[Transcript T_i] -> Text Encoder ----> [Shared Backbone (Frozen)] -> v_text (768d)
[Audio Signal A_i] -> Audio Encoder -> [Shared Backbone (Frozen)] -> v_audio (768d)
```
* **Precision Note**: Numerical Precision 충돌 및 `NaN` 방지를 위해 `bfloat16` 정밀도 고정.

---

## 3. 핵심 실험 세부 디자인 (Experimental Designs)

---

### [Exp 1] Alignment & Modality Bias Evaluation
**"서로 다른 모달리티로 표현된 동일 상황은 공통 벡터 공간에서 얼마나 가까우며, 내용 중심으로 클러스터링되는가?"**

#### 1. 가설 (Hypothesis)
* 잘 정렬된 멀티모달 공간이라면 동일 발화 세션의 $(v_{text}, v_{audio})$는 모달리티 장벽을 넘어 높은 코사인 유사도를 가져야 하며, UMAP 시각화 시 모달리티가 아닌 **상황/주제(Topic)** 단위로 클러스터링된다.

#### 2. 실험 절차
1. N개의 동일 상황 페어 $(T_i, A_i)$에 대해 $v_{text, i}$ 및 $v_{audio, i}$ 추출.
2. 코사인 유사도 $Sim(v_{text, i}, v_{audio, i})$ 분포 산출.
3. 전체 벡터 집합에 대한 UMAP/t-SNE 차원 축소 수행.
4. HDBSCAN 클러스터링 적용 후 **실루엣 계수(Silhouette Coefficient)** 측정:
   * **Topic-based Silhouette**: 발화 주제/맥락별 응집도
   * **Modality-based Silhouette**: 텍스트 vs 오디오 형태별 분리도 (낮을수록 Modality Bias가 적고 잘 정렬됨)

---

### [Exp 2] Stress Test: Degradation & Robustness to Information Loss
**"인풋에 노이즈나 누락(Masking/Blurring)이 발생했을 때 어느 모달리티 표현이 정보 손실에 더 강건한가?"**

#### 1. 가설 (Hypothesis)
* 오디오 신호는 시간적/주파수적 중복성(Redundancy)을 내포하므로, 텍스트의 키워드 누락(Masking)보다 오디오의 부분적 무음/노이즈 오염 시 의미론적 임베딩 거리의 변동성이 더 적을 것이다.

#### 2. 열화(Degradation) 조건 정의
* **Text Degradation ($T'$)**: 핵심 어휘 Random Masking (10%, 30%, 50%), Stopword 누락, Typos 추가.
* **Audio Degradation ($A'$)**: Gaussian Noise 추가, 구간 Masking/Silence (10%, 30%, 50%), Bandpass Filtering.

#### 3. 평가 지표
* **Robustness Score**: 원본 벡터 $v$와 열화 벡터 $v'$ 간의 Cosine Distance $D(v, v') = 1 - \frac{v \cdot v'}{\|v\| \|v'\|}$
* **Retrieval Retention Rate**: 열화된 쿼리 $T'$ 또는 $A'$로 정답 Pair를 찾아내는 Top-1, Top-5 MRR(Mean Reciprocal Rank) 측정.

---

### [Exp 3] Linear Probing: Information Density by Modality
**"임베딩을 동결하고 동일한 Classification/Regression Head를 올렸을 때, 각 모달리티는 어떤 정보(문법 structure vs. 비언어적 징후)를 더 풍부하게 담고 있는가?"**

#### 1. 가설 (Hypothesis)
* 문법적 복잡도($MLU$) 추정에는 $v_{text}$가 우수하지만, 아동의 연령($Age$) 및 발달 지연($Diagnosis$) 분류에는 아동의 미세 발음/어조/망설임 정보가 반영된 $v_{audio}$ 헤더가 더 높은 성능을 보일 것이다.

#### 2. 모델 구조
* **Backbone**: EmbeddingGemma 2 (가중치 완전 동결 / Frozen)
* **Downstream Head**: 2-Layer MLP (ReLU, Dropout=0.2) - 텍스트/오디오 동일 가중치 구조 적용

#### 3. 태스크 설정
* **Task A (Regression)**: Target = 아동의 $Age$ (Months) 또는 $MLU$ score $\rightarrow$ Metric: $R^2$, MAE
* **Task B (Classification)**: Target = 정상 발달 vs 발달 지연 위험군 $\rightarrow$ Metric: F1-Score, ROC-AUC

---

### [Exp 4] Cross-Modal Gap as a Healthcare Diagnostic Metric
**"의도된 의미(Transcript)와 실제로 발현된 표현(Audio) 사이의 거리가 아동의 발달 지연을 스크리닝하는 이상 지표(Anomaly Score)가 될 수 있는가?"**

#### 1. 가설 (Hypothesis)
* 정상 발달 아동은 의도 언어 $T_i$와 음성 표현 $A_i$가 일치하여 $Gap(T_i, A_i)$가 작지만, 발음 왜곡·말더듬·발달 지연 아동은 음성 신호의 불안정성으로 인해 벡터 공간 상의 $Gap$이 유의미하게 벌어질 것이다.

#### 2. Cross-Modal Gap Metric 수식
$$Gap(T_i, A_i) = 1 - \frac{v_{text, i} \cdot v_{audio, i}}{\|v_{text, i}\| \|v_{audio, i}\|}$$

#### 3. 검증 방법론 (Zero-shot Anomaly Detection)
1. 별도의 Downstream 모델 학습 없이, 대상 아동 발화들의 평균 $Gap$ 점수 계산.
2. 정상 발달 군집 vs 발달 지연 군집 간 $Gap$ 점수 분포 비교 (Mann-Whitney U Test).
3. $Gap$ 점수 단독을 분류 분계값(Threshold)으로 활용했을 때의 발달 지연 감지 ROC-AUC 측정.

---

## 4. 종합 평가 매트릭스 (Evaluation Matrix Summary)

| 실험 항목 | Input Modality | Target Metric / Evaluation Method | 핵심 검증 목표 |
| :--- | :--- | :--- | :--- |
| **Exp 1: Alignment** | $T_i$ vs $A_i$ | Silhouette Coefficient, Cosine Sim | Modality Bias 극복 수준 검증 |
| **Exp 2: Degradation** | $T'_i$ vs $A'_i$ | MRR Retention Rate, Cosine Distance | 노이즈/누락에 대한 강건성 비교 |
| **Exp 3: Linear Probe** | $v_{text}$ vs $v_{audio}$ | $R^2$ (Regression), F1 / AUC (Classification) | 표현 공간 내 유의 정보 결(Domain) 분석 |
| **Exp 4: Cross Gap** | Pair Distance | ROC-AUC, Mann-Whitney U Test $p$-value | 벡터 간격 기반 제로샷 발달 진단 지표 검증 |

---

## 5. 결론 및 기대 효과
본 연구 실험 기획을 통해, 경량 로컬 멀티모달 모델인 **EmbeddingGemma 2**의 단일 벡터 공간 성능을 체계적으로 검증할 수 있습니다. 특히 **Exp 4 (Cross-Modal Gap)**는 모델 학습 없이 오직 임베딩 기하학 특성만을 활용하여 프라이버시가 중요한 로컬 헬스케어 엣지 디바이스에 즉시 적용 가능한 실용적 지표를 제공합니다.

---

## 6. 구현 및 설계 보완 기록 (2026-10-28)

이 문서는 실험 구현과 함께 갱신하는 작업 문서입니다. 위 1~5절은 최초 제안이며, 아래 항목은 구현·해석 시 우선 적용하는 보완 사항입니다. 구현 및 실행 방법은 [README.md](README.md)를 참조합니다.

### 6.1 현재 구현 상태

- Exp 1의 입력 검증, 동결 임베딩 추출, 코사인 유사도, 양방향 Recall@1/5·MRR, 원래 임베딩 공간에서의 실루엣 평가를 구현했습니다.
- 합성 벡터로 평가 로직 테스트 3개가 통과했습니다. 실제 EmbeddingGemma 2 추론 및 CHILDES 데이터 실험은 아직 수행하지 않았습니다.
- CHAT 파싱, 시간 정렬, 발화 클립 생성, UMAP/t-SNE 및 HDBSCAN은 아직 구현하지 않았습니다.
- Exp 2~4는 실제 데이터의 정렬 품질과 라벨 가용성을 확인한 뒤 구현합니다.

### 6.2 모델 및 데이터 조건

- 모델 ID는 `google/embeddinggemma-2`입니다. 공식 가이드는 텍스트·오디오의 공통 768차원 공간을 지원한다고 설명합니다.
- 정밀도는 BF16 지원 CUDA에서 `bfloat16`, 그 외에는 `float32`를 기본으로 합니다. `float16`은 사용하지 않습니다. BF16 고정은 필수 조건이 아닙니다.
- CHILDES의 모든 코퍼스에 발화 시간 정렬, 오디오, 연령, 진단 라벨이 갖춰져 있다고 가정하지 않습니다. 코퍼스별 가용성·접근 조건을 확인합니다.
- 전사는 관찰된 발화의 기록이며 아동의 의도나 이상적인 언어 내용을 직접 나타내는 정답으로 취급하지 않습니다.

### 6.3 실험별 보완

- **Exp 1:** 올바른 페어와 잘못된 페어를 비교하고 텍스트→오디오 및 오디오→텍스트 검색을 모두 평가합니다. 실루엣은 차원 축소 전 공간에서 계산합니다. 주제별 실루엣에는 별도 주제 라벨이 필요합니다. UMAP은 탐색적 시각화이며 정렬의 통계적 증거로 단독 사용하지 않습니다.
- **Exp 2:** 동일 삭제 비율을 동일 정보 손실량으로 해석하지 않습니다. 오디오 노이즈는 SNR로 정의하고, 열화 강도별 성능 곡선을 비교합니다. Recall@1/5와 MRR은 별도 지표로 보고합니다.
- **Exp 3:** 2층 MLP는 비선형 probe입니다. 선형 회귀·로지스틱 회귀를 기본 비교로 두고 MLP를 추가합니다. 학습·검증·평가는 아동 단위로 분리하며 같은 아동의 여러 세션도 같은 분할에 배치합니다. MLU의 형태소/단어 기준 및 아동·세션별 집계 단위를 명시하고 단일 발화 길이와 구분합니다.
- **Exp 4:** Gap의 발달군 차이는 검증 대상 가설입니다. 녹음 품질, 연령, 언어, 발화 길이, 전사·시간 정렬 오류 및 코퍼스 차이를 검토합니다. 발화들의 독립성을 가정하지 않고 아동 단위 집계·재표집으로 신뢰구간을 산출합니다. 임계값을 선택할 때는 평가 아동과 분리된 데이터에서 선택합니다. Gap 계산에 학습이 없더라도 임계값 조정은 별도의 보정 절차입니다.

### 6.4 해석 범위 및 다음 단계

최초 결론의 “즉시 적용 가능한” 진단 지표라는 표현은 현재 근거로 뒷받침되지 않습니다. 본 프로젝트의 목적은 연구용 후보 지표의 타당성을 탐색하는 것이며, 실제 스크리닝 적용 가능성은 독립적인 검증 이후 판단합니다.

다음 단계는 실제 코퍼스를 선정하고 여러 아동의 소규모 발화 페어를 준비하여 Exp 1을 실행하는 것입니다. 실험 결과가 생기면 데이터 출처, 제외 기준, 모델 revision, 실행 환경 및 한계를 이 문서와 함께 갱신합니다.

### 참고

- [Google 공식 멀티모달 구현 가이드](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)
- [CHILDES 데이터 접근 안내](https://talkbank.org/childes/access.html)


### 6.5 데이터 조사 진행

2026-10-28: CHILDES-Aligned를 첫 파일럿 후보, Gleason을 대안으로 확인했습니다. 아직 데이터를 다운로드하지 않았고 미디어 페이지는 Login을 표시합니다. 세부 접근 상태 및 필터링 편향은 [DATA_SELECTION.md](DATA_SELECTION.md)에 기록했습니다.


### 6.6 실행 확인용 데이터 준비

2026-10-28: Gleason Mother/Katie의 아동 발화 23쌍을 준비했습니다. 원본에 명시된 반복 손상 구간 이후 자료 및 복잡한 주석·중첩·중복 등을 제외했습니다. 한 아동의 실행 확인용이며 여러 아동의 연구용 파일럿은 아닙니다. 수동 청취와 실제 추론은 아직 완료하지 않았습니다. 자세한 기준과 재실행 방법은 README를 참조합니다.


### 6.7 실제 추론 및 Exp 1 실행 확인 완료

2026-10-28: 고정 revision의 EmbeddingGemma 2로 3쌍 실행 확인 후 Katie 23쌍 전체 CPU/FP32 추론과 평가를 완료했습니다. 정답 페어 평균 유사도 0.7664, 다른 페어 0.7274, 양방향 Recall@1은 텍스트→오디오 56.52%, 오디오→텍스트 26.09%입니다. 단일 아동의 청취 검증 전 자료에 대한 기술 통계이며 발달 관련 결론이 아닙니다. 실제 실행 환경·한계·재실행 방법은 [RESULTS.md](RESULTS.md)에 기록했습니다.


### 6.8 긴 대화 정렬 실험의 엄격한 페어 조건

사용자 요구에 따라 전체 화자를 포함하는 전사·오디오의 내용 및 시작·끝 일치를 필수 조건으로 합니다. 타임스탬프 기준 12개 후보를 준비했으나 청취 검증은 완료하지 못했습니다. 기존 Katie 자료의 손상 경고 이후는 제외했습니다. 검증되지 않은 긴 대화는 추론·평가 단계에서 차단합니다. 긴 대화의 Exp 1 결과는 아직 산출하지 않았습니다.


### 6.9 긴 대화 Exp 1 탐색 결과

사용자가 첫 3쌍을 검토했고 나머지 9쌍은 일치한다고 가정하여 실행하도록 요청했습니다. 전체 12쌍 실제 추론에서 양방향 Top-1 12/12, 평균 정답 유사도 0.8597, 다른 페어 0.7473(차이 0.1124)을 얻었습니다. 길이 외에도 화자 구성·후보 수가 달라 기존 단일 발화와의 차이를 길이 효과로 단정하지 않습니다. 가정한 9쌍은 검증 완료로 취급하지 않습니다. 자세한 결과는 RESULTS.md에 있습니다.


## 확장 Exp 1 결과 (2026-10-28)

6명·8세션·221쌍의 실제 추론과 확장 평가를 완료했습니다. 전체 후보 Top-1은 텍스트→오디오 72.4%, 오디오→텍스트 65.6%이며 모달리티 선형 분류는 100%였습니다. 내용 검색 정렬과 모달리티 편향이 공존합니다. 사용자 검토 3쌍, 일치 가정 218쌍이며 독립 주제 주석은 아직 없습니다. 최신 방법·수치·한계는 [확장 결과 문서](EXP1_FULL_RESULTS.md), 사전 분석 계획은 [EXP1_PROTOCOL.md](EXP1_PROTOCOL.md)를 참조합니다.
