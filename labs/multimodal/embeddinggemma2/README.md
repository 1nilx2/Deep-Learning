# EmbeddingGemma 2: audio–text alignment experiments

아동·보호자 대화의 전사와 오디오 정렬을 평가하는 실험입니다. 최신 [Exp 1 보고서](EXP1_FULL_RESULTS.md)는 6명·8세션·221쌍의 검색 성능과 모달리티 편향을 분석합니다. 실제 추론 결과는 [RESULTS.md](RESULTS.md)에 기록했습니다. Exp 2~4는 이 파일럿의 정렬 성능과 데이터 메타데이터를 확인한 뒤 확장합니다.

연구 가설과 설계 보완 기록은 [실험 설계 문서](embeddinggemma2_experimental_design.md)에서 관리합니다.

## 실행

저장소 루트에서 실행합니다. 환경·캐시·입력·결과는 이 lab 아래에 둡니다.

```bash
python3 -m venv labs/multimodal/embeddinggemma2/.venv
labs/multimodal/embeddinggemma2/.venv/bin/python -m pip install --cache-dir labs/multimodal/embeddinggemma2/.cache/pip -r labs/multimodal/embeddinggemma2/requirements.txt
labs/multimodal/embeddinggemma2/.venv/bin/python labs/multimodal/embeddinggemma2/experiment.py validate --manifest data/pilot.jsonl
labs/multimodal/embeddinggemma2/.venv/bin/python labs/multimodal/embeddinggemma2/experiment.py extract --manifest data/pilot.jsonl --output outputs/pilot
labs/multimodal/embeddinggemma2/.venv/bin/python labs/multimodal/embeddinggemma2/experiment.py evaluate --input outputs/pilot
```

모델 저장소가 접근 승인을 요구하면 Hugging Face 계정에서 Gemma 이용 조건에 동의하고 `HF_TOKEN` 환경변수를 사용하세요. 토큰을 파일이나 Git에 넣지 마세요. 재현 실행에는 `extract --revision <model commit hash>`를 지정하세요. 최초 실행에는 패키지와 모델 다운로드가 필요합니다. 자동 장치 선택은 CUDA 또는 CPU이며, BF16 지원 CUDA에서만 BF16을 사용하고 나머지는 FP32입니다. MPS는 `--device mps`로 명시적으로 시도할 수 있지만 오디오 경로 호환성은 검증하지 않았습니다. FP16은 사용하지 않습니다.

코퍼스 후보와 접근 상태: [데이터 조사 기록](DATA_SELECTION.md).

## 데이터 계약

`data/pilot.jsonl`에 한 줄당 한 개의 JSON 객체를 넣으세요. 경로는 모두 **이 lab 폴더 기준**입니다.

```json
{"id":"corpus-child01-session01-u001","child_id":"corpus-child01","session_id":"session01","text":"I want the red ball .","audio_path":"data/audio/u001.wav","topic":"play","age_months":36,"diagnosis":null}
```

필수 키: `id`, `child_id`, `session_id`, `text`, `audio_path`. `topic`, `age_months`, `diagnosis`는 선택입니다. 최소 두 쌍이 필요하며 실제 파일럿은 여러 아동에서 50~200개 정도의 발화를 권장합니다. ID는 코퍼스 간에도 겹치지 않도록 만드세요.

- `audio_path`에는 해당 아동 발화만 잘라낸 로컬 파일을 넣습니다. 세션 전체 오디오나 아직 잘라내지 않은 타임스탬프 행은 사용하지 않습니다. 자동 CHAT 파싱·시간 정렬·클립 생성은 현재 범위에 포함되지 않습니다.
- 전사는 관찰된 발화를 나타냅니다. 정답 의미나 아동의 의도를 보장하지 않습니다. CHAT 주석 제거, 불명료 발화 처리, 중첩 발화 제외 기준을 데이터 준비 시 기록하세요.
- 진단은 실제 코퍼스 라벨이 있을 때만 입력합니다. 누락 라벨을 정상 발달로 바꾸지 않습니다. MLU는 발화 길이가 아닌 정의된 아동/세션 단위 지표로 별도 산출해야 합니다.
- CHILDES의 전사·미디어·진단 메타데이터 가용성 및 이용 조건은 코퍼스마다 확인해야 합니다. 원본과 전사 및 임베딩은 Git에서 제외됩니다.

## 산출물과 해석

`embeddings.npz`는 정규화된 텍스트/오디오 벡터, `rows.json`은 동일 순서의 메타데이터입니다. `provenance.json`에는 모델 revision, 패키지 버전, 장치, 정밀도, 입력 해시가 저장됩니다. 같은 출력 폴더를 재사용하면 덮어씁니다.

`metrics.json`에는 올바른 페어 대 전체 잘못된 페어의 평균 코사인 유사도, 양방향 Recall@1/5·MRR, 원래 768차원 공간의 모달리티·주제 실루엣을 기록합니다. 주제 라벨이 없거나 유효한 여러 그룹을 만들지 못하면 주제 실루엣은 null입니다. `pair_scores.json`은 발화별 코사인 유사도와 Gap을 제공합니다. 정확히 같은 점수의 검색 후보는 보수적으로 정답보다 앞선 순위로 계산합니다.

모든 잘못된 페어를 비교하므로 계산·메모리는 O(N²)입니다. 소규모 파일럿용이며, 발화가 같은 전사를 공유하면 정확한 페어 검색에 거짓 음성이 생길 수 있습니다. Recall@5는 후보가 5개 이하일 때 정보가 거의 없습니다. 현재 지표는 기술 통계이며 아동 간 일반화나 진단 성능의 증거가 아닙니다.

## 검증과 다음 단계

```bash
labs/multimodal/embeddinggemma2/.venv/bin/python -m unittest discover -s labs/multimodal/embeddinggemma2 -v
```

합성 벡터로 올바른/뒤섞인 페어, 양방향 검색, 동률, 잘못된 입력을 검증합니다. 합성 테스트는 모델의 실제 정렬 성능을 검증하지 않습니다.

실제 페어에서 정렬이 확인되면 UMAP 시각화와 선택적 HDBSCAN 탐색을 추가합니다. 이후 Exp 2는 SNR·열화 강도별 성능 곡선, Exp 3은 아동 단위 분할의 선형 probe 및 MLP 비교, Exp 4는 아동 단위 Gap 분석과 신뢰구간·교란요인 통제로 확장합니다. 발달군 차이의 방향과 임상적 유효성은 사전 가정하지 않습니다.

공식 구현 참고: [Google 멀티모달 가이드](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers), [CHILDES 접근 안내](https://talkbank.org/childes/access.html).


## 확보한 데이터의 준비 결과 (2026-10-28)

Gleason Mother/Katie 세션에서 **실행 확인용 23쌍**을 만들었습니다. 한 아동의 자료이므로 여러 아동의 연구용 파일럿을 대체하지 않습니다. 실제 모델 추론이나 수동 청취는 아직 수행하지 않았습니다.

```bash
python3 labs/multimodal/embeddinggemma2/prepare_gleason.py
labs/multimodal/embeddinggemma2/.venv/bin/python labs/multimodal/embeddinggemma2/experiment.py validate --manifest data/katie_smoke.jsonl
```

`prepare_gleason.py`는 확보된 Katie 자료에만 적용하는 보수적 전처리입니다. 일반 CHAT 파서가 아닙니다. 원본의 음성 손상 경고 이후 모든 발화, 복잡한 CHAT 표기, 불명료 발화, 다른 화자와 시간상 중첩된 발화, 0.7초 미만/10초 초과, 두 단어 미만 및 중복 전사를 제외합니다. 단순 pause 표기와 장음 기호를 제거하고 밑줄로 연결된 단어를 분리합니다. 원본 전사와 시간·행 번호는 유지합니다.

로컬 산출물:

- `data/katie_smoke.jsonl`: 추론 입력 manifest
- `data/clips/gleason-katie/`: 발화별 원본 샘플레이트 PCM WAV
- `data/preparation_report.json`: 제외 사유, 원본 해시, 출처 및 검증 상태
- `data/LISTENING_REVIEW.md`: 클립별 청취 확인 목록

원본은 318개의 아동 발화 행을 포함합니다. 경고 이후 229개 행을 제외했으며 나머지에도 보수적 필터를 적용해 23쌍을 확보했습니다. 진단은 null로 두고 원본 그룹 TD를 별도 필드에 보존했습니다. 나이는 원본 `3;02.02` 및 완료 개월 수 38을 함께 보존합니다. 수동 청취가 완료되지 않았으므로 `manual_alignment_verified`는 false입니다. 검증 전 연구용 정렬 품질이 확보됐다고 간주하지 않습니다.


## 실제 추론 완료

2026-10-28: Katie 3쌍 실행 확인과 23쌍 전체 CPU/FP32 추론·평가를 완료했습니다. 앞선 준비 상태 기록 중 “추론 미수행”은 준비 당시 상태입니다. 최신 결과는 [RESULTS.md](RESULTS.md)를 기준으로 봅니다. 수동 청취 및 여러 아동으로의 확장은 아직 진행 전입니다.


## 긴 대화 Exp 1: 내용·경계 검증 대기

`prepare_dialogues.py`로 11.8~24.2초 길이의 비중첩 후보 12개(전사 총 507단어)를 준비했습니다. 기존 단일 아동 발화와 달리 엄마·아이 등 구간 내 모든 화자의 발화를 포함합니다. 시간표시가 없거나 복잡한 주석·불명료 발화·중첩·손상이 있는 행을 건너뛰어 하나의 구간 안에서 이어 붙이지 않습니다. 완전한 연속 발화 구간만 선택하고 300ms보다 긴 행간 공백도 끊습니다. 클립 경계와 전사 범위는 같은 원본 타임스탬프를 사용하며 샘플 인덱스를 기록합니다. 실측 음성 경계가 맞는지는 별도 검증이 필요합니다.

로컬 검토본은 `data/DIALOGUE_REVIEW.md`, 입력 후보는 `data/katie_dialogues_candidates.jsonl`, WAV는 `data/dialogues/`입니다. 아직 청취 검증이 되지 않아 추론·평가를 실행하지 않았습니다. 긴 대화 자료는 `manual_alignment_verified`가 true인 행만 추론·평가하도록 차단했습니다. 실제 내용 누락 및 시작·끝 일치를 확인하고 필요한 수정을 마친 페어만 별도 manifest로 선정합니다. 자동 타임스탬프 검사를 청취 검증으로 대체하지 않습니다.


## 긴 대화 실행 결과 갱신

첫 3쌍 사용자 검토 및 나머지 9쌍 일치 가정에 따라 전체 12쌍의 실제 추론·평가를 완료했습니다. 양방향 Top-1 12/12, 평균 정답−다른 페어 차이는 0.1124입니다. `--allow-assumed-alignment`을 명시한 탐색 실행이며 9쌍의 수동 검증 상태는 false로 유지합니다. 최신 수치와 비교 한계는 RESULTS.md를 참조합니다.


## 확장 Exp 1 결과 (2026-10-28)

6명·8세션·221쌍의 실제 추론과 확장 평가를 완료했습니다. 전체 후보 Top-1은 텍스트→오디오 72.4%, 오디오→텍스트 65.6%이며 모달리티 선형 분류는 100%였습니다. 내용 검색 정렬과 모달리티 편향이 공존합니다. 사용자 검토 3쌍, 일치 가정 218쌍이며 독립 주제 주석은 아직 없습니다. 최신 방법·수치·한계는 [확장 결과 문서](EXP1_FULL_RESULTS.md), 사전 분석 계획은 [EXP1_PROTOCOL.md](EXP1_PROTOCOL.md)를 참조합니다.
