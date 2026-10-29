# Katie 실행 확인 결과

실행일: 2026-10-28. 실제 EmbeddingGemma 2 추론으로 3쌍 확인 후 23쌍 전체 추론·평가를 완료했습니다. 합성 데이터 결과가 아닙니다.

- 표본: Gleason Mother/Katie 한 아동, 23개 클립, 총 48.475초. 수동 청취 검증 전입니다.
- 모델: `google/embeddinggemma-2`, 지정 revision `f9972ac22e696a105c482202646979e33e4b5a1e`.
- 실행: CPU, float32, batch size 1, vision encoder 제외. 텍스트/오디오 모두 (23, 768), 유한한 정규화 벡터를 확인했습니다.
- 환경: Python 3.11, PyTorch 2.15.0, Transformers 5.21.0, Sentence Transformers 6.2.0. 전체 패키지는 `requirements-macos.lock.txt`에 기록했습니다.
- 23개 텍스트 처리 약 1.3초, 오디오 약 17초였습니다. 진행 표시 기준 대략적인 값이며 모델 로딩·다운로드 시간은 제외합니다. 정식 속도 벤치마크가 아닙니다.

## 정렬·검색 결과

정답 페어 평균 코사인 유사도는 **0.7664**(표준편차 0.0446), 506개 다른 페어의 평균은 **0.7274**, 차이는 **0.0390**입니다.

텍스트→오디오: Recall@1 **56.52% (13/23)**, Recall@5 **82.61% (19/23)**, MRR **0.6626**.

오디오→텍스트: Recall@1 **26.09% (6/23)**, Recall@5 **65.22% (15/23)**, MRR **0.4543**.

참고로 23개 후보에서 균등 무작위 Top-1 선택의 기대값은 4.35%입니다. 이는 유의성 검정 결과가 아닙니다. 원래 임베딩 공간의 모달리티 실루엣은 0.1815이며, 주제 라벨이 없어 주제 실루엣은 계산하지 않았습니다.

이 표본에서 짝을 연결하는 신호가 보이고 양방향 성능 차이가 있습니다. 그러나 단일 아동·동일 세션·소규모 표본이며 정렬을 청취 검증하지 않았으므로 원인을 단정하거나 발달·진단 관련 결론으로 확장하지 않습니다. 정답 페어가 아닌 후보들도 같은 대화 맥락과 유사한 표현을 공유합니다.

## 재실행 및 산출물

저장소 루트에서 실행합니다. 최초 모델 다운로드 후에는 아래처럼 오프라인 실행할 수 있습니다.

```bash
HF_HUB_OFFLINE=1 labs/multimodal/embeddinggemma2/.venv/bin/python labs/multimodal/embeddinggemma2/experiment.py extract --manifest data/katie_smoke.jsonl --output outputs/katie23 --revision f9972ac22e696a105c482202646979e33e4b5a1e --batch-size 1
labs/multimodal/embeddinggemma2/.venv/bin/python labs/multimodal/embeddinggemma2/experiment.py evaluate --input outputs/katie23
```

`outputs/katie23/`에 embeddings.npz, rows.json, provenance.json, metrics.json, pair_scores.json을 저장했습니다. provenance의 requested_revision은 위 commit으로 고정했습니다. 라이브러리 config에서 resolved_revision을 제공하지 않아 해당 필드는 null입니다. 입력 manifest·오디오 해시도 기록돼 있습니다. 데이터와 결과 디렉터리는 Git에서 제외됩니다.

프로세서 초기화에 Pillow와 Torchvision도 필요함을 실제 실행에서 확인해 의존성 목록에 추가했습니다. 별도 모델 접근 승인은 요구되지 않았습니다. 모델은 로컬 실행했고 전사·오디오를 외부 추론 서비스에 업로드하지 않았습니다.

평가·전처리 테스트 5개 및 가상환경 의존성 검사가 통과했습니다. 다음 단계는 낮은 검색 성능 페어부터 청취·전사·경계 검증을 하고, 여러 아동으로 표본을 확장하는 것입니다.


## 긴 대화 Exp 1 재실행 (2026-10-28)

사용자가 처음 3개 구간을 검토해 적합하다고 확인했고, 나머지 구간은 모두 일치한다고 가정하여 실행하도록 요청했습니다. 따라서 **12쌍 중 직접 검토 3쌍, 일치 가정 9쌍**인 탐색 실험입니다. 가정한 자료를 청취 검증 완료로 기록하지 않았습니다.

각 구간은 11.8~24.2초, 총 227.775초, 평균 전사 42.25단어입니다. 엄마와 아이의 전체 발화를 순서대로 포함하되 화자 이름은 임베딩 텍스트에 넣지 않았습니다. 클립 간 겹침은 없고, 동일 세션의 손상 경고 이전 자료만 사용했습니다. 기존과 같은 모델 revision, CPU/FP32, batch size 1로 실제 추론했습니다.

- 정답 페어 평균 유사도: **0.8597** (표준편차 0.0253).
- 다른 132개 페어 평균 유사도: **0.7473**.
- 정답−다른 페어 평균 차이: **0.1124**.
- 텍스트→오디오: Recall@1/5 및 MRR 모두 **1.0** (Top-1 12/12).
- 오디오→텍스트: Recall@1/5 및 MRR 모두 **1.0** (Top-1 12/12).
- 정답−가장 가까운 오답 차이: 텍스트→오디오 중앙값 **0.0725**, 최소 **0.0384**; 오디오→텍스트 중앙값 **0.0708**, 최소 **0.0172**.
- 원래 임베딩 공간의 모달리티 실루엣: **0.2016**. 주제 라벨은 없어 주제별 수치는 없습니다.

사용자가 검토한 첫 3쌍만 별도 실행한 결과도 양방향 Top-1 3/3입니다. 이때 후보 수가 3이므로 Recall@5는 자동으로 1이며 별도 의미가 없습니다. 3쌍 결과는 `outputs/dialogues_verified3/`에 있습니다.

기존 짧은 발화의 평균 차이 0.0390과 비교하면 긴 구간의 정답 분리가 더 뚜렷합니다. 다만 길이 외에도 전체 화자 포함, 후보 수(23→12), 발화 내용 및 표본 구성이 바뀌었습니다. 길이를 늘린 것만의 효과로 해석할 수 없고, 모달리티 실루엣도 낮아진 것은 아닙니다. 이 결과는 같은 대화 구간을 양방향으로 찾을 수 있었다는 증거이며, 모델이 일반적인 상황을 이해한다거나 발달 특성을 판별한다는 증거는 아닙니다. 다음 비교는 여러 세션/아동, 동일한 후보 수, 같은 주제의 어려운 오답을 포함해야 합니다.

재실행:

```bash
HF_HUB_OFFLINE=1 labs/multimodal/embeddinggemma2/.venv/bin/python labs/multimodal/embeddinggemma2/experiment.py extract --manifest data/katie_dialogues_exploratory12.jsonl --output outputs/dialogues_exploratory12 --revision f9972ac22e696a105c482202646979e33e4b5a1e --batch-size 1 --allow-assumed-alignment
labs/multimodal/embeddinggemma2/.venv/bin/python labs/multimodal/embeddinggemma2/experiment.py evaluate --input outputs/dialogues_exploratory12 --allow-assumed-alignment
```

`--allow-assumed-alignment`은 사용자가 명시적으로 가정한 행만 허용합니다. 기본 검증 차단은 유지됩니다. 입력·벡터·개별 점수·모델 및 데이터 해시는 `outputs/dialogues_exploratory12/`에 저장했고, 검토 상태를 metrics.json에도 기록했습니다. 전처리·평가 테스트 8개가 통과했습니다.


## 확장 Exp 1 결과 (2026-10-28)

6명·8세션·221쌍의 실제 추론과 확장 평가를 완료했습니다. 전체 후보 Top-1은 텍스트→오디오 72.4%, 오디오→텍스트 65.6%이며 모달리티 선형 분류는 100%였습니다. 내용 검색 정렬과 모달리티 편향이 공존합니다. 사용자 검토 3쌍, 일치 가정 218쌍이며 독립 주제 주석은 아직 없습니다. 최신 방법·수치·한계는 [확장 결과 문서](EXP1_FULL_RESULTS.md), 사전 분석 계획은 [EXP1_PROTOCOL.md](EXP1_PROTOCOL.md)를 참조합니다.
