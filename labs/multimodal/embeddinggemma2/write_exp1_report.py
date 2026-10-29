"""Render the expanded experiment record from saved results."""
import json
import shutil
from pathlib import Path
from experiment import ROOT


def main():
    out=ROOT/'outputs/exp1_full'
    assets=ROOT/'assets/exp1'
    assets.mkdir(parents=True,exist_ok=True)
    for filename in ['umap.png','distributions.png']:
        shutil.copy2(out/filename,assets/filename)
    m=json.loads((out/'full_metrics.json').read_text())
    prep=json.loads((ROOT/'data/exp1_full_preparation.json').read_text())
    percent=lambda x:f'{x*100:.1f}%'
    lines=['# 확장 Exp 1: 검색 정렬과 모달리티 편향', '',
           '실행일: 2026-10-28. 실제 모델 추론 결과입니다. **대화 내용의 cross-modal retrieval은 상당히 가능하지만, 모달리티 편향이 극복됐다는 가설은 지지되지 않았습니다.**', '',
           '## 데이터 및 검증 상태', '',
           f'Gleason 코퍼스 **{m["n_children"]}명·{len(prep["sessions"])}세션·{m["n_pairs"]}쌍**, 총 {prep["total_seconds"]/60:.2f}분입니다. 전체 화자의 연속 대화 구간(10~30초)을 사용했습니다. 사용자 청취 검토 3쌍과 사용자 요청에 따른 내용·경계 일치 가정 218쌍으로 구성됩니다. 독립적인 전체 청취 검증이 완료된 자료로 취급하지 않습니다.', '',
           '처음 뽑은 222쌍 중 길이 조건을 벗어난 38.7초 단일 발화 1개를 점수 확인 전에 제외했습니다. 생성된 모든 최종 클립에서 프레임 수, PCM payload 길이, 10~30초 조건, 포함된 전사 행의 시간 범위를 확인했습니다. 원본 Katie WAV는 헤더보다 실제 payload가 짧아 실제 읽은 프레임과 전사의 손상 경고를 적용했습니다. 타임스탬프 기준 일치가 실제 언어 내용 일치를 증명하지는 않습니다.', '',
           '세션별 최대 30개를 seed 20261028로 선택했습니다. Andy·Frank는 엄마/아빠와의 두 세션, 나머지 아동은 한 세션입니다. 다운로드 크기와 시간 정렬 가용성에 따른 편의 표본이며, 코퍼스·집단 대표성을 보장하지 않습니다.', '',
           '## 주요 검색 결과', '',
           '| 지표 | 텍스트 → 오디오 | 오디오 → 텍스트 |',
           '| :--- | ---: | ---: |',
           f'| Top-1 / 후보 221개 | {percent(m["text_to_audio"]["recall_at_1"])} (160/221) | {percent(m["audio_to_text"]["recall_at_1"])} (145/221) |',
           f'| Top-5 / 후보 221개 | {percent(m["text_to_audio"]["recall_at_5"])} | {percent(m["audio_to_text"]["recall_at_5"])} |',
           f'| MRR | {m["text_to_audio"]["mrr"]:.4f} | {m["audio_to_text"]["mrr"]:.4f} |', '',
           f'정답 평균 코사인은 **{m["matched_cosine_mean"]:.4f}**, 다른 48,620개 페어 평균은 **{m["mismatched_cosine_mean"]:.4f}**, 차이는 **{m["matched_minus_mismatched"]:.4f}**입니다. 무작위 Top-1 기대값은 후보 221개에서 약 0.45%이나 이것을 통계적 유의성 검정으로 해석하지 않습니다.', '',
           '같은 세션 후보(12~30개)로 제한했을 때 발화 수 가중 Top-1은 **텍스트→오디오 83.7%, 오디오→텍스트 81.0%**입니다. 후보 수가 더 적어 전체 검색과 난이도가 동일하지 않습니다. 동일 세션 안에서도 대응 정보를 찾을 수 있다는 보조 증거입니다.', '',
           f'후보를 무작위 12개로 줄인 500개 부분집합에서 평균 Top-1은 **{percent(m["fixed_12_candidate_subsets"]["text_recall1_mean"])} / {percent(m["fixed_12_candidate_subsets"]["audio_recall1_mean"])}**입니다. 초기 Katie 12쌍의 100%가 그대로 일반화되지는 않았습니다. 반복 부분집합은 독립 실험도 신뢰구간도 아닙니다.', '',
           '## 아동별 결과', '',
           '모든 수치는 전체 221개 후보에서의 Top-1입니다.', '',
           '| 아동 | 페어 수 | 텍스트 → 오디오 | 오디오 → 텍스트 |',
           '| :--- | ---: | ---: | ---: |']
    for child,r in m['per_child'].items():
        lines.append(f'| {child.removeprefix("gleason-")} | {r["n_pairs"]} | {percent(r["global_text_recall_at_1"])} | {percent(r["global_audio_recall_at_1"])} |')
    uncertainty=m['uncertainty']['matched_minus_same_session_negative']
    lines += ['', '## 불확실성 및 편향', '',
              f'정답−동일 세션 오답 평균 차이를 아동별로 집계한 macro 평균은 **{uncertainty["child_macro_mean"]:.4f}**, 아동 단위 bootstrap 2,000회 95% 구간은 **[{uncertainty["child_bootstrap_95ci"][0]:.4f}, {uncertainty["child_bootstrap_95ci"][1]:.4f}]**입니다. 아동이 6명뿐이므로 모집단 추론에 충분한 정밀도를 보장하지 않습니다.', '',
              '동일 세션 내 대응을 섞은 permutation 1,000회에서 단측 p=0.0010(시뮬레이션 해상도 하한)이었습니다. 세션 내부 발화 교환 가능성이 근사 가정이고 시간 자기상관은 제거되지 않았으므로 탐색적 보조 검정입니다.', '',
              f'모달리티 실루엣은 원래 768차원 공간에서 **{m["modality_silhouette_cosine"]:.4f}**입니다. 아동별 leave-one-child-out 선형 모달리티 분류의 balanced accuracy는 모든 6개 fold에서 **100%**였습니다. 즉 새 아동에서도 텍스트/오디오를 구분하는 신호가 매우 강하게 남았습니다. 이는 대화 의미 정렬과 공존할 수 있습니다.', '',
              f'아동 실루엣 **{m["child_silhouette_cosine"]:.4f}**, 세션 실루엣 **{m["session_silhouette_cosine"]:.4f}**이며, 높은 검색 성능이 아동별 군집만으로 설명된다는 증거는 이 지표에서 보이지 않았습니다. 음성·상황 교란이 제거됐다는 의미는 아닙니다.', '',
              '## UMAP 및 HDBSCAN', '',
              '![UMAP](assets/exp1/umap.png)', '',
              '왼쪽은 모달리티, 가운데는 아동, 오른쪽은 전사 어휘 규칙에 따른 약한 주제 라벨입니다. 모달리티에 따른 분리가 뚜렷합니다. UMAP은 투영이므로 2D 거리를 원래 코사인 거리처럼 해석하지 않습니다. seed 42·7·123의 투영을 저장했습니다.', '',
              '![Similarity distributions](assets/exp1/distributions.png)', '',
              '정답 점수는 오른쪽으로 이동하지만 오답과 일부 겹칩니다. 오른쪽의 margin은 정답 점수에서 가장 높은 오답 점수를 뺀 것으로, 0 이하이면 정답이 단독 1위가 아닙니다.', '',
              f'HDBSCAN은 정규화 원본 768d 벡터, Euclidean, min_cluster_size=10, min_samples=5로 **{m["hdbscan"]["n_clusters"]}개 군집**, noise **{percent(m["hdbscan"]["noise_fraction"])}**를 만들었습니다. 비noise 군집 실루엣은 **{m["hdbscan"]["cluster_silhouette_cosine"]:.4f}**입니다. 군집 하나는 오디오 199개·텍스트 16개, 다른 하나는 텍스트 28개로 구성되어 주제별 공통 군집이라고 해석할 수 없습니다.', '',
              '## 주제별 가설의 현재 한계', '',
              '원본에는 독립적인 구간별 주제 정답이 없습니다. 고정 어휘 규칙 보조 분석에서는 animals 17, vehicles 19, food 15, book_story 9, health 1, mixed 13, unassigned 147쌍입니다. 단일 약한 주제 라벨의 실루엣은 0.0696이지만, 이 라벨은 골드 주제 주석을 대체하지 않습니다. 미할당이 다수이고 실제로 같은 책의 서로 다른 장면처럼 의미적으로 어려운 오답을 온전히 식별하지 못합니다.', '',
              '따라서 **확장 Exp 1의 검색·편향·시각화·군집 실행은 완료**했지만, 원래 가설의 엄격한 주제별 공통 군집 검증은 독립적인 주제 주석 후 판단해야 합니다. `data/exp1_full_topic_annotation.csv`에 주석 시트를 준비했습니다.', '',
              '## 실패 사례와 다음 검증', '',
              '실패 목록에는 같은 책에서 고양이·경찰관·아이를 묘사하는 다른 장면을 혼동한 사례와, “이것/저것” 위주로 구체적인 대상이 전사에 드러나지 않는 대화가 있습니다. 이러한 내용 유사성과 실제 시간 정렬 오류를 분리하려면 해당 실패 페어를 청취 확인해야 합니다. 점수가 낮다는 이유만으로 표본에서 제거하지 않습니다.', '',
              '이 결과는 성인과 아동을 포함한 대화 구간의 언어 내용 정렬을 평가합니다. 아동 단독 음성의 발달 정보나 임상적 유효성을 평가한 결과가 아닙니다. 후속 검증은 실패 페어의 내용·경계 확인, 독립 주제 주석, 다른 코퍼스 재현입니다.', '',
              '## 재현 및 산출물', '',
              '고정 분석 계획은 `EXP1_PROTOCOL.md`, 표본 준비는 `prepare_exp1_full.py`, 확장 평가는 `evaluate_exp1_full.py`입니다. `requirements-macos.lock.txt`에 환경을 기록했습니다. 모델 revision은 f9972ac22e696a105c482202646979e33e4b5a1e, CPU/FP32, batch size 1입니다.', '',
              '```bash',
              'python3 labs/multimodal/embeddinggemma2/prepare_exp1_full.py',
              'HF_HUB_OFFLINE=1 labs/multimodal/embeddinggemma2/.venv/bin/python labs/multimodal/embeddinggemma2/experiment.py extract --manifest data/exp1_full_manifest.jsonl --output outputs/exp1_full --revision f9972ac22e696a105c482202646979e33e4b5a1e --batch-size 1 --allow-assumed-alignment',
              'labs/multimodal/embeddinggemma2/.venv/bin/python labs/multimodal/embeddinggemma2/evaluate_exp1_full.py',
              '```', '',
              '이번 실제 실행에서는 추론 도중 발견한 길이 부적합 1개를 `finalize_exp1_sample.py`로 이미 추출된 벡터에서 제외했습니다. 초기 추론은 `outputs/exp1_full_before_qc/`, 최종 221쌍은 `outputs/exp1_full/`에 별도로 보존했습니다. 제외는 검색 점수 확인 전 고정 길이 기준에 따라 수행했으며 최종 provenance에 원본 추론·제외 ID·manifest 해시를 기록했습니다.', '',
              '최종 산출물은 embeddings.npz, rows.json, provenance.json, full_metrics.json, analysis_provenance.json, query_diagnostics.json, failure_examples.json, cluster_labels.json, UMAP 3개 seed의 좌표, 분포·UMAP 그림입니다. 221개 PCM 클립 검사, 테스트 10개, 패키지 의존성 검사를 통과했습니다. 데이터 및 원시 결과는 Git에서 제외합니다. 시각화에는 집계값과 전사가 포함돼 있어 로컬 검토용으로 보관합니다.', '',
              'GitHub에서는 이 보고서와 `assets/exp1/`의 정적 그림, `assets/exp1/summary.json`의 집계 결과를 읽을 수 있습니다. 원본 음성·전사·개별 페어와 전사가 포함된 인터랙티브 시각화는 커밋 대상에서 제외했습니다. 로컬 인터랙티브 그림은 `visuals/exp1-alignment.html`이며, GitHub Markdown에서는 실행되지 않습니다.', '',
              '## 출처', '',
              '- [Gleason 코퍼스와 인용 문헌](https://talkbank.org/childes/access/Eng-NA/Gleason.html)',
              '- [Gleason corpus DOI](https://doi.org/10.21415/T5101R)',
              '- [Google 공식 멀티모달 EmbeddingGemma 2 가이드](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)',
              '- Bellinger, D., & Gleason, J. (1982). Sex differences in parental directives to young children. Sex Roles, 8, 1123–1139. 코퍼스 페이지에 명시된 인용 문헌 중 하나입니다.']
    lines[2:2]=['![Exp 1 overview](assets/exp1/overview.svg)', '']
    (ROOT/'EXP1_FULL_RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
