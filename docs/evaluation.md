# 평가 결과 (Shenzhen CXR, 662장)

마지막 갱신: 2026-11-02 · 재현: `model/README.md` 의 "데이터 준비와 평가" 5단계 · 원본 수치: `model/outputs/evaluation/screening_metrics.json`, `model/outputs/classifier/classifier_report.json`

## 데이터

| 항목 | 값 |
| --- | --- |
| 출처 | Shenzhen No.3 People's Hospital chest X-ray set (NIH/NLM 공개) |
| 영상 수 | 662장 (TB 336 · 정상 326), 환자당 1장 |
| 라벨 | 영상 단위 (TB / 정상). 임상 소견 텍스트 동봉 |
| 병변 마스크 | 정상 영상에는 없음. TB 영상 약 330장의 병변 마스크로 U-Net 을 학습 (팀 프로젝트 단계) |

> **주의**: U-Net 은 이 데이터셋의 TB 영상 대부분으로 학습되었습니다. 따라서 아래 수치 중 TB 쪽 분할 결과는 **in-sample** 이고, 정상 영상 326장은 학습에 쓰인 적 없는 **out-of-sample** 입니다. 분류기 수치는 5-fold 교차검증(out-of-fold)입니다. 외부 데이터셋 검증은 아직 없습니다.

## 1. 분할 게이트 (이미지 단위 검출)

U-Net 마스크 면적 비율(영상 대비)을 TB 점수로 썼을 때:

| 지표 | 값 |
| --- | --- |
| ROC-AUC (면적 비율) | **0.868** |
| ROC-AUC (최대 픽셀 확률) | 0.868 |
| Average precision | 0.887 |
| 운영 게이트 0.1% 에서 민감도 / 특이도 | 99.1% / 12.9% |
| Youden 최적 임계(1.8%)에서 민감도 / 특이도 | 72.0% / 88.3% |

운영 게이트(0.1%)는 "놓치지 않는" 쪽으로 낮게 잡고, 특이도는 다음 단계 분류기가 책임집니다. 정상 영상 326장 중 284장(87%)에서 U-Net 이 작은 위양성 영역을 만들어 냅니다 (면적 중앙값 0.7%, TB 는 2.8%).

## 2. Radiomics 분류기 (영역 단위, 5-fold stratified CV)

학습 행 617개 = 게이트를 통과한 영상 (TB 333 · 정상 위양성 영역 284), 특징 102개.

| 모델 | ROC-AUC (fold 평균 ± SD) | OOF AUC | AP | 민감도 @0.5 | 특이도 @0.5 | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest (400 trees, balanced) | **0.891 ± 0.025** | 0.890 | 0.913 | 82.6% | 83.5% | 0.133 |
| Logistic Regression (C=0.5, balanced) | 0.887 ± 0.025 | 0.888 | 0.914 | 77.5% | 84.9% | 0.133 |

내보낸 번들은 Random Forest 입니다 (`model/outputs/classifier/tb_classifier.joblib`, 1.2MB).

상위 특징 (RF importance): `glszm_SizeZoneNonUniformity` 0.061, `glrlm_RunLengthNonUniformity` 0.052, `glszm_GrayLevelNonUniformity` 0.042, `glrlm_GrayLevelNonUniformity` 0.032, `shape2D_Perimeter` 0.029.
특징군 기여: GLCM 22.5% · GLSZM 20.0% · first-order 15.7% · GLRLM 15.5% · shape2D 12.7% · GLDM 7.8% · NGTDM 5.8%.

## 3. 종단 (게이트 + 분류기, 이미지 단위)

게이트를 통과하지 못한 영상은 "정상"으로, 통과한 영상은 분류기 OOF 확률 ≥ 0.5 를 "TB" 로 판정:

| 지표 | 값 |
| --- | --- |
| ROC-AUC | **0.897** |
| Average precision | 0.910 |
| 민감도 | 81.9% (275 / 336) |
| 특이도 | 85.6% (279 / 326) |
| 정확도 | 83.7% |
| 혼동행렬 | TP 275 · FP 47 · TN 279 · FN 61 |

## 4. 품질 검사 (QC) 통과율

파이프라인의 QC 단계를 662장 전체에 적용한 결과 (`screening_metrics.json` 의 `quality_control`):

| 지표 | 값 |
| --- | --- |
| QC 통과율 (`qc.passed`) | **100%** — 662장 모두 정면 흉부 영상 조건(크기·대비·비율)과 특징 유효성을 만족 |
| 환자용 설명 사용 가능 비율 (`usable_for_patient`) | 93.4% |
| 플래그 분포 | `tiny_roi` 44건 (분할 영역이 영상의 0.3% 미만 → 질감 특징 신뢰도 낮음, 환자용 설명 제한) |

평가셋에 포함한 160px 축소 영상은 `image_too_small` 로 QC 에 실패하며, 이 경우 리포트는 신뢰도 저하와 의사 검토를 명시합니다 (`evals/run_evals.py` 의 `qc_handling` 지표).

## 5. 처리 속도 (Apple M-series CPU, TensorFlow 2.15)

| 단계 | 시간 |
| --- | --- |
| 전처리 (3000×3000 → 512) | ~40 ms |
| U-Net 추론 | ~300 ms |
| PyRadiomics 102개 특징 | ~30 ms |
| 분류기 + QC | <5 ms |
| **API `/analyze` (LLM 제외)** | **~0.3–0.5 s** |
| CT 마스크 경로 (64×64×48 팬텀, 3D Radiomics 107개) | ~0.4 s |

## 종전 수치와의 차이

팀 프로젝트 단계의 `classify_report.json` 은 정상군 100개를 **합성(synthetic)** 으로 만들어 ROC-AUC 1.0 이 나왔습니다. 이 버전은 합성 데이터를 제거하고 실제 정상 영상으로 평가했으며, 그 결과가 위의 0.89 입니다. 수치는 내려갔지만 비교 가능한 수치입니다.

## 한계

- 단일 기관 공개 데이터, 외부 검증 없음.
- TB 쪽 분할은 in-sample (U-Net 학습 데이터 포함). 정상 쪽만 out-of-sample.
- 라벨이 영상 단위라 "병변 위치가 맞는지"는 평가하지 못함 (정상 영상에는 마스크가 없음).
- 임계값 0.5 는 임상 운영점으로 조정된 값이 아님.
- 연구·교육용 프로토타입. 의료기기가 아니며 임상 판단을 대체하지 않음.
