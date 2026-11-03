# Model Card — X-ray TB Radiomics Pipeline

## 개요

| 항목 | 내용 |
| --- | --- |
| 목적 | 흉부 X-ray 에서 결핵 의심 영역을 분할하고, 영역의 Radiomics 특징으로 TB 가능성을 점수화해 임상의용 리포트의 근거로 제공 |
| 구성 | U-Net(세그멘테이션) → 게이트 → PyRadiomics(102 특징) → Random Forest(분류) → LLM 2단계(해석) |
| 입력 | 정면 흉부 X-ray PNG/JPEG (내부적으로 512×512 grayscale) |
| 출력 | 병변 마스크·오버레이, 구역(zone)·면적 통계, `tb_probability`, `risk_level`, 구조화 JSON 리포트, 한국어 요약 |
| 용도 제한 | 연구·교육용. 의료기기가 아니며 진단·치료 결정을 대체하지 않음 |

## 학습 데이터

- **U-Net**: Shenzhen 세트의 TB 영상 약 330장 + 병변 마스크 (팀 프로젝트 단계에서 학습, 80/10/10 분할, seed 42). 정상 영상은 학습에 사용하지 않음.
- **분류기**: 같은 세트 662장 전체에 U-Net 을 적용해 얻은 영역 617개 (TB 333 · 정상 영상의 위양성 영역 284). 합성 데이터 없음.
- 인구 통계: Shenzhen No.3 People's Hospital 외래, 성별·연령은 임상 소견 파일에 포함 (매니페스트에 파싱됨).

## 성능

[docs/evaluation.md](evaluation.md) 참고. 요약: 영역 단위 분류기 5-fold ROC-AUC 0.891 ± 0.025, 종단(게이트+분류기) 이미지 단위 ROC-AUC 0.897, 민감도 81.9%, 특이도 85.6%.

## 품질 검사 (QC)

분석마다 입력·분할·특징을 검사해 `qc` 블록을 남깁니다. 차단 플래그(`image_too_small`, `low_contrast`, `unusual_aspect`, `huge_roi`, `nan_features`)가 있으면 `passed=false` 로 보고하고, `tiny_roi`·`low_confidence` 는 통과하되 환자용 설명에는 쓰지 않습니다(`usable_for_patient=false`). LLM 리포트는 플래그를 신뢰도 서술과 한계에 반영하고, 환자용 설명은 결과 대신 "신뢰도 낮음·의사 검토" 를 전합니다. 임계값은 `model/config.py` 의 `QC_*` 입니다.

## CT 경로 (마스크 입력)

CT 분할 모델은 없습니다. 외부(nnU-Net 등)에서 만든 NIfTI 라벨맵(0 배경 / 1 간 / 2 종양)과 볼륨을 받아 `model/src/ct.py` 가 기하 일치·라벨 존재·ROI 크기·슬라이스 간격을 검사하고, 종양 라벨에 3D PyRadiomics(shape·first-order·GLCM·GLRLM·GLSZM·GLDM·NGTDM 107개)를 계산해 부피·길이·HU 통계와 함께 같은 리포트 흐름으로 보냅니다. **종양 라벨이 없으면 "종양 없음"이 아니라 "라벨 없음"** 으로 보고하고 검토를 권고합니다. 분류기는 없으므로 `tb_probability` 는 null 이며 특성화만 제공합니다.

## 판정 규칙

| 상황 | `primary_label` | `risk_level` |
| --- | --- | --- |
| 마스크 면적 < 0.1% | `no_lesion_detected` | low |
| 분류기 확률 < 0.5 | `low_tb_likelihood` | low (<0.35) / moderate |
| 분류기 확률 ≥ 0.5 | `tb_suspected` | moderate / high (≥0.65) |
| 분류기 미탑재 | `lesion_detected_unclassified` | moderate |
| CT: 종양 라벨 있음 | `ct_lesion_characterized` | moderate (분류기 없음) |
| CT: 종양 라벨 없음 | `ct_tumor_label_missing` | low (라벨 부재 ≠ 종양 없음) |

## 알려진 한계와 실패 양상

- 정상 영상의 87% 에서 U-Net 이 작은 위양성 영역을 만들며, 이를 분류기가 걸러낸다. 분류기가 놓치면 위양성 리포트가 나간다.
- 학습 분포(Shenzhen, 성인, 정면 PA)와 다른 장비·소아·측면 영상·회전된 영상에서는 성능을 보장하지 못한다.
- 폐외 결핵, 흉수 단독, 미세 결절은 분할 대상 밖이다.
- 영상 단위 라벨로만 평가했기 때문에 "맞는 위치를 잡았는지"는 검증되지 않았다.
- LLM 리포트는 파이프라인 출력에 근거하도록 프롬프트로 제약하지만, 생성 모델의 오류 가능성은 남아 있다.

## 재현

```bash
scripts/download_shenzhen.sh
cd model && pip install -e ".[inference]"
python -m cxr_tb.src.build_manifest
python -m cxr_tb.src.extract_dataset_features --save_overlays
python -m cxr_tb.src.evaluate_screening
python -m cxr_tb.src.train_classifier --model auto
```

## 버전

| 구성요소 | 버전 |
| --- | --- |
| `cxr_tb` 패키지 | 0.2.0 |
| U-Net 가중치 | `best_model.h5` (2025-11 학습, 7.8M 파라미터) |
| 분류기 번들 | `tb_classifier.joblib` (scikit-learn 1.9, 2026-11-02) |
