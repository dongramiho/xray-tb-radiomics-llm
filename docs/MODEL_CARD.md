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

## 판정 규칙

| 상황 | `primary_label` | `risk_level` |
| --- | --- | --- |
| 마스크 면적 < 0.1% | `no_lesion_detected` | low |
| 분류기 확률 < 0.5 | `low_tb_likelihood` | low (<0.35) / moderate |
| 분류기 확률 ≥ 0.5 | `tb_suspected` | moderate / high (≥0.65) |
| 분류기 미탑재 | `lesion_detected_unclassified` | moderate |

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
