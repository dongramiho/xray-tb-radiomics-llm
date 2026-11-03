# model/ — Chest X-ray TB pipeline (`cxr_tb`)

흉부 X-ray에서 결핵 의심 병변을 찾고(U-Net), 병변 영역의 Radiomics 특징을 수치화하고(PyRadiomics), 결핵 여부를 분류(scikit-learn)하는 파이프라인입니다. 서빙(`WebBack`)과 학습·평가 스크립트가 **같은 코드**(`src/pipeline.py`)로 이미지를 전처리하고 특징을 뽑기 때문에, 학습 때 본 분포와 운영 때 들어오는 분포가 어긋나지 않습니다.

## 설치

```bash
cd model
pip install -e ".[inference]"        # 서빙·평가: tensorflow 2.15, pyradiomics, SimpleITK
pip install -e ".[inference,train]"  # U-Net 재학습까지: + tensorflow-addons, matplotlib
```

`cxr_tb` 패키지로 설치되며, 모든 스크립트는 `python -m cxr_tb.src.<script>` 로 어느 디렉터리에서나 실행할 수 있습니다.

## 구성

```
model/
├── config.py                    # 해상도·학습 하이퍼파라미터·경로·판정 임계값 (한 곳에서 관리)
├── src/
│   ├── pipeline.py              # ★ 전처리 → U-Net → 병변 통계 → Radiomics → 분류기 (서빙/학습 공용)
│   ├── model.py · losses.py     # U-Net 정의, BCE+Dice 손실
│   ├── dataset.py · train.py    # tf.data 파이프라인, U-Net 학습
│   ├── evaluate.py              # 분할 성능 (Dice / IoU, 정답 마스크 필요)
│   ├── model_loader.py          # 구버전 Keras 설정 호환 로더
│   ├── radiomics_extract.py     # PyRadiomics 추출기 설정 (+ CSV 기반 일괄 추출 CLI)
│   ├── build_manifest.py        # 이미지 폴더 → data/dataset.csv (라벨·임상 소견 파싱)
│   ├── extract_dataset_features.py  # 전체 데이터셋에 파이프라인 적용 → 이미지당 1행 CSV
│   ├── evaluate_screening.py    # 이미지 단위 검출 지표 (ROC-AUC, 민감도/특이도)
│   ├── train_classifier.py      # 교차검증 + 분류기 번들(joblib) 내보내기
│   └── infer.py                 # 단일 영상 CLI
└── outputs/
    ├── checkpoints/best_model.h5   # U-Net 가중치 (git 제외, Releases 배포)
    ├── classifier/tb_classifier.joblib + classifier_report.json
    └── evaluation/pipeline_features.csv, screening_metrics.json
```

## 파이프라인 동작 (`src/pipeline.py`)

1. **전처리**: 그레이스케일 → 512×512 bilinear 리사이즈. 학습 시 사용한 `tf.image.resize`(안티앨리어싱 없음)를 NumPy로 그대로 재현해, PIL/OpenCV 리사이즈와의 텍스처 차이로 마스크가 달라지는 문제를 막습니다.
2. **분할**: U-Net 확률맵 → 0.5 임계 → 16px 미만 조각 제거.
3. **병변 통계**: 면적 비율, 연결 성분 수, 성분별 bbox·중심·폐야 구역(right/left × upper/middle/lower, 영상 좌측 = 환자 우측), 최대 확률.
4. **게이트**: 마스크 면적이 영상의 0.1% 미만이면 `no_lesion_detected` 로 종료 (빈 마스크에서는 Radiomics 가 정의되지 않음).
5. **Radiomics**: 512 해상도의 영상·마스크를 SimpleITK 이미지로 바로 넘겨 102개 특징 추출 (shape2D 9, first-order 18, GLCM 24, GLRLM 16, GLSZM 16, GLDM 14, NGTDM 5).
6. **분류**: joblib 번들(estimator + feature_names + threshold + 학습 메타데이터) 로드 → `tb_probability`, `risk_level`(low <0.35 / moderate / high ≥0.65), 상위 기여 특징.

## 데이터 준비와 평가 (Shenzhen 공개 데이터)

```bash
# 1. 데이터 내려받기 (662장, 3.7GB) → model/data/shenzhen/
../scripts/download_shenzhen.sh

# 2. 매니페스트: image_id, image_path, label(1=TB), sex, age, finding
python -m cxr_tb.src.build_manifest

# 3. 전체 이미지에 파이프라인 적용 (U-Net + Radiomics, 이미지당 ~0.3s) → 이미지당 1행
python -m cxr_tb.src.extract_dataset_features --save_overlays

# 4. 이미지 단위 검출 지표 (병변 면적 비율을 점수로 한 ROC-AUC 등)
python -m cxr_tb.src.evaluate_screening

# 5. 분류기 교차검증 + 번들 내보내기 (outputs/classifier/)
python -m cxr_tb.src.train_classifier --model auto
```

결과 수치는 [docs/evaluation.md](../docs/evaluation.md) 에 정리되어 있습니다.

## U-Net 재학습

`data/dataset.csv` 에 `image_path`, `mask_path`(병변 정답 마스크) 컬럼이 있어야 합니다.

```bash
python -m cxr_tb.src.train          # outputs/checkpoints/best_model.h5 (val Dice 기준 best)
python -m cxr_tb.src.evaluate       # test split Dice / IoU
```

512×512 grayscale 입력, encoder–decoder + skip connection, BCE+Dice 손실, Adam(1e-4), batch 4, 최대 100 epoch, EarlyStopping(10)·ReduceLROnPlateau(0.5), 증강(좌우 반전·회전·밝기·대비). 세부 구조는 [MODEL_ARCH.md](MODEL_ARCH.md).

## 단일 영상 실행

```bash
python -m cxr_tb.src.infer --image path/to/cxr.png --output_dir outputs/predictions/single
```

마스크·오버레이 PNG 와 케이스 JSON(분할 통계, 102개 특징, 분류 결과, 단계별 소요 시간)을 저장합니다.
