# Chest X-ray TB 분석 및 Radiomics 파이프라인

> **💡 요약 (Summary)**  
> 이 폴더는 흉부 X-ray 사진에서 **결핵을 자동으로 찾고 분석하는 AI 엔진**이 담긴 핵심 공간입니다.
> 1.  **찾기 (Deep Learning)**: 인공지능이 병변 위치를 자동으로 표시합니다.
> 2.  **분석하기 (Radiomics)**: 표시된 부위의 거칠기, 모양 등 미세한 특징을 수치화합니다.
> 3.  **진단하기 (Classification)**: 분석된 데이터를 바탕으로 결핵 여부를 최종 판단하고 리포트를 작성합니다.

---

이 디렉토리는 흉부 X-ray(CXR) 이미지에서 결핵(TB) 병변을 **딥러닝으로 찾고(Segmentation)**, 해당 부위의 **영상 특징을 정량적으로 분석(Radiomics)** 하여 최종적으로 질병 유무를 **진단(Classification)** 하는 통합 AI 시스템입니다.

처음 접하는 분들을 위해 간단히 설명하면, 이 시스템은 의사 선생님이 X-ray를 보고 "여기가 의심되네요(Segmentation)"라고 표시한 뒤, 그 부위의 "거칠기나 모양(Radiomics)"을 정밀하게 분석해서 "이건 결핵일 확률이 90%입니다(Classification)"라고 리포트를 써주는 과정을 자동화한 것입니다.

---

## 📂 폴더 및 파일 구조 설명

이 `model` 폴더는 크게 **딥러닝 모델(U-Net)**, **데이터 처리**, **Radiomics 분석** 세 부분으로 나뉩니다.

```
model/
├── config.py             # [설정] 모든 하이퍼파라미터(이미지 크기, 학습률, 경로 등)를 관리하는 컨트롤 타워
├── README.md             # [문서] 현재 보고 계신 사용자 가이드
├── MODEL_ARCH.md         # [문서] 모델의 기술적 아키텍처와 작동 원리 상세 설명
├── WORKLOG.md            # [기록] 개발 작업 일지
├── src/                  # [소스코드] 실제 실행 가능한 파이썬 스크립트들이 모여 있는 곳
│   │  # --- 1. 딥러닝 (병변 찾기) ---
│   ├── model.py          # U-Net 모델 구조 정의 (뇌 역할)
│   ├── train.py          # 모델 학습 실행 스크립트 (훈련)
│   ├── infer.py          # 이미지 1장에 대해 병변 위치 추론 (진료)
│   ├── batch_infer.py    # 이미지 여러 장을 한꺼번에 추론
│   ├── evaluate.py       # 모델 성능 평가 (시험)
│   ├── losses.py         # 학습 오차 계산 함수 (채점 기준)
│   │
│   │  # --- 2. 데이터 처리 ---
│   ├── dataset.py        # 이미지 로딩, 전처리, 증강 (데이터 준비)
│   ├── model_loader.py   # 저장된 모델(.h5)을 안전하게 불러오는 유틸리티
│   │
│   │  # --- 3. Radiomics (특징 분석 & 진단) ---
│   ├── radiomics_extract.py   # 병변 부위에서 수치적 특징(질감, 모양 등) 추출
│   ├── radiomics_classify.py  # 추출된 특징으로 결핵/정상 분류 (머신러닝)
│   └── radiomics_report.py    # 분류 결과를 JSON 리포트로 저장
│
├── outputs/              # [결과물] 학습된 모델과 예측 결과가 저장되는 곳
│   ├── checkpoints/      # 학습된 모델 가중치(.h5) 저장소
│   ├── logs/             # 학습 진행 상황 기록 (TensorBoard)
│   ├── predictions/      # 모델이 예측한 병변 마스크 이미지
│   └── radiomics/        # 추출된 특징 CSV 및 분석 리포트 JSON
│
└── data/ (외부)           # 학습에 사용할 CSV와 이미지들이 위치해야 함
```

---

## 🚀 워크플로우 (초보자 가이드)

이 시스템을 처음부터 끝까지 돌려보려면 아래 순서를 따라가세요.

### 0. 환경 설정
필요한 파이썬 라이브러리를 설치합니다.
```bash
# 가상환경 권장
pip install "tensorflow==2.15.*" "tensorflow-addons==0.22.*" pandas matplotlib pillow pyradiomics SimpleITK scikit-learn
```

### 1. 데이터 준비
`data/dataset.csv` 파일이 필요합니다. 이 파일은 학습시킬 이미지 경로(`image_path`)와 정답 마스크 경로(`mask_path`)를 담고 있어야 합니다.

### 2. 병변 탐지 모델 학습 (Deep Learning)
U-Net 모델을 학습시켜서 X-ray에서 병변이 어디에 있는지 찾는 능력을 기릅니다.
```bash
python src/train.py
```
*   결과물: `outputs/checkpoints/best_model.h5` (가장 똑똑해진 모델 파일)

### 3. 병변 위치 예측 (Inference)
학습된 모델을 이용해 새로운 이미지들의 병변 위치(마스크)를 예측하고 저장합니다. Radiomics 분석을 위해 필요합니다.
```bash
# 학습된 모델로 모든 데이터의 마스크 생성
python src/batch_infer.py \
  --model_path outputs/checkpoints/best_model.h5 \
  --csv_path data/dataset.csv \
  --output_dir outputs/predictions/batch \
  --count -1 \
  --overwrite
```
*   결과물: `outputs/predictions/batch/masks/` 폴더에 마스크 이미지 생성

### 4. Radiomics 특징 추출 (Feature Extraction)
예측된 병변 위치(마스크) 안쪽의 이미지 패턴(질감, 밝기 분포 등)을 수치 데이터로 변환합니다.
```bash
python src/radiomics_extract.py \
  --csv_path data/dataset.csv \
  --mask_source pred \
  --pred_mask_dir outputs/predictions/batch/masks \
  --output_csv outputs/radiomics/features.csv
```
*   결과물: `outputs/radiomics/features.csv` (이미지 하나당 수백 개의 숫자가 적힌 엑셀 표 같은 파일)

### 5. 최종 진단 리포트 생성 (Classification & Reporting)
추출된 특징 데이터를 바탕으로 머신러닝(Random Forest 등)을 돌려 결핵 여부를 최종 판정하고 리포트를 씁니다.
```bash
python src/radiomics_report.py \
  --features_csv outputs/radiomics/features.csv \
  --output_json outputs/radiomics/classify_report.json
```
*   결과물: `outputs/radiomics/classify_report.json` (웹사이트나 앱에서 보여줄 최종 분석 결과)

---

## 🔬 주요 기능 상세

### A. Segmentation (U-Net)
*   **역할**: 이미지에서 "어디가 아픈지" 픽셀 단위로 색칠합니다.
*   **특징**: U자 형태의 신경망 구조를 사용하여, 이미지의 전체적인 맥락(Context)과 세밀한 위치 정보(Localization)를 동시에 잡아냅니다.

### B. Radiomics (PyRadiomics)
*   **역할**: "어떻게 아픈지"를 숫자로 표현합니다.
*   **추출 정보**:
    *   **Shape**: 병변의 크기, 둘레, 찌그러짐 정도.
    *   **First Order**: 병변 부위의 평균 밝기, 밝기 치우침.
    *   **Texture (GLCM, GLRLM 등)**: 병변 내부가 균일한지, 거친지, 규칙적인 패턴이 있는지 등 의사의 눈으로도 보기 힘든 미세 패턴을 분석합니다.

### C. Diagnosis (Machine Learning)
*   **역할**: 위에서 구한 숫자들을 종합해서 "그래서 결핵인가요?"에 답합니다.
*   **알고리즘**: Random Forest와 Logistic Regression 같은 전통적이고 설명 가능한 머신러닝 모델을 사용하여 과적합을 방지하고 분석의 근거를(Feature Importance) 제공합니다.
