# 기술 아키텍처 및 데이터 파이프라인 명세서

> **💡 요약 (Summary)**  
> 이 문서는 AI 시스템이 **어떤 원리로 작동하는지** 보여주는 설계도입니다.
> *   **눈 (Segmentation)**: U-Net 모델(Deep Learning)을 사용해 X-ray에서 병변의 **위치**를 찾습니다.
> *   **두뇌 (Radiomics & ML)**: 병변의 모양과 질감을 수학적으로 정밀 **분석**하여 질병을 판단합니다.

---

이 문서는 `model` 디렉토리 내의 AI 시스템이 **어떤 원리로 작동하는지**, **데이터가 어떻게 흐르는지**를 엔지니어와 연구자를 위해 상세히 기술합니다.

---

## 1. 시스템 전체 구조도 (System Architecture)

이 시스템은 **Hybrid AI Approach**를 채택하고 있습니다. 
1. **Deep Learning (CNN)** 은 병변의 **위치(Localization)** 를 찾고,
2. **Machine Learning (Radiomics)** 은 그 위치의 **성질(Characterization)** 을 분석합니다.

```mermaid
graph TD
    Input[CXR Image (X-ray)] --> |Preprocessing| DL_Input[512x512 Tensor]
    
    subgraph "Phase 1: Segmentation (Deep Learning)"
        DL_Input --> Encoder[Encoder (Downsampling)]
        Encoder --> Bottleneck[Bottleneck Feature]
        Bottleneck --> Decoder[Decoder (Upsampling)]
        Encoder -.-> |Skip Connection| Decoder
        Decoder --> OutputHead[Sigmoid Activation]
    end
    
    OutputHead --> |Prediction| Mask[Binary Mask (Lesion Area)]
    
    subgraph "Phase 2: Radiomics Analysis"
        Input & Mask --> PyRadiomics[Feature Extractor]
        PyRadiomics --> Features[Quantitative Features\n(Shape, Texture, Intensity)]
    end
    
    subgraph "Phase 3: Clinical Diagnosis"
        Features --> Classifier[Random Forest Classifier]
        Classifier --> Diagnosis[Final TB Probability]
    end
```

---

## 2. Phase 1: Segmentation (U-Net)

병변의 위치를 찾는 딥러닝 모델은 **U-Net** 아키텍처를 사용합니다.

### 2.1 아키텍처 특징
*   **U-Shape 구조**: 
    *   **Encoder (수축 경로)**: 이미지를 점점 줄여가며 "무엇(What)"이 있는지를 압축 요약합니다. (Convolution + Max Pooling)
    *   **Decoder (확장 경로)**: 요약된 정보를 다시 원본 크기로 늘리며 "어디(Where)"에 있는지 복원합니다. (Up-Convolution)
    *   **Skip Connection**: 줄어들 때 잃어버렸던 디테일한 위치 정보를 Encoder에서 Decoder로 직접 전달해줍니다. 이게 없으면 경계선이 흐릿해집니다.
*   **Backbone**: Custom CNN (가볍고 빠른 학습을 위해 ResNet 등이 아닌 기본 CNN Block을 직접 쌓아 사용)
*   **Loss Function**: `BCE (Binary Cross Entropy) + Dice Loss`
    *   픽셀 하나하나의 정확도(BCE)와 덩어리 모양의 일치도(Dice)를 동시에 고려하여 학습합니다.

### 2.2 입력 및 출력
*   **Input**: `(Batch, 512, 512, 1)` - 흑백 흉부 X-ray 이미지.
*   **Output**: `(Batch, 512, 512, 1)` - 각 픽셀이 병변일 확률값 (0.0 ~ 1.0).

---

## 3. Phase 2: Radiomics (Feature Extraction)

딥러닝이 찾아낸 병변 부위(Mask) 내의 이미지 픽셀값을 통계적으로 분석합니다. **"이미지를 데이터로 변환(Images to Data)"** 하는 과정입니다.

### 3.1 추출하는 주요 특징 (Features)
의사가 눈으로 보는 "뿌옇다", "거칠다" 같은 주관적 느낌을 수학적 수치로 변환합니다.

| 분류 | 설명 | 예시 |
| :--- | :--- | :--- |
| **First Order** | 픽셀값(밝기)의 통계적 분포 | 평균 밝기, 최대값, 왜도(Skewness), 엔트로피 |
| **Shape (2D)** | 병변의 기하학적 형태 | 크기(MeshSurface), 둥근 정도(Sphericity), 둘레길이 |
| **GLCM** | 픽셀 쌍의 밝기 관계 (질감) | Contrast(대비), Correlation(상관관계), Homogeneity(균일함) |
| **GLRLM** | 같은 밝기가 연속되는 길이 | 거친 질감 vs 부드러운 질감 구분 |

### 3.2 전처리 (Preprocessing)
*   **Resampling**: 픽셀 간격을 통일하여 이미지 크기가 달라도 일관된 특징을 뽑습니다.
*   **Discretization**: 연속된 픽셀 밝기값을 일정한 구간(Bin)으로 나누어(예: 25단위) 노이즈를 줄이고 패턴을 단순화합니다.

---

## 4. Phase 3: Classification (Machine Learning)

추출된 수백 개의 Radiomics 특징(숫자)을 입력으로 받아 최종 진단을 내립니다.

### 4.1 모델 선택: Random Forest
딥러닝 대신 전통적인 머신러닝을 사용하는 이유는 **설명력(Explainability)** 때문입니다.
*   **Feature Importance**: 어떤 특징(예: "병변의 거칠기")이 결핵 판정에 결정적이었는지 알 수 있습니다.
*   **데이터 효율성**: 수백~수천 장 정도의 적은 의료 데이터에서는 거대 딥러닝 모델보다 Random Forest가 과적합(Overfitting) 없이 안정적인 성능을 냅니다.

### 4.2 학습 과정
1.  **Synthetic Normals**: 정상 데이터(병변 없음)와 결핵 데이터(병변 있음)의 비율을 맞추기 위해, 정상 이미지의 폐 영역에서 가상의 정상 패치를 추출하여 대조군으로 사용합니다.
2.  **Stratified Split**: 학습/테스트 데이터를 나눌 때 환자 비율을 일정하게 유지합니다.
3.  **Ensemble**: 여러 개의 결정 트리(Decision Tree) 투표를 통해 가장 합리적인 결과를 도출합니다.
