# 개발자 노트 및 트러블슈팅 (Developer Notes & Troubleshooting)

> **💡 요약 (Summary)**  
> 이 문서는 개발자 및 유지보수 담당자를 위한 **기술 노트**입니다.
> *   **문제 해결**: RTX 5060 등 최신 GPU에서의 호환성 문제 및 해결법
> *   **내부 원리**: 이미지 포맷 변환, 임시 파일 처리 등 숨겨진 로직 설명
> *   **작업 이력**: 프로젝트 기능 추가 및 수정 내역

---

이 문서는 프로젝트 개발 과정에서 발견된 **기술적 이슈, 환경 설정의 특이사항, 내부 구현의 디테일**을 기록합니다. 일반 사용자가 아닌, 이 코드를 수정하거나 깊게 분석하려는 개발자를 위한 문서입니다.

---

## ⚠️ 중요: 하드웨어 및 환경 이슈

### 1. RTX 5060 및 최신 CUDA 호환성
*   **증상**: TensorFlow 2.x와 최신 NVIDIA 드라이버(RTX 5060) 환경에서 `ptxas` 관련 에러나 커널 호환성 문제가 간헐적으로 발생할 수 있습니다.
*   **해결책**: 학습(Training) 단계가 아닌 **추론(Inference) 및 Radiomics 추출 단계**에서는 GPU를 끄고 CPU 모드로 실행하는 것이 더 안정적일 수 있습니다.
    ```bash
    # CPU 모드로 강제 실행하는 법
    CUDA_VISIBLE_DEVICES="" python src/batch_infer.py ...
    ```

### 2. PyRadiomics 의존성
*   `pyradiomics`는 C++ 확장을 사용하므로 설치 시 OS에 맞는 빌드 도구가 필요할 수 있습니다.
*   **Tip**: 에러 발생 시 `conda`를 사용하여 설치하는 것이 `pip`보다 안정적입니다.
    ```bash
    conda install -c conda-forge pyradiomics
    ```

---

## 🛠 내부 구현 디테일 (Implementation Details)

### 1. 이미지 포맷 변환 (Radiomics Pipeline)
`src/radiomics_extract.py`는 내부적으로 다음과 같은 임시 변환 과정을 거칩니다.
*   **문제**: 의료 영상 표준(DICOM/NRRD)이 아닌 일반 PNG/JPG 이미지는 PyRadiomics가 직접 처리하기 어렵습니다.
*   **해결**:
    1. **Scalar 변환**: RGB 이미지가 들어오면 강제로 단일 채널(Grayscale)로 변환합니다.
    2. **임시 파일**: 메모리 상의 객체를 직접 넘기는 대신, 안전하게 `.nrrd` 포맷의 임시 파일을 생성하여 PyRadiomics에 전달하고 처리가 끝나면 삭제합니다.
    *   *디버깅 시 `finally` 블록의 삭제 코드를 주석 처리하면 임시 파일을 확인해볼 수 있습니다.*

### 2. 마스크 이진화 (Binarization)
*   학습된 U-Net이 뱉어내는 값은 0.0 ~ 1.0 사이의 실수(확률값)입니다.
*   Radiomics 분석을 위해서는 명확한 **영역(0 또는 1)**이 필요하므로, `Threshold = 0.5`를 기준으로 이진화(Binarization)를 수행합니다.

---

## 📂 데이터 관리 노트

### 데이터셋 구성
*   **Normal (정상)** 데이터가 부족한 경우, `src/dataset.py` 혹은 분석 단계에서 **Synthetic Normal (가상 정상 데이터)** 기법을 사용할 수 있도록 설계되어 있습니다.
*   현재 `outputs/radiomics/features.csv`에는 주로 TB(결핵) 양성 데이터가 포함되어 있으므로, 분류기 학습 시 클래스 불균형에 주의해야 합니다. (Synthetic Data 활용 권장)

---

## 📝 작업 로그 (Work Log)

*   **초기 구축**: U-Net 모델 구현 및 Shenzhen 데이터셋 연동.
*   **기능 확장**: `batch_infer.py` 추가 (대량 추론용).
*   **Radiomics 통합**: PyRadiomics 연동 및 특징 추출 파이프라인 완성.
*   **리포팅 시스템**: 머신러닝 기반 분류기 및 JSON 리포트 생성기(`radiomics_report.py`) 추가.
*   **문서화**: 초보자를 위한 가이드(README, MODEL_ARCH) 대폭 보강 완료.

---

## 2026-11 파이프라인 정리 (포트폴리오 버전)

*   **패키지화**: `sys.path` 조작을 제거하고 `cxr_tb` 로 설치 가능하게 변경 (`pip install -e ".[inference]"`). 모든 스크립트는 `python -m cxr_tb.src.<name>`.
*   **서빙/학습 공용 파이프라인** `src/pipeline.py` 추가. 전처리는 학습 시 `tf.image.resize`(bilinear, 안티앨리어싱 없음)와 동일하게 NumPy 로 재현 — PIL 리사이즈를 쓰면 마스크 IoU 가 떨어지는 것을 확인(0329_1 케이스에서 병변이 사라짐).
*   **빈 마스크 게이트**: 면적 0.1% 미만이면 `no_lesion_detected`. Radiomics 는 빈 ROI 에서 정의되지 않으므로 분류기를 돌리지 않는다.
*   **Synthetic normal 제거**: 분류기는 Shenzhen 실제 정상 영상에서 U-Net 이 잡은 위양성 영역 vs TB 영상의 병변 영역으로 학습·교차검증 (`train_classifier.py`). 종전 ROC-AUC 1.0 은 합성 정상군 때문이었음.
*   **PyRadiomics 반환값**: 일부 특징이 0-차원 `numpy.ndarray` 로 와서 `np.generic` 필터에 걸러지던 버그 수정 (102개 전부 추출되도록).
*   제거: `batch_infer.py`, `radiomics_classify.py`(중복), 합성 데이터 CSV 사본들.
