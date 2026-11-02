"""
radiomics_llm_report_raw.py  (1차 LLM 엔진)

역할:
- Radiomics Raw Data + ML 결과 + TB/Radiomics 지식베이스 + 모델 메타정보를 입력으로 받고
- 오직 "임상의·연구자 참고용 structured JSON 리포트"만 생성한다.

특징:
- 환자용 설명은 절대 포함하지 않는다. (2차 LLM에서 환자와 대화를 담당)
- 대화체 사용 금지.
- Chatbot 용 문장을 작성하지 않는다.
- 분석 중심, 의학적 근거 중심의 요약만 생성한다.

사용 예시:
from radiomics_llm_report_raw import generate_radiomics_report
report = generate_radiomics_report(case_json)
"""

import os
import json
from typing import Dict, Any, List
from pathlib import Path
from openai import OpenAI

# ======================================
# 0. OpenAI Client
# ======================================

def _create_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다."
        )
    return OpenAI(api_key=api_key)

client = _create_client()


# ======================================
# 1. Model Meta Information (항상 포함되는 전역 메타)
#    사용자가 제공한 JSON을 그대로 사용
# ======================================

MODEL_META: Dict[str, Any] = {
    "dataset": {
        "file": "outputs/radiomics/features_with_synthetic_normals.csv",
        "samples": 430,
        "features": 102,
        "tb": 330,
        "normal": 100,
        "note": "Normal 100 samples are synthetic; analysis only, not clinical."
    },
    "logreg": {
        "roc_auc": 1.0,
        "report": {
            "0": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 20.0
            },
            "1": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 66.0
            },
            "accuracy": 1.0,
            "macro avg": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 86.0
            },
            "weighted avg": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 86.0
            }
        }
    },
    "random_forest": {
        "roc_auc": 1.0,
        "report": {
            "0": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 20.0
            },
            "1": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 66.0
            },
            "accuracy": 1.0,
            "macro avg": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 86.0
            },
            "weighted avg": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 86.0
            }
        },
        "feature_importance_top10": [
            {
                "feature": "original_glcm_Imc1",
                "importance": 0.22146515262089497
            },
            {
                "feature": "original_firstorder_Maximum",
                "importance": 0.09920766439252111
            },
            {
                "feature": "original_firstorder_Range",
                "importance": 0.09046836293554351
            },
            {
                "feature": "original_glcm_Imc2",
                "importance": 0.09046461237081396
            },
            {
                "feature": "original_glcm_Idmn",
                "importance": 0.08295208563768676
            },
            {
                "feature": "original_glcm_Idn",
                "importance": 0.0776288087458565
            },
            {
                "feature": "original_glszm_ZoneEntropy",
                "importance": 0.04313120124481508
            },
            {
                "feature": "original_glcm_MCC",
                "importance": 0.036601263782084933
            },
            {
                "feature": "original_firstorder_Minimum",
                "importance": 0.025533780704809675
            },
            {
                "feature": "original_glszm_SmallAreaEmphasis",
                "importance": 0.017861450200795617
            }
        ],
        "group_importance": {
            "firstorder": 0.2396907092672334,
            "glcm": 0.5722353732746164,
            "glrlm": 0.042304072164474696,
            "glszm": 0.07950720366207635,
            "gldm": 0.030843772974466065,
            "ngtdm": 0.017670140370472022,
            "shape2D": 0.017748728286661283
        }
    },
    "disclaimer": [
        "Synthetic normals used; external validation required.",
        "Not for clinical decision-making."
    ]
}


# ======================================
# 2. TB/Radiomics Knowledge Base 로딩
# ======================================

def load_kb_text() -> str:
    base_dir = Path(__file__).parent
    parts: List[str] = []
    for i in range(1, 9):
        p = base_dir / "documents" / f"tb_{i}.txt"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                parts.append(f.read())
    kb = "\n\n".join(parts).strip()
    return kb if kb else "[Warning] TB KB missing."

KB_TEXT = load_kb_text()


# ======================================
# 3. SYSTEM PROMPT (RAW DATA 해석 전용)
# ======================================

SYSTEM_PROMPT = """
You are a medical AI assistant whose sole role is to analyze radiomics feature sets,
machine-learning outputs, TB-related radiomics patterns, and model metadata.

You DO NOT talk to patients.
You DO NOT create conversational explanations.
You DO NOT create emotional or human-facing language.

Your sole purpose:
- Interpret radiomics features (shape2D, firstorder, GLCM, GLRLM, GLSZM, GLDM, NGTD-M).
- Interpret ML predictions (probabilities, risk score, label).
- Use TB/radiomics knowledge base and model metadata, including:
  - dataset configuration
  - internal performance (ROC-AUC, classification report)
  - feature_importance_top10
  - group_importance by feature families
- Produce a compact JSON report for clinicians and researchers.

REQUIRED JSON STRUCTURE
(MUST return exactly one JSON object with at least the following keys):

{
  "clinical_overview": "short, technical summary of overall findings",

  "tb_risk_summary": {
    "level": "low | moderate | high",
    "rationale": "technical explanation for the assigned TB risk level"
  },

  "lesion_summary": {
    "num_lesions": 1,
    "lesion_regions": ["RUL", "LLL"]
  },

  "lesions": [
    {
      "lesion_id": 1,
      "location": "Right Upper Lobe",
      "size_mm": {
        "major_axis_length": 0,
        "minor_axis_length": 0,
        "area_mm2": 0
      },
      "shape_features": {
        "elongation": 0,
        "roundness": 0,
        "eccentricity": 0,
        "compactness": 0
      },
      "intensity_features": {
        "mean_intensity": 0,
        "median_intensity": 0,
        "variance": 0,
        "range": 0,
        "entropy": 0
      },
      "texture_features": {
        "glcm_correlation": 0,
        "glcm_contrast": 0,
        "glcm_dissimilarity": 0,
        "glcm_homogeneity": 0,
        "glrlm_long_run_emphasis": 0,
        "glrlm_short_run_emphasis": 0,
        "glszm_zone_variance": 0
      },
      "radiomics_risk_score": {
        "tb_risk": 0.0,
        "malignancy_risk": 0.0,
        "inflammation_score": 0.0
      }
    }
  ],

  "model_outputs": {
    "tb_probability": 0.0,
    "pneumonia_probability": 0.0,
    "nodule_probability": 0.0
  },

  "ml_summary": {
    "primary_label": "string",
    "primary_probability": 0.0,
    "risk_score": 0.0,
    "model_confidence": "low | moderate | high",
    "reliability_comment": "technical summary of model reliability and limitations"
  },

  "recommendations_for_clinician": [
    "technical, clinician-facing recommendations only, not patient-facing"
  ],

  "supporting_evidence": [
    {
      "json_key": "path.to.field",
      "category": "radiomics | ml | meta",
      "role": "tb_risk | morphology | texture | reliability",
      "explanation": "why this field is important for interpretation"
    }
  ],

  "limitations": {
    "data_limitations": ["string"],
    "model_limitations": ["string"],
    "usage_restrictions": "string"
  }
}

Rules:
- NO patient-friendly wording.
- NO conversational or chatty tone.
- NO reassurance statements.
- NO emotional expressions.
- Summaries must be objective and data-driven.

- Radiomics / lesion interpretation:
  - Describe shape, size, homogeneity, heterogeneity, and texture.
  - If exact millimeter values are not available, approximate from radiomics shape features
    or set numeric fields to null or 0 and describe this uncertainty in text.

- TB risk interpretation:
  - Use radiomics patterns + ML outputs + TB knowledge to assign tb_risk_summary.level.
  - tb_risk_summary.rationale must mention the key radiomics/ML drivers.

- Model outputs:
  - Map from ml_results in the case JSON where possible.
  - If probabilities for pneumonia or nodule are not available, set them to null.

- ML interpretation (ml_summary):
  - Use label, primary_probability, risk_score (if provided).
  - Reflect the limitation that synthetic normal data were used.
  - Account for the fact that reported metrics are from internal validation only.

- Limitations / disclaimer:
  - MUST mention at least:
    - synthetic normals in training,
    - research-only / analysis-only use,
    - external validation required,
    - not for direct clinical decision-making.

Output:
- Return ONLY a single JSON object with the required keys.
- Do NOT include any extra commentary or formatting.
"""


# ======================================
# 4. USER PROMPT Builder
# ======================================

def build_user_prompt(
    case_json: Dict[str, Any],
    kb_text: str = KB_TEXT,
    meta: Dict[str, Any] = MODEL_META
) -> str:
    """
    case_json: 1개 케이스에 대한 radiomics + ML 결과 (image / radiomics_features / ml_results)
    meta: 전역 모델 메타정보 (dataset, 성능, feature_importance, disclaimer 등)
    """

    return f"""
[MODEL_METADATA]
{json.dumps(meta, ensure_ascii=False, indent=2)}
[END_OF_MODEL_METADATA]

[TB_RADIO_KNOWLEDGE_BASE]
{kb_text}
[END_OF_TB_RADIO_KNOWLEDGE_BASE]

[RADIOMICS_CASE_JSON]
{json.dumps(case_json, ensure_ascii=False, indent=2)}
[END_OF_RADIOMICS_CASE_JSON]

Instructions:
- Interpret the full set of RAW radiomics values and ML outputs with a primary focus on TB-related morphologic, textural, and intensity patterns.
- Integrate the model’s top important radiomics features (feature_importance_top10) into a unified, high-level clinical advisory section that summarizes how these features collectively support or weaken the likelihood of TB involvement.
- Use MODEL_METADATA (dataset information, feature_importance_top10, group_importance, disclaimer)
  only to contextualize general limitations. Do NOT output or list metadata values directly.
- Do NOT mention model performance, internal validation metrics, or any evaluation scores.
- Do NOT mention synthetic normal controls or any synthetic data used during development.
- Analyze structure, shape descriptors, texture matrices, and intensity distribution to identify signatures commonly associated with TB, such as cavity-like patterns, consolidative morphology, nodularity, or heterogeneous texture.
- Do NOT generate patient-oriented language or conversational explanations.
- Do NOT describe what radiomics is.
- Output only a single JSON object that follows the required schema, with no additional commentary.


"""


# ======================================
# 5. LLM Output JSON Parse Helper
# ======================================

def _parse_llm_json(msg) -> Dict[str, Any]:
    """
    - JSON 모드일 경우 message.parsed 우선 사용
    - 아닐 경우 content를 json.loads로 파싱
    - 실패 시 마지막 '}'까지 잘라 재시도 후, 그래도 안 되면 에러
    """
    parsed = getattr(msg, "parsed", None)
    if parsed is not None:
        return parsed

    raw = (msg.content or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    last = raw.rfind("}")
    if last != -1:
        candidate = raw[: last + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("Invalid JSON from LLM:\n" + raw)


# ======================================
# 6. MAIN FUNCTION — RAW → Structured Report
# ======================================

def generate_radiomics_report(
    case_json: Dict[str, Any],
    model: str = "gpt-4.1",
    temperature: float = 0.1,
    max_tokens: int = 2000
) -> Dict[str, Any]:
    """
    단일 케이스 radiomics JSON을 입력으로 받아
    1차 LLM 기반 Radiomics 리포트를 생성하여 dict로 반환한다.
    (환자용 설명 X, 의사용/연구용 분석 리포트만)
    """

    if not isinstance(case_json, dict):
        raise TypeError("case_json must be dict.")

    prompt = build_user_prompt(case_json)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    msg = completion.choices[0].message
    parsed = _parse_llm_json(msg)

    if not isinstance(parsed, dict):
        raise ValueError("1st-stage LLM output is not a JSON object")

    # ------- 최소 정규화 (supporting_evidence 보정) -------

    report: Dict[str, Any] = dict(parsed)  # shallow copy

    # supporting_evidence: 문자열/딕셔너리 혼합 가능성 보정
    evid = report.get("supporting_evidence", [])
    if isinstance(evid, str):
        report["supporting_evidence"] = [{
            "json_key": "unknown",
            "category": "radiomics",
            "role": "morphology",
            "explanation": evid.strip()
        }]
    elif isinstance(evid, list):
        normalized: List[Dict[str, Any]] = []
        for item in evid:
            if isinstance(item, str):
                normalized.append({
                    "json_key": "unknown",
                    "category": "radiomics",
                    "role": "morphology",
                    "explanation": item.strip()
                })
            elif isinstance(item, dict):
                normalized.append({
                    "json_key": item.get("json_key", "unknown"),
                    "category": item.get("category", "radiomics"),
                    "role": item.get("role", "morphology"),
                    "explanation": str(
                        item.get("explanation", "")
                    ).strip()
                })
        report["supporting_evidence"] = normalized
    else:
        report["supporting_evidence"] = []

    # lesion_summary / lesions / model_outputs / tb_risk_summary 없을 경우 기본값 채우기
    report.setdefault("tb_risk_summary", {
        "level": "",
        "rationale": ""
    })

    report.setdefault("lesion_summary", {
        "num_lesions": None,
        "lesion_regions": []
    })

    report.setdefault("lesions", [])

    report.setdefault("model_outputs", {
        "tb_probability": None,
        "pneumonia_probability": None,
        "nodule_probability": None
    })

    report.setdefault("limitations", {
        "data_limitations": [],
        "model_limitations": [],
        "usage_restrictions": ""
    })

    return report


# ======================================
# 7. 단독 실행 테스트 (예시 케이스 → report 출력)
# ======================================

if __name__ == "__main__":
    # 예시용 raw radiomics 데이터 (CHNCXR_0619_1)
    raw_case = {
        "image_id": "CHNCXR_0619_1",
        "image_path": "/home/ubuntu1/project_root (2)/project_root/data/shenzhen_images/CHNCXR_0619_1.png",
        "mask_path": "/home/ubuntu1/project_root (2)/project_root/data/shenzhen_combined_masks/CHNCXR_0619_1.png",
        "original_shape2D_Elongation": 0.9398413092,
        "original_shape2D_MajorAxisLength": 2773.2533538268,
        "original_shape2D_MaximumDiameter": 3290.3515009798,
        "original_shape2D_MeshSurface": 5371890.0,
        "original_shape2D_MinorAxisLength": 2606.4180627492,
        "original_shape2D_Perimeter": 10639.0651506263,
        "original_shape2D_PerimeterSurfaceRatio": 0.0019805069,
        "original_shape2D_PixelSurface": 5371889.0,
        "original_shape2D_Sphericity": 0.772262629,
        "original_firstorder_10Percentile": 27.0,
        "original_firstorder_90Percentile": 213.0,
        "original_firstorder_Energy": 88261390337.0,
        "original_firstorder_Entropy": 3.2775691777,
        "original_firstorder_InterquartileRange": 93.0,
        "original_firstorder_Kurtosis": 2.2987307181,
        "original_firstorder_Maximum": 254.0,
        "original_firstorder_MeanAbsoluteDeviation": 53.8120472454,
        "original_firstorder_Mean": 109.8718683502,
        "original_firstorder_Median": 105.0,
        "original_firstorder_Minimum": 0.0,
        "original_firstorder_Range": 254.0,
        "original_firstorder_RobustMeanAbsoluteDeviation": 40.0043564751,
        "original_firstorder_RootMeanSquared": 128.1804721863,
        "original_firstorder_Skewness": 0.4108167932,
        "original_firstorder_TotalEnergy": 88261390337.0,
        "original_firstorder_Uniformity": 0.1115333056,
        "original_firstorder_Variance": 4358.405995141,
        "original_glcm_Autocorrelation": 30.0498450302,
        "original_glcm_ClusterProminence": 1532.5247495966,
        "original_glcm_ClusterShade": 51.3199505315,
        "original_glcm_ClusterTendency": 25.5833030703,
        "original_glcm_Contrast": 2.3865327467,
        "original_glcm_Correlation": 0.8293476654,
        "original_glcm_DifferenceAverage": 0.4923979855,
        "original_glcm_DifferenceEntropy": 1.0344682134,
        "original_glcm_DifferenceVariance": 2.143045164,
        "original_glcm_Id": 0.8449336252,
        "original_glcm_Idm": 0.8425621951,
        "original_glcm_Idmn": 0.987738577,
        "original_glcm_Idn": 0.9666859679,
        "original_glcm_Imc1": -0.6412494196,
        "original_glcm_Imc2": 0.9924619949,
        "original_glcm_InverseVariance": 0.2646613498,
        "original_glcm_JointAverage": 4.9244951031,
        "original_glcm_JointEnergy": 0.0648097984,
        "original_glcm_JointEntropy": 4.4532977856,
        "original_glcm_MCC": 0.9500535287,
        "original_glcm_MaximumProbability": 0.1290864896,
        "original_glcm_SumAverage": 9.8489902062,
        "original_glcm_SumEntropy": 4.0456065031,
        "original_glcm_SumSquares": 6.9924589543,
        "original_glrlm_GrayLevelNonUniformity": 152116.3868554872,
        "original_glrlm_GrayLevelNonUniformityNormalized": 0.0974297726,
        "original_glrlm_GrayLevelVariance": 8.2099533655,
        "original_glrlm_HighGrayLevelRunEmphasis": 38.4891892328,
        "original_glrlm_LongRunEmphasis": 91.8975177334,
        "original_glrlm_LongRunHighGrayLevelEmphasis": 1060.5239320464,
        "original_glrlm_LongRunLowGrayLevelEmphasis": 22.2582746252,
        "original_glrlm_LowGrayLevelRunEmphasis": 0.1429252509,
        "original_glrlm_RunEntropy": 6.1075550291,
        "original_glrlm_RunLengthNonUniformity": 409493.9944836092,
        "original_glrlm_RunLengthNonUniformityNormalized": 0.2611873911,
        "original_glrlm_RunPercentage": 0.2906171833,
        "original_glrlm_RunVariance": 79.9239505965,
        "original_glrlm_ShortRunEmphasis": 0.5200772644,
        "original_glrlm_ShortRunHighGrayLevelEmphasis": 21.592953662,
        "original_glrlm_ShortRunLowGrayLevelEmphasis": 0.0720317812,
        "original_glszm_GrayLevelNonUniformity": 20782.4310556597,
        "original_glszm_GrayLevelNonUniformityNormalized": 0.0959634987,
        "original_glszm_GrayLevelVariance": 10.2246495769,
        "original_glszm_HighGrayLevelZoneEmphasis": 47.8271658524,
        "original_glszm_LargeAreaEmphasis": 2271503.6151981382,
        "original_glszm_LargeAreaHighGrayLevelEmphasis": 51223927.28528024,
        "original_glszm_LargeAreaLowGrayLevelEmphasis": 236034.8047720733,
        "original_glszm_LowGrayLevelZoneEmphasis": 0.1271837888,
        "original_glszm_SizeZoneNonUniformity": 73134.7571733328,
        "original_glszm_SizeZoneNonUniformityNormalized": 0.3377019346,
        "original_glszm_SmallAreaEmphasis": 0.6015122806,
        "original_glszm_SmallAreaHighGrayLevelEmphasis": 28.979845292,
        "original_glszm_SmallAreaLowGrayLevelEmphasis": 0.0769890024,
        "original_glszm_ZoneEntropy": 5.8766540517,
        "original_glszm_ZonePercentage": 0.0403146826,
        "original_glszm_ZoneVariance": 2270888.3341897056,
        "original_gldm_DependenceEntropy": 6.0135470189,
        "original_gldm_DependenceNonUniformity": 876953.5243142589,
        "original_gldm_DependenceNonUniformityNormalized": 0.1632486308,
        "original_gldm_DependenceVariance": 4.9040031457,
        "original_gldm_GrayLevelNonUniformity": 599144.5372205196,
        "original_gldm_GrayLevelVariance": 6.9959130128,
        "original_gldm_HighGrayLevelEmphasis": 31.2274404776,
        "original_gldm_LargeDependenceEmphasis": 49.4604629768,
        "original_gldm_LargeDependenceHighGrayLevelEmphasis": 1338.3026793741,
        "original_gldm_LargeDependenceLowGrayLevelEmphasis": 7.8707291848,
        "original_gldm_LowGrayLevelEmphasis": 0.1532601446,
        "original_gldm_SmallDependenceEmphasis": 0.0579119243,
        "original_gldm_SmallDependenceHighGrayLevelEmphasis": 2.4062764746,
        "original_gldm_SmallDependenceLowGrayLevelEmphasis": 0.0079879403,
        "original_ngtdm_Busyness": 7324.0775051381,
        "original_ngtdm_Coarseness": 0.0000045939,
        "original_ngtdm_Complexity": 20.4820867748,
        "original_ngtdm_Contrast": 0.0602524373,
        "original_ngtdm_Strength": 0.0001569792
    }

    # LLM에 전달할 케이스 구조로 변환
    demo_case = {
        "image": {
            "id": raw_case.get("image_id"),
            "path": raw_case.get("image_path"),
            "mask_path": raw_case.get("mask_path"),
        },
        "radiomics_features": {
            k: v for k, v in raw_case.items() if k.startswith("original_")
        },
        "ml_results": {
            "primary_label": "tb_suspected",
            "primary_probability": 0.85,
            "risk_score": 0.80,
            "note": "CHNCXR_0619_1 radiomics 특징을 이용한 데모 케이스."
        }
    }

    print(">> Radiomics RAW → Report 생성 테스트 실행...\n")
    report = generate_radiomics_report(demo_case)

    # report 전체 출력
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n>> 완료.")
