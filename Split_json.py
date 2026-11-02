"""
radiomics_llm_report.py

역할:
- Radiomics + 머신러닝 결과(JSON)와 TB/Radiomics 지식베이스를 입력으로 받아
- GPT 모델이 해석한 의사용 요약 / 환자용 설명 / 권고사항 / 근거(evidence) / 디스클레이머를 JSON으로 반환하는 모듈

사용 예시:
from radiomics_llm_report2 import generate_radiomics_report

report = generate_radiomics_report(case_json)
print(report)
"""

import os
import json
from typing import Dict, Any
from pathlib import Path

from openai import OpenAI

# ============================
# 0. OpenAI 클라이언트
# ============================

def _create_client() -> OpenAI:
    """
    OPENAI_API_KEY 환경변수가 없으면 바로 에러를 던져서
    왜 안 되는지 한 번에 알 수 있게 함.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.\n"
            "PowerShell 예시:  setx OPENAI_API_KEY \"실제_API_키\"\n"
            "새 터미널을 연 뒤에 다시 실행해주세요."
        )
    return OpenAI(api_key=api_key)

client = _create_client()

# ============================
# 1. TB & Radiomics 지식베이스
# ============================

def load_kb_text() -> str:
    """
    같은 폴더에 있는 tb_1.txt ~ tb_8.txt를 순서대로 읽어서
    하나의 긴 텍스트로 합친다.
    """
    base_dir = Path(__file__).parent
    parts = []
    for i in range(1, 9):  # tb_1.txt ~ tb_8.txt
        path = base_dir / f"tb_{i}.txt"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                parts.append(f.read())

    kb_text = "\n\n".join(parts).strip()

    # 혹시 파일이 하나도 없을 때를 대비한 안전장치
    if not kb_text:
        kb_text = (
            "[주의] TB 지식베이스 파일(tb_1.txt ~ tb_8.txt)을 찾지 못했습니다. "
            "이 메시지는 디버깅용이며, 실제 서비스 시에는 해당 파일들을 반드시 준비해야 합니다."
        )
    return kb_text

KB_TEXT = load_kb_text()

# ============================
# 2. 시스템 프롬프트
# ============================

SYSTEM_PROMPT = """
You are an AI assistant that supports clinicians by interpreting machine-learning based
radiomics predictions from chest X-ray images, with a focus on tuberculosis (TB) and lung lesions.

Your role:
- Use the provided medical knowledge base AND the radiomics JSON together.
- Summarize likely diagnoses and overall risk level based on model outputs and radiomics context.
- Explain qualitatively how confident the model seems (low / moderate / high).
- Suggest reasonable next steps (e.g., CT, PET-CT, follow-up interval, referral).

Important constraints:
- You are a decision support tool, NOT a doctor.
- Never make definitive diagnoses or treatment decisions.
- Always include a disclaimer that this is a research prototype and must be reviewed by a physician.
- If the input is ambiguous or probabilities are similar, clearly say that uncertainty is high.
- Be conservative and safety-oriented in your recommendations.

Output must be compact and concise enough to fit within about 600–800 tokens total.
"""

# ============================
# 3. 유저 프롬프트 생성 함수
# ============================

def build_user_prompt(case_json: Dict[str, Any], kb_text: str = KB_TEXT) -> str:
    """
    Radiomics + ML 결과(JSON)와 TB/Radiomics 지식베이스 텍스트를 받아
    LLM에 줄 user 프롬프트 문자열을 구성한다.

    기대하는 case_json 구조:

    {
      "image": {
        "id": "CHNCXR_0619_1",
        "path": "...png",
        "mask_path": "...png"
      },
      "radiomics_features": {
        "original_shape2D_Elongation": ...,
        "original_firstorder_Mean": ...,
        "original_glcm_Contrast": ...,
        ...
      },
      "ml_results": {
        "primary_label": "tb_suspected" | "normal_like" | ...,
        "primary_probability": 0.0~1.0,
        "risk_score": 0.0~1.0,
        "note": "옵션"
      }
    }
    """

    return f"""
다음은 Radiomics 기반 머신러닝 모델이 흉부 X-ray에서 추출한 결과(JSON)와,
결핵(TB) 및 폐결절, Radiomics 관련 의학 지식베이스입니다.

[MEDICAL_KNOWLEDGE_BASE]
{kb_text}
[END_OF_KNOWLEDGE_BASE]

아래 JSON은 한 명의 케이스에 대한 정보입니다.
- image: 영상 파일 경로 및 마스크 정보
- radiomics_features: shape2D / firstorder / GLCM / GLRLM / GLSZM / GLDM / NGTD-M 등 Radiomics 수치
- ml_results: 모델이 예측한 주요 라벨, 확률, 위험도 점수 등

[RADIOMICS_CASE_JSON]
{json.dumps(case_json, ensure_ascii=False, indent=2)}
[END_OF_RADIOMICS_CASE_JSON]

주의:
- radiomics_features에 있는 수치들은 '절대값' 기준은 없지만,
  서로 상대적 크기, 그룹별 특징(예: shape2D=크기/형태, firstorder=전체 밝기/분포, glcm=질감, glszm=병변 크기 분포 등)을
  이용해 영상 소견을 '정성적으로' 설명하는 데 사용하세요.
- 숫자 값을 그대로 나열하지 말고, "병변 크기가 비교적 크다/중간이다/작다",
  "음영 분포가 비교적 균일하다/불균일하다", "질감이 거칠다/부드럽다" 같은 식으로 설명하세요.

요구사항:
1. doctor_summary:
   - 임상의(의사)를 대상으로 하는 전문 요약입니다.
   - radiomics_features와 ml_results를 모두 참고하여,
     가장 가능성이 높은 진단/감별진단과 TB 여부, 전반적 위험도(high/moderate/low)를 서술하세요.
   - shape2D(크기/형태), firstorder(평균 밝기/분포), glcm/rlm/szm 등의 질감 정보를
     적어도 1~2번은 근거로 언급하세요.
   - Fleischner guideline 또는 TB guideline과 연결되는 내용이 있으면 간단히 언급하세요.

2. patient_friendly:
   - 환자 또는 비의료인이 이해하기 쉬운 설명입니다.
   - radiomics 수치를 직접 언급하지 말고,
     "영상의 밝기와 선명도, 병변의 크기와 모양을 수치로 분석했을 때 ~" 와 같이
     풀어서 설명하세요.
   - 현재 결과가 의미하는 바(정상에 가깝다 / 결핵 가능성이 있다 / 다른 질환 가능성 등)와
     왜 추가 검사가 필요할 수 있는지 차분하게 설명하세요.
   - 과도한 공포를 유발하지 않되, 필요한 경각심은 유지하세요.

3. risk_level:
   - overall clinical risk level을 "low", "moderate", "high" 중 하나로 선택하세요.
   - Radiomics 예측 확률, 위험도 점수, radiomics 특징(크기/모양/질감)을 종합적으로 고려합니다.

4. recommendations:
   - 2~4개의 구체적인 권고사항을 리스트로 제시합니다.
   - 예: "6개월 후 저선량 CT 추적 검사 권장", "호흡기내과 전문의 진료 의뢰" 등
   - TB / 폐결절 관련 가이드라인과 지식베이스를 참고하여,
     현실적인 검사/추적 전략을 제안하세요.

5. evidence:
   - doctor_summary, risk_level, recommendations를 작성할 때
     radiomics JSON과 ml_results에서 어떤 항목을 근거로 사용했는지 명시하세요.
   - 각 근거는 아래 필드를 가진 객체(object)로 작성합니다.
     - json_key: 사용한 JSON 경로 (예: "ml_results.primary_probability",
                                  "radiomics_features.original_firstorder_Entropy")
     - explanation_for_doctor: 의사/전문가를 위한 근거 설명
       (예: "primary_probability=0.85로 TB_suspected에 대한 high risk 범위")
     - explanation_for_patient: 비의료인이 이해하기 쉬운 설명
       (예: "인공지능이 여러 수치를 종합해서 볼 때 결핵일 가능성이 높은 편입니다.")
   - 2~5개 정도의 핵심 근거만 간결하게 작성하세요.

6. disclaimer:
   - 이 결과는 Radiomics 기반 AI 모델과 제한된 지식베이스를 바탕으로 한
     연구용/보조용 도구에 불과하며, 최종 진단과 치료 결정은 반드시
     담당 의사(전문의)가 내려야 한다는 점을 명시하세요.
   - Radiomics 모델이 synthetic normal 데이터를 포함한 연구용 모델임을
     한 줄 정도로 언급해 주세요.

응답은 **반드시 아래 JSON 형식으로만** 작성하세요.
여분의 설명 문장, 마크다운, 키 설명 등을 추가하지 마세요.

응답 형식(JSON):

{{
  "doctor_summary": "string",
  "patient_friendly": "string",
  "risk_level": "low | moderate | high",
  "recommendations": ["string", "string", "..."],
  "evidence": [
    {{
      "json_key": "string",
      "explanation_for_doctor": "string",
      "explanation_for_patient": "string"
    }},
    {{
      "json_key": "string",
      "explanation_for_doctor": "string",
      "explanation_for_patient": "string"
    }}
  ],
  "disclaimer": "string"
}}
"""

# ============================
# 4. LLM JSON 파싱 헬퍼
# ============================

def _parse_llm_json(msg) -> Dict[str, Any]:
    """
    - 우선 message.parsed가 있으면 그걸 사용
    - 없으면 content를 json.loads
    - JSONDecodeError가 나면, 마지막 '}' 기준으로 잘라서 다시 시도
    """
    # 1) 최신 openai SDK에서 JSON 모드일 때
    parsed = getattr(msg, "parsed", None)
    if parsed is not None:
        return parsed

    # 2) content 기반 수동 파싱
    raw = (msg.content or "").strip()

    # 먼저 그대로 시도
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 3) 응답이 중간에 잘린 경우: 마지막 '}'까지 자르고 다시 시도
    last_brace = raw.rfind("}")
    if last_brace != -1:
        candidate = raw[: last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 4) 여기까지 안 되면 진짜로 실패
    raise ValueError("LLM output is not valid JSON:\n" + raw)

# ============================
# 5. 메인 함수: JSON -> LLM -> 리포트
# ============================

def generate_radiomics_report(
    case_json: Dict[str, Any],
    model: str = "gpt-4.1-mini",
    temperature: float = 0.2,
    max_tokens: int = 1200,  # 조금 여유 있게
) -> Dict[str, Any]:
    if not isinstance(case_json, dict):
        raise TypeError("case_json은 dict 형태여야 합니다.")

    user_prompt = build_user_prompt(case_json, KB_TEXT)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    msg = completion.choices[0].message
    parsed = _parse_llm_json(msg)

    result: Dict[str, Any] = {
        "doctor_summary": parsed.get("doctor_summary", "").strip(),
        "patient_friendly": parsed.get("patient_friendly", "").strip(),
        "risk_level": parsed.get("risk_level", "moderate").strip(),
        "recommendations": parsed.get("recommendations", []),
        "evidence": parsed.get("evidence", []),
        "disclaimer": parsed.get("disclaimer", "").strip(),
    }

    # recommendations가 string이면 리스트로 감싸기
    if isinstance(result["recommendations"], str):
        result["recommendations"] = [result["recommendations"]]

    # evidence 정규화
    if isinstance(result["evidence"], str):
        result["evidence"] = [{
            "json_key": "unknown",
            "explanation_for_doctor": result["evidence"],
            "explanation_for_patient": result["evidence"],
        }]
    elif isinstance(result["evidence"], list):
        normalized = []
        for item in result["evidence"]:
            if isinstance(item, str):
                normalized.append({
                    "json_key": "unknown",
                    "explanation_for_doctor": item,
                    "explanation_for_patient": item,
                })
            elif isinstance(item, dict):
                normalized.append({
                    "json_key": item.get("json_key", "unknown"),
                    "explanation_for_doctor": item.get("explanation_for_doctor", "").strip(),
                    "explanation_for_patient": item.get("explanation_for_patient", "").strip(),
                })
        result["evidence"] = normalized
    else:
        result["evidence"] = []

    return result

# ============================
# 6. 단독 실행 테스트용 (CHNCXR_0619_1 예시)
# ============================

if __name__ == "__main__":
    # 1) 네가 가진 raw radiomics JSON (CHNCXR_0619_1)
    raw_case = {
        "image_id":"CHNCXR_0619_1",
        "image_path":"/home/ubuntu1/project_root (2)/project_root/data/shenzhen_images/CHNCXR_0619_1.png",
        "mask_path":"/home/ubuntu1/project_root (2)/project_root/data/shenzhen_combined_masks/CHNCXR_0619_1.png",
        "original_shape2D_Elongation":0.9398413092,
        "original_shape2D_MajorAxisLength":2773.2533538268,
        "original_shape2D_MaximumDiameter":3290.3515009798,
        "original_shape2D_MeshSurface":5371890.0,
        "original_shape2D_MinorAxisLength":2606.4180627492,
        "original_shape2D_Perimeter":10639.0651506263,
        "original_shape2D_PerimeterSurfaceRatio":0.0019805069,
        "original_shape2D_PixelSurface":5371889.0,
        "original_shape2D_Sphericity":0.772262629,
        "original_firstorder_10Percentile":27.0,
        "original_firstorder_90Percentile":213.0,
        "original_firstorder_Energy":88261390337.0,
        "original_firstorder_Entropy":3.2775691777,
        "original_firstorder_InterquartileRange":93.0,
        "original_firstorder_Kurtosis":2.2987307181,
        "original_firstorder_Maximum":254.0,
        "original_firstorder_MeanAbsoluteDeviation":53.8120472454,
        "original_firstorder_Mean":109.8718683502,
        "original_firstorder_Median":105.0,
        "original_firstorder_Minimum":0.0,
        "original_firstorder_Range":254.0,
        "original_firstorder_RobustMeanAbsoluteDeviation":40.0043564751,
        "original_firstorder_RootMeanSquared":128.1804721863,
        "original_firstorder_Skewness":0.4108167932,
        "original_firstorder_TotalEnergy":88261390337.0,
        "original_firstorder_Uniformity":0.1115333056,
        "original_firstorder_Variance":4358.405995141,
        "original_glcm_Autocorrelation":30.0498450302,
        "original_glcm_ClusterProminence":1532.5247495966,
        "original_glcm_ClusterShade":51.3199505315,
        "original_glcm_ClusterTendency":25.5833030703,
        "original_glcm_Contrast":2.3865327467,
        "original_glcm_Correlation":0.8293476654,
        "original_glcm_DifferenceAverage":0.4923979855,
        "original_glcm_DifferenceEntropy":1.0344682134,
        "original_glcm_DifferenceVariance":2.143045164,
        "original_glcm_Id":0.8449336252,
        "original_glcm_Idm":0.8425621951,
        "original_glcm_Idmn":0.987738577,
        "original_glcm_Idn":0.9666859679,
        "original_glcm_Imc1":-0.6412494196,
        "original_glcm_Imc2":0.9924619949,
        "original_glcm_InverseVariance":0.2646613498,
        "original_glcm_JointAverage":4.9244951031,
        "original_glcm_JointEnergy":0.0648097984,
        "original_glcm_JointEntropy":4.4532977856,
        "original_glcm_MCC":0.9500535287,
        "original_glcm_MaximumProbability":0.1290864896,
        "original_glcm_SumAverage":9.8489902062,
        "original_glcm_SumEntropy":4.0456065031,
        "original_glcm_SumSquares":6.9924589543,
        "original_glrlm_GrayLevelNonUniformity":152116.3868554872,
        "original_glrlm_GrayLevelNonUniformityNormalized":0.0974297726,
        "original_glrlm_GrayLevelVariance":8.2099533655,
        "original_glrlm_HighGrayLevelRunEmphasis":38.4891892328,
        "original_glrlm_LongRunEmphasis":91.8975177334,
        "original_glrlm_LongRunHighGrayLevelEmphasis":1060.5239320464,
        "original_glrlm_LongRunLowGrayLevelEmphasis":22.2582746252,
        "original_glrlm_LowGrayLevelRunEmphasis":0.1429252509,
        "original_glrlm_RunEntropy":6.1075550291,
        "original_glrlm_RunLengthNonUniformity":409493.9944836092,
        "original_glrlm_RunLengthNonUniformityNormalized":0.2611873911,
        "original_glrlm_RunPercentage":0.2906171833,
        "original_glrlm_RunVariance":79.9239505965,
        "original_glrlm_ShortRunEmphasis":0.5200772644,
        "original_glrlm_ShortRunHighGrayLevelEmphasis":21.592953662,
        "original_glrlm_ShortRunLowGrayLevelEmphasis":0.0720317812,
        "original_glszm_GrayLevelNonUniformity":20782.4310556597,
        "original_glszm_GrayLevelNonUniformityNormalized":0.0959634987,
        "original_glszm_GrayLevelVariance":10.2246495769,
        "original_glszm_HighGrayLevelZoneEmphasis":47.8271658524,
        "original_glszm_LargeAreaEmphasis":2271503.6151981382,
        "original_glszm_LargeAreaHighGrayLevelEmphasis":51223927.2852802426,
        "original_glszm_LargeAreaLowGrayLevelEmphasis":236034.8047720733,
        "original_glszm_LowGrayLevelZoneEmphasis":0.1271837888,
        "original_glszm_SizeZoneNonUniformity":73134.7571733328,
        "original_glszm_SizeZoneNonUniformityNormalized":0.3377019346,
        "original_glszm_SmallAreaEmphasis":0.6015122806,
        "original_glszm_SmallAreaHighGrayLevelEmphasis":28.979845292,
        "original_glszm_SmallAreaLowGrayLevelEmphasis":0.0769890024,
        "original_glszm_ZoneEntropy":5.8766540517,
        "original_glszm_ZonePercentage":0.0403146826,
        "original_glszm_ZoneVariance":2270888.3341897056,
        "original_gldm_DependenceEntropy":6.0135470189,
        "original_gldm_DependenceNonUniformity":876953.5243142589,
        "original_gldm_DependenceNonUniformityNormalized":0.1632486308,
        "original_gldm_DependenceVariance":4.9040031457,
        "original_gldm_GrayLevelNonUniformity":599144.5372205196,
        "original_gldm_GrayLevelVariance":6.9959130128,
        "original_gldm_HighGrayLevelEmphasis":31.2274404776,
        "original_gldm_LargeDependenceEmphasis":49.4604629768,
        "original_gldm_LargeDependenceHighGrayLevelEmphasis":1338.3026793741,
        "original_gldm_LargeDependenceLowGrayLevelEmphasis":7.8707291848,
        "original_gldm_LowGrayLevelEmphasis":0.1532601446,
        "original_gldm_SmallDependenceEmphasis":0.0579119243,
        "original_gldm_SmallDependenceHighGrayLevelEmphasis":2.4062764746,
        "original_gldm_SmallDependenceLowGrayLevelEmphasis":0.0079879403,
        "original_ngtdm_Busyness":7324.0775051381,
        "original_ngtdm_Coarseness":0.0000045939,
        "original_ngtdm_Complexity":20.4820867748,
        "original_ngtdm_Contrast":0.0602524373,
        "original_ngtdm_Strength":0.0001569792
    }

    # 2) LLM에 전달할 구조로 변환
    demo_case = {
        "image": {
            "id": raw_case.get("image_id"),
            "path": raw_case.get("image_path"),
            "mask_path": raw_case.get("mask_path"),
        },
        "radiomics_features": {
            k: v for k, v in raw_case.items()
            if k.startswith("original_")
        },
        # 여기 ml_results는 예시용. 실제 파이프라인에서는 모델 출력으로 채우면 됨.
        "ml_results": {
            "primary_label": "tb_suspected",   # 예시 라벨
            "primary_probability": 0.85,       # 예시 확률
            "risk_score": 0.80,                # 예시 위험도 점수 (0~1)
            "note": "이 예시는 CHNCXR_0619_1 radiomics 특징을 이용한 데모 케이스입니다."
        }
    }

    print(">> Radiomics + TB LLM report demo 실행 중...\n")
    report = generate_radiomics_report(demo_case)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n>> 완료.")

    # ========== 각 key 별로 JSON 파일로 저장 ==========
    save_dir = Path("./output_jsons")
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "report_doctor_summary.json", "w", encoding="utf-8") as f:
        json.dump({"doctor_summary": report["doctor_summary"]}, f, ensure_ascii=False, indent=2)

    with open(save_dir / "report_patient_friendly.json", "w", encoding="utf-8") as f:
        json.dump({"patient_friendly": report["patient_friendly"]}, f, ensure_ascii=False, indent=2)

    with open(save_dir / "report_risk_level.json", "w", encoding="utf-8") as f:
        json.dump({"risk_level": report["risk_level"]}, f, ensure_ascii=False, indent=2)

    with open(save_dir / "report_recommendations.json", "w", encoding="utf-8") as f:
        json.dump({"recommendations": report["recommendations"]}, f, ensure_ascii=False, indent=2)

    with open(save_dir / "report_evidence.json", "w", encoding="utf-8") as f:
        json.dump({"evidence": report["evidence"]}, f, ensure_ascii=False, indent=2)

    with open(save_dir / "report_disclaimer.json", "w", encoding="utf-8") as f:
        json.dump({"disclaimer": report["disclaimer"]}, f, ensure_ascii=False, indent=2)

    print(f"\n>> 각 키를 JSON 파일로 저장 완료: {save_dir.resolve()}")
