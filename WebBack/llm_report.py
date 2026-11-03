"""LLM layer: stage 1 (structured clinical report) and stage 2 (clinician-facing summary).

Stage 1 turns the pipeline's case JSON (segmentation summary, radiomics features, classifier
output) plus the TB knowledge base and the classifier's training metadata into a strict JSON
report for clinicians / researchers. Stage 2 rewrites that report into a short Korean summary
for the chat UI and answers follow-up questions in the context of the latest report.

The OpenAI client is created lazily so importing this module never requires an API key, and a
``FakeChatClient`` provides deterministic output for tests and offline UI work.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

from settings import Settings

logger = logging.getLogger(__name__)

KB_FILE_PATTERN = "tb_*.txt"


# ------------------------------------------------------------------------------------- clients


class ChatClient(Protocol):
    def complete(
        self, system: str, user: str, *, model: str, temperature: float, max_tokens: int, json_mode: bool
    ) -> str: ...


class OpenAIChatClient:
    def __init__(self, api_key: str):
        from openai import OpenAI  # local import: optional at import time

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (or set LLM_BACKEND=fake).")
        self._client = OpenAI(api_key=api_key)

    def complete(
        self, system: str, user: str, *, model: str, temperature: float, max_tokens: int, json_mode: bool
    ) -> str:
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        completion = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return completion.choices[0].message.content or ""


class FakeChatClient:
    """Returns a schema-valid report / a canned summary; used when LLM_BACKEND=fake and in tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(
        self, system: str, user: str, *, model: str, temperature: float, max_tokens: int, json_mode: bool
    ) -> str:
        self.calls.append({"model": model, "json_mode": json_mode, "user": user})
        if json_mode:
            return json.dumps(
                {
                    "clinical_overview": "FAKE report: segmented region with heterogeneous texture.",
                    "tb_risk_summary": {"level": "moderate", "rationale": "fake rationale"},
                    "lesion_summary": {
                        "num_regions": 1,
                        "zones": ["right upper zone"],
                        "area_fraction": 0.02,
                        "morphology": "irregular",
                        "texture": "heterogeneous",
                    },
                    "ml_summary": {
                        "primary_label": "tb_suspected",
                        "tb_probability": 0.81,
                        "model_confidence": "moderate",
                        "reliability_comment": "fake",
                    },
                    "recommendations_for_clinician": ["Correlate with sputum testing."],
                    "supporting_evidence": [
                        {
                            "json_key": "ml_results.tb_probability",
                            "category": "ml",
                            "role": "tb_risk",
                            "explanation": "fake",
                        }
                    ],
                    "limitations": {
                        "data_limitations": ["fake"],
                        "model_limitations": ["fake"],
                        "usage_restrictions": "research only",
                    },
                }
            )
        return (
            "FAKE 요약: 우상부 폐야에 분할된 병변이 있으며 결핵 가능성은 중등도입니다. 최종 판단은 담당의에게 있습니다."
        )


def build_chat_client(settings: Settings) -> ChatClient:
    if settings.llm_backend == "fake":
        logger.warning("Using FAKE LLM backend; summaries are canned text.")
        return FakeChatClient()
    return OpenAIChatClient(settings.openai_api_key)


# ------------------------------------------------------------------------------------- knowledge base


def load_kb_text(kb_dir: Path) -> str:
    """Concatenate the TB knowledge-base documents. Fails loudly instead of silently prompting without them."""
    files = sorted(Path(kb_dir).glob(KB_FILE_PATTERN), key=lambda p: (len(p.stem), p.stem))
    parts = [f.read_text(encoding="utf-8").strip() for f in files]
    parts = [p for p in parts if p]
    if not parts:
        raise RuntimeError(f"No knowledge-base documents ({KB_FILE_PATTERN}) found in {kb_dir}")
    return "\n\n".join(parts)


# ------------------------------------------------------------------------------------- stage 1: structured report

REPORT_SYSTEM_PROMPT = """
You are a medical AI assistant whose sole role is to analyze the output of a chest X-ray
tuberculosis (TB) pipeline: a U-Net lesion segmentation summary, PyRadiomics features of the
segmented region, a radiomics-based TB classifier output, and the classifier's training metadata.

You DO NOT talk to patients.
You DO NOT create conversational explanations.
You DO NOT create emotional or human-facing language.

REQUIRED JSON STRUCTURE (return exactly one JSON object with these keys):

{
  "clinical_overview": "short, technical summary of overall findings",
  "tb_risk_summary": {
    "level": "low | moderate | high",
    "rationale": "technical explanation that cites the key segmentation / radiomics / ML drivers"
  },
  "lesion_summary": {
    "num_regions": 0,
    "zones": ["right upper zone"],
    "area_fraction": 0.0,
    "morphology": "shape / size description grounded in shape2D features",
    "texture": "homogeneity / heterogeneity description grounded in GLCM / GLRLM / GLSZM features"
  },
  "ml_summary": {
    "primary_label": "copied from ml_results.primary_label",
    "tb_probability": 0.0,
    "model_confidence": "low | moderate | high",
    "reliability_comment": "technical summary of reliability and limitations for this case"
  },
  "recommendations_for_clinician": ["technical, clinician-facing recommendations only"],
  "supporting_evidence": [
    {
      "json_key": "path.to.field in the case JSON",
      "category": "segmentation | radiomics | ml | meta",
      "role": "tb_risk | morphology | texture | reliability",
      "explanation": "why this field matters for the interpretation"
    }
  ],
  "limitations": {
    "data_limitations": ["string"],
    "model_limitations": ["string"],
    "usage_restrictions": "string"
  }
}

Rules:
- Ground every statement in the provided case JSON; never invent measurements that are not present.
- Zones are radiographic lung zones (right/left x upper/middle/lower); do not upgrade them to lobes.
- If ml_results.primary_label is "no_lesion_detected", say so, set tb_probability to null, set level to "low",
  and recommend clinical correlation rather than describing lesions.
- Use MODEL_METADATA only to contextualize reliability (training-set size, cross-validated performance,
  known limitations). Do not restate metadata values as findings about this patient.
- NO patient-friendly wording, NO reassurance, NO emotional expressions.
- Output only the JSON object, with no extra commentary.
"""


def build_report_prompt(case_json: dict[str, Any], kb_text: str, model_meta: dict[str, Any]) -> str:
    return f"""
[MODEL_METADATA]
{json.dumps(model_meta, ensure_ascii=False, indent=2)}
[END_OF_MODEL_METADATA]

[TB_RADIO_KNOWLEDGE_BASE]
{kb_text}
[END_OF_TB_RADIO_KNOWLEDGE_BASE]

[RADIOMICS_CASE_JSON]
{json.dumps(case_json, ensure_ascii=False, indent=2)}
[END_OF_RADIOMICS_CASE_JSON]

Instructions:
- Interpret the segmentation summary (zones, number of regions, area fraction), the radiomics values
  (shape2D, first-order, GLCM, GLRLM, GLSZM, GLDM, NGTDM) and the ML output together, focusing on
  TB-related morphologic, textural, and intensity patterns from the knowledge base.
- Reflect the classifier's cross-validated reliability from MODEL_METADATA in ml_summary.reliability_comment
  and in limitations, without quoting the numbers as findings.
- Output only a single JSON object that follows the required schema.
"""


def parse_llm_json(raw: str) -> dict[str, Any]:
    """Parse a JSON object from the model output, tolerating trailing text."""
    text = (raw or "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as err:
        last = text.rfind("}")
        if last == -1:
            raise ValueError("Invalid JSON from LLM:\n" + text) from err
        parsed = json.loads(text[: last + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM output is not a JSON object")
    return parsed


def normalize_report(parsed: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults and coerce loosely typed fields so downstream code can rely on the shape."""
    report: dict[str, Any] = dict(parsed)

    evidence = report.get("supporting_evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    normalized: list[dict[str, Any]] = []
    for item in evidence if isinstance(evidence, list) else []:
        if isinstance(item, str):
            normalized.append(
                {"json_key": "unknown", "category": "radiomics", "role": "morphology", "explanation": item.strip()}
            )
        elif isinstance(item, dict):
            normalized.append(
                {
                    "json_key": str(item.get("json_key", "unknown")),
                    "category": str(item.get("category", "radiomics")),
                    "role": str(item.get("role", "morphology")),
                    "explanation": str(item.get("explanation", "")).strip(),
                }
            )
    report["supporting_evidence"] = normalized

    report.setdefault("clinical_overview", "")
    report.setdefault("tb_risk_summary", {"level": "", "rationale": ""})
    report.setdefault(
        "lesion_summary", {"num_regions": None, "zones": [], "area_fraction": None, "morphology": "", "texture": ""}
    )
    report.setdefault(
        "ml_summary", {"primary_label": "", "tb_probability": None, "model_confidence": "", "reliability_comment": ""}
    )
    report.setdefault("recommendations_for_clinician", [])
    report.setdefault("limitations", {"data_limitations": [], "model_limitations": [], "usage_restrictions": ""})
    if isinstance(report["recommendations_for_clinician"], str):
        report["recommendations_for_clinician"] = [report["recommendations_for_clinician"]]
    return report


def generate_radiomics_report(
    case_json: dict[str, Any],
    *,
    client: ChatClient,
    kb_text: str,
    model_meta: dict[str, Any],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 2000,
) -> dict[str, Any]:
    """Stage 1: case JSON -> structured clinician/research report (dict)."""
    if not isinstance(case_json, dict):
        raise TypeError("case_json must be a dict")
    raw = client.complete(
        REPORT_SYSTEM_PROMPT,
        build_report_prompt(case_json, kb_text, model_meta),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=True,
    )
    return normalize_report(parse_llm_json(raw))


# ------------------------------------------------------------------------------------- stage 2: clinician summary

SUMMARY_SYSTEM_PROMPT = """
You are a medical AI assistant supporting a clinician.
The input is a structured JSON analysis produced from a chest X-ray by a research pipeline
(U-Net segmentation, radiomics, a TB classifier, and a first-stage LLM report).
Based on this input, provide an interpretation-focused summary covering:
- what the analysis found (regions, zones, texture, classifier output)
- risk level assessment
- recommendations for the clinician

Requirements:
- Do NOT provide definitive diagnoses; speak in likelihoods, patterns, and clinical considerations.
- Do NOT discuss internal validation metrics of the model.
- Clearly state that the final clinical judgment belongs to the physician.
- Keep it concise (about 150-250 Korean words) and use plain paragraphs or short bullet lists.
- The output must be written in Korean.
"""

QUESTION_SYSTEM_PROMPT = """
You are a medical AI assistant supporting a clinician in a chat about a chest X-ray case.
If a structured analysis of the most recent X-ray is provided, use it as context; otherwise answer
generally. Do NOT provide definitive diagnoses, do NOT invent findings that are not in the context,
and state that the final clinical judgment belongs to the physician. Answer in Korean, concisely.
"""


def summarize_for_clinician(
    report: dict[str, Any],
    *,
    client: ChatClient,
    model: str,
    clinician_note: str = "",
    temperature: float = 0.2,
    max_tokens: int = 1000,
) -> str:
    """Stage 2: structured report -> Korean clinician-facing summary."""
    user = json.dumps(report, ensure_ascii=False, indent=2)
    if clinician_note.strip():
        user += f"\n\n[CLINICIAN_NOTE]\n{clinician_note.strip()}\n[END_OF_CLINICIAN_NOTE]"
    return client.complete(
        SUMMARY_SYSTEM_PROMPT, user, model=model, temperature=temperature, max_tokens=max_tokens, json_mode=False
    ).strip()


def answer_clinician_question(
    question: str,
    *,
    client: ChatClient,
    model: str,
    context_report: dict[str, Any] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    """Follow-up chat turn without a new image; the latest report (if any) is passed as context."""
    user = question.strip()
    if context_report:
        user = (
            f"[LATEST_ANALYSIS]\n{json.dumps(context_report, ensure_ascii=False)}\n[END_OF_LATEST_ANALYSIS]\n\n{user}"
        )
    return client.complete(
        QUESTION_SYSTEM_PROMPT, user, model=model, temperature=temperature, max_tokens=max_tokens, json_mode=False
    ).strip()
