"""LLM layer: stage 1 (structured clinical report) and stage 2 (clinician-facing summary / follow-ups).

Design
- One ``ChatClient`` abstraction (OpenAI or an offline ``FakeChatClient``) with three operations:
  ``complete`` (text), ``complete_structured`` (JSON validated against a pydantic schema) and ``stream``.
- Every call goes through ``call_with_resilience``: timeout, exponential-backoff retries on rate limits
  and transient errors, then a fallback model. The result carries an ``LLMTrace`` (model actually used,
  latency, tokens, estimated cost, attempts) that the API persists for cost / latency telemetry.
- Stage 1 retrieves the top-k knowledge-base chunks for the case (RAG) instead of pasting all documents,
  and asks the model to cite chunk ids in ``supporting_evidence``; citations are validated afterwards.
- Stage 2 output passes the guardrails (no definitive diagnosis, physician deferral); a violating draft
  is regenerated once with explicit feedback before the disclaimer is appended.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import guardrails
from kb import HashingEmbedder, KnowledgeBase, OpenAIEmbedder, case_query, format_context
from llm_clients import (  # noqa: F401  (re-exported for callers and tests)
    ChatClient,
    FakeChatClient,
    GeminiChatClient,
    LLMError,
    LLMRouter,
    OpenAIChatClient,
    Usage,
    build_llm_router,
    is_retryable,
)
from llm_pricing import estimate_cost_usd
from llm_schemas import RadiomicsReport
from settings import Settings

logger = logging.getLogger(__name__)

PROMPT_VERSION = "2026.11.02-r1"
KB_FILE_PATTERN = "tb_*.txt"


# ------------------------------------------------------------------------------------- telemetry types


@dataclass
class LLMTrace:
    stage: str
    requested_model: str
    model: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    attempts: int = 0
    fallback_used: bool = False
    prompt_version: str = PROMPT_VERSION
    success: bool = False
    error: str | None = None
    guardrail_violations: list[str] = field(default_factory=list)
    regenerated: bool = False
    cached: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_usage(
        cls, stage: str, model: str, usage: Usage | None, latency_ms: float = 0.0, cached: bool = False
    ) -> LLMTrace:
        trace = cls(
            stage=stage, requested_model=model, model=model, attempts=0 if cached else 1, success=True, cached=cached
        )
        if usage is not None:
            trace.input_tokens, trace.output_tokens = usage.input_tokens, usage.output_tokens
            trace.cost_usd = estimate_cost_usd(model, usage.input_tokens, usage.output_tokens)
        trace.latency_ms = round(latency_ms, 1)
        return trace


def build_chat_client(settings: Settings) -> ChatClient:
    """The provider router (OpenAI + optional Gemini), or the offline fake client when LLM_BACKEND=fake."""
    return build_llm_router(settings)


def build_knowledge_base(settings: Settings) -> KnowledgeBase:
    embedder = (
        HashingEmbedder()
        if settings.llm_backend == "fake"
        else OpenAIEmbedder(settings.openai_api_key, settings.embedding_model)
    )
    return KnowledgeBase.build(Path(settings.kb_dir), embedder)


# ------------------------------------------------------------------------------------- resilience


def call_with_resilience(
    stage: str,
    models: list[str],
    attempt: Callable[[str], tuple[Any, Usage]],
    *,
    max_retries: int = 2,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Any, LLMTrace]:
    """Try ``attempt(model)`` on each model in order with retries; return the result and a trace."""
    trace = LLMTrace(stage=stage, requested_model=models[0])
    started = time.perf_counter()
    last_error: BaseException | None = None
    for index, model in enumerate(models):
        for retry in range(max_retries + 1):
            trace.attempts += 1
            try:
                result, usage = attempt(model)
            except LLMError:
                raise
            except Exception as exc:  # noqa: BLE001 - we classify below
                last_error = exc
                if not is_retryable(exc):
                    logger.warning("%s: non-retryable error on %s: %s", stage, model, exc)
                    break
                if retry < max_retries:
                    delay = min(8.0, 0.5 * (2**retry)) + random.uniform(0, 0.25)
                    logger.warning(
                        "%s: %s on %s, retry %d in %.1fs", stage, type(exc).__name__, model, retry + 1, delay
                    )
                    sleep(delay)
                continue
            trace.model = model
            trace.fallback_used = index > 0
            trace.input_tokens, trace.output_tokens = usage.input_tokens, usage.output_tokens
            trace.cost_usd = estimate_cost_usd(model, usage.input_tokens, usage.output_tokens)
            trace.latency_ms = round((time.perf_counter() - started) * 1000, 1)
            trace.success = True
            return result, trace
        if index + 1 < len(models):
            logger.warning("%s: falling back from %s to %s", stage, model, models[index + 1])
    trace.latency_ms = round((time.perf_counter() - started) * 1000, 1)
    trace.error = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown"
    raise LLMError(f"{stage} failed after {trace.attempts} attempts: {trace.error}")


# ------------------------------------------------------------------------------------- stage 1: structured report

REPORT_SYSTEM_PROMPT = """
You are a medical AI assistant whose sole role is to analyze the output of a chest X-ray
tuberculosis (TB) pipeline: a U-Net lesion segmentation summary, PyRadiomics features of the
segmented region, a radiomics-based TB classifier output, and the classifier's training metadata.

You DO NOT talk to patients. You DO NOT create conversational, emotional, or reassuring language.

You must return a JSON object matching the provided schema. Field guidance:
- clinical_overview: short, technical summary of the overall findings.
- tb_risk_summary.level: low | moderate | high, consistent with ml_results.risk_level unless the
  radiomics pattern clearly argues otherwise (then explain why in rationale).
- lesion_summary.zones: copy radiographic lung zones from segmentation.zones (right/left x
  upper/middle/lower). Never upgrade zones to lobes. num_regions = segmentation.num_components.
- lesion_summary.morphology / texture: ground them in shape2D and GLCM / GLRLM / GLSZM features.
- ml_summary: copy primary_label and tb_probability from ml_results; model_confidence should reflect
  the cross-validated reliability in MODEL_METADATA and the size of the segmented region.
- supporting_evidence.json_key: a path in the case JSON (e.g. ml_results.tb_probability,
  segmentation.zones, radiomics_features.original_glcm_Contrast) or a knowledge-base chunk id
  exactly as given in brackets (e.g. tb_5#0, category knowledge_base). Include at least three items,
  at least one of them a knowledge-base chunk.
- recommendations_for_clinician: technical, clinician-facing next steps.
- limitations: data and model limitations for this case, plus usage restrictions.

Rules:
- Ground every statement in the provided case JSON and knowledge-base chunks; never invent values.
- If ml_results.primary_label is "no_lesion_detected": num_regions = 0, zones = [], tb_probability = null,
  level = low, and recommend clinical correlation instead of describing lesions.
- Quality control: if case.qc.passed is false or qc.flags is non-empty, state the reduced reliability in
  ml_summary.reliability_comment and limitations.model_limitations (name the flags), lower model_confidence,
  and recommend physician review of the images.
- CT cases (case.modality == "ct"): the mask was produced outside this system; describe the tumor-labelled
  region (volume, extent, HU statistics, 3D texture) and set tb_probability = null. If
  ml_results.primary_label is "ct_tumor_label_missing", say that no tumor label was present, which is NOT
  evidence of absence, and recommend review; never conclude "no tumor".
- Use MODEL_METADATA only to contextualize reliability; do not restate its numbers as patient findings.
"""


def build_report_prompt(case_json: dict[str, Any], kb_context: str, model_meta: dict[str, Any]) -> str:
    return f"""
[MODEL_METADATA]
{json.dumps(model_meta, ensure_ascii=False, indent=2)}
[END_OF_MODEL_METADATA]

[TB_KNOWLEDGE_BASE_CHUNKS]
{kb_context}
[END_OF_TB_KNOWLEDGE_BASE_CHUNKS]

[RADIOMICS_CASE_JSON]
{json.dumps(case_json, ensure_ascii=False, indent=2)}
[END_OF_RADIOMICS_CASE_JSON]

Interpret the segmentation summary, the radiomics values and the ML output together, focusing on
TB-related morphologic, textural and intensity patterns described in the knowledge-base chunks.
Cite chunk ids (in brackets above) in supporting_evidence where a pattern from the knowledge base is used.
"""


def generate_radiomics_report(
    case_json: dict[str, Any],
    *,
    client: ChatClient,
    kb: KnowledgeBase,
    model_meta: dict[str, Any],
    model: str,
    fallback_model: str | None = None,
    top_k: int = 6,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> tuple[dict[str, Any], LLMTrace]:
    """Stage 1: case JSON -> validated structured report (dict) + trace."""
    if not isinstance(case_json, dict):
        raise TypeError("case_json must be a dict")
    results = kb.search(case_query(case_json), k=top_k)
    prompt = build_report_prompt(case_json, format_context(results), model_meta)
    retrieved_ids = {chunk.id for chunk, _ in results}

    def attempt(model_name: str) -> tuple[dict[str, Any], Usage]:
        return client.complete_structured(
            REPORT_SYSTEM_PROMPT,
            prompt,
            schema=RadiomicsReport,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    models = [model] + ([fallback_model] if fallback_model and fallback_model != model else [])
    parsed, trace = call_with_resilience("report", models, attempt, max_retries=max_retries)
    report = guardrails.validate_evidence(parsed, case_json, retrieved_ids | kb.chunk_ids)
    report["retrieved_chunks"] = [{"id": chunk.id, "score": round(score, 3)} for chunk, score in results]
    return report, trace


# ------------------------------------------------------------------------------------- stage 2: clinician summary

SUMMARY_SYSTEM_PROMPT = """
You are a medical AI assistant supporting a clinician.
The input is a structured JSON analysis produced from a chest X-ray by a research pipeline
(U-Net segmentation, radiomics, a TB classifier, and a first-stage LLM report).
Write an interpretation-focused summary covering:
- what the analysis found (regions, zones, texture, classifier output)
- the risk level and why
- recommendations for the clinician

Requirements:
- Do NOT provide definitive diagnoses; speak in likelihoods, patterns, and clinical considerations.
- Do NOT discuss internal validation metrics of the model.
- Clearly state that the final clinical judgment belongs to the physician (담당 의사).
- Keep it concise (about 150-250 Korean words), plain paragraphs or short "- " bullet lists.
- The output must be written in Korean.
"""

QUESTION_SYSTEM_PROMPT = """
You are a medical AI assistant supporting a clinician in a chat about a chest X-ray case.
If a structured analysis of the most recent X-ray is provided, use it as context; otherwise answer
generally. Do NOT provide definitive diagnoses, do NOT invent findings that are not in the context,
and state that the final clinical judgment belongs to the physician (담당 의사). Answer in Korean, concisely.
"""

REGENERATE_SUFFIX = (
    "\n\n[REVISION_REQUEST]\nYour previous draft contained definitive diagnostic language "
    "(matched: {patterns}). Rewrite it using likelihood language only, and keep the physician deferral."
)


def _summary_user_prompt(report: dict[str, Any], clinician_note: str) -> str:
    payload = {k: v for k, v in report.items() if k not in {"retrieved_chunks", "validation"}}
    user = json.dumps(payload, ensure_ascii=False, indent=2)
    if clinician_note.strip():
        user += f"\n\n[CLINICIAN_NOTE]\n{clinician_note.strip()}\n[END_OF_CLINICIAN_NOTE]"
    return user


def _question_user_prompt(question: str, context_report: dict[str, Any] | None) -> str:
    user = question.strip()
    if context_report:
        payload = {k: v for k, v in context_report.items() if k not in {"retrieved_chunks", "validation"}}
        user = f"[LATEST_ANALYSIS]\n{json.dumps(payload, ensure_ascii=False)}\n[END_OF_LATEST_ANALYSIS]\n\n{user}"
    return user


def _guarded_text_call(
    stage: str,
    system: str,
    user: str,
    *,
    client: ChatClient,
    model: str,
    fallback_model: str | None,
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
) -> tuple[str, LLMTrace]:
    """Text completion + guardrails, regenerating once if the draft asserts a diagnosis."""
    models = [model] + ([fallback_model] if fallback_model and fallback_model != model else [])

    def attempt(model_name: str) -> tuple[str, Usage]:
        return client.complete(
            system, user, model=model_name, temperature=temperature, max_tokens=max_tokens, timeout=timeout
        )

    text, trace = call_with_resilience(stage, models, attempt, max_retries=max_retries)
    guard = guardrails.check_summary(text)
    if not guard.passed:
        trace.guardrail_violations = list(guard.violations)
        trace.regenerated = True
        patterns = ", ".join(v.split(":", 1)[1] for v in guard.violations)
        retry_user = user + REGENERATE_SUFFIX.format(patterns=patterns)
        text2, trace2 = call_with_resilience(
            stage,
            [trace.model or model],
            lambda m: client.complete(
                system, retry_user, model=m, temperature=temperature, max_tokens=max_tokens, timeout=timeout
            ),
            max_retries=max_retries,
        )
        trace.attempts += trace2.attempts
        trace.input_tokens += trace2.input_tokens
        trace.output_tokens += trace2.output_tokens
        trace.cost_usd = (trace.cost_usd or 0.0) + (trace2.cost_usd or 0.0)
        trace.latency_ms += trace2.latency_ms
        guard = guardrails.check_summary(text2)
        if guard.passed:
            text = guard.text
        else:
            logger.warning("%s: definitive language persisted after regeneration; appending disclaimer", stage)
            text = guard.text + ("" if guard.appended_disclaimer else "\n\n" + guardrails.DISCLAIMER_KO)
        return text, trace
    return guard.text, trace


def summarize_for_clinician(
    report: dict[str, Any],
    *,
    client: ChatClient,
    model: str,
    fallback_model: str | None = None,
    clinician_note: str = "",
    temperature: float = 0.2,
    max_tokens: int = 1000,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> tuple[str, LLMTrace]:
    """Stage 2: structured report -> Korean clinician-facing summary (guardrailed)."""
    return _guarded_text_call(
        "summary",
        SUMMARY_SYSTEM_PROMPT,
        _summary_user_prompt(report, clinician_note),
        client=client,
        model=model,
        fallback_model=fallback_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )


def answer_clinician_question(
    question: str,
    *,
    client: ChatClient,
    model: str,
    fallback_model: str | None = None,
    context_report: dict[str, Any] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> tuple[str, LLMTrace]:
    """Follow-up chat turn without a new image; the latest report (if any) is passed as context."""
    return _guarded_text_call(
        "question",
        QUESTION_SYSTEM_PROMPT,
        _question_user_prompt(question, context_report),
        client=client,
        model=model,
        fallback_model=fallback_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )


def stream_text(
    stage: str,
    system: str,
    user: str,
    *,
    client: ChatClient,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> Iterator[tuple[str, Any]]:
    """Yield ("delta", text) events then one ("final", (text, trace)).

    Streaming cannot retry mid-response, so the primary model is used without fallback; the caller
    applies guardrails to the full text before persisting (the UI replaces the streamed draft).
    """
    trace = LLMTrace(stage=stage, requested_model=model, model=model, attempts=1)
    usage_box: list[Usage] = []
    started = time.perf_counter()
    pieces: list[str] = []
    try:
        for delta in client.stream(
            system,
            user,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            on_usage=usage_box.append,
        ):
            pieces.append(delta)
            yield "delta", delta
    except Exception as exc:  # noqa: BLE001
        trace.error = f"{type(exc).__name__}: {exc}"
        trace.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        raise LLMError(trace.error) from exc
    text = "".join(pieces)
    if usage_box:
        trace.input_tokens, trace.output_tokens = usage_box[-1].input_tokens, usage_box[-1].output_tokens
        trace.cost_usd = estimate_cost_usd(model, trace.input_tokens, trace.output_tokens)
    trace.latency_ms = round((time.perf_counter() - started) * 1000, 1)
    trace.success = True
    guard = guardrails.check_summary(text)
    if not guard.passed:
        trace.guardrail_violations = list(guard.violations)
        final_text = guard.text + ("" if guard.appended_disclaimer else "\n\n" + guardrails.DISCLAIMER_KO)
    else:
        final_text = guard.text
    yield "final", (final_text, trace)


def stream_summary_for_clinician(report, *, client, model, clinician_note="", timeout=60.0):
    return stream_text(
        "summary",
        SUMMARY_SYSTEM_PROMPT,
        _summary_user_prompt(report, clinician_note),
        client=client,
        model=model,
        temperature=0.2,
        max_tokens=1000,
        timeout=timeout,
    )


def stream_answer(question, *, client, model, context_report=None, timeout=60.0):
    return stream_text(
        "question",
        QUESTION_SYSTEM_PROMPT,
        _question_user_prompt(question, context_report),
        client=client,
        model=model,
        temperature=0.2,
        max_tokens=800,
        timeout=timeout,
    )


# ------------------------------------------------------------------------------------- compatibility helpers


def load_kb_text(kb_dir: Path) -> str:
    """Concatenate the knowledge-base documents (used by scripts that want the full text)."""
    files = sorted(Path(kb_dir).glob(KB_FILE_PATTERN), key=lambda p: (len(p.stem), p.stem))
    parts = [f.read_text(encoding="utf-8").strip() for f in files]
    parts = [p for p in parts if p]
    if not parts:
        raise RuntimeError(f"No knowledge-base documents ({KB_FILE_PATTERN}) found in {kb_dir}")
    return "\n\n".join(parts)
