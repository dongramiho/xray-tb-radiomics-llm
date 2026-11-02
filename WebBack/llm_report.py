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
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

import guardrails
from kb import HashingEmbedder, KnowledgeBase, OpenAIEmbedder, case_query, format_context
from llm_pricing import estimate_cost_usd
from llm_schemas import RadiomicsReport
from settings import Settings

logger = logging.getLogger(__name__)

PROMPT_VERSION = "2026.11.02-r1"
KB_FILE_PATTERN = "tb_*.txt"


# ------------------------------------------------------------------------------------- telemetry types


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMError(RuntimeError):
    """Raised when every attempt (including the fallback model) failed."""


# ------------------------------------------------------------------------------------- clients


class ChatClient(Protocol):
    name: str

    def complete(
        self, system: str, user: str, *, model: str, temperature: float, max_tokens: int, timeout: float
    ) -> tuple[str, Usage]: ...

    def complete_structured(
        self,
        system: str,
        user: str,
        *,
        schema: type[BaseModel],
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> tuple[dict[str, Any], Usage]: ...

    def stream(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> Iterator[str]: ...


def _usage_from(response: Any) -> Usage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage()
    return Usage(int(getattr(usage, "prompt_tokens", 0) or 0), int(getattr(usage, "completion_tokens", 0) or 0))


class OpenAIChatClient:
    name = "openai"

    def __init__(self, api_key: str):
        from openai import OpenAI

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (or set LLM_BACKEND=fake).")
        self._client = OpenAI(api_key=api_key)

    @staticmethod
    def _messages(system: str, user: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def complete(self, system, user, *, model, temperature, max_tokens, timeout):
        completion = self._client.chat.completions.create(
            model=model,
            messages=self._messages(system, user),
            temperature=temperature,
            max_completion_tokens=max_tokens,
            timeout=timeout,
        )
        return (completion.choices[0].message.content or ""), _usage_from(completion)

    def complete_structured(self, system, user, *, schema, model, temperature, max_tokens, timeout):
        completion = self._client.chat.completions.parse(
            model=model,
            messages=self._messages(system, user),
            response_format=schema,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            timeout=timeout,
        )
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise LLMError(f"model refused: {message.refusal}")
        parsed = message.parsed
        if parsed is None:
            raise LLMError("structured output missing")
        return parsed.model_dump(mode="json"), _usage_from(completion)

    def stream(self, system, user, *, model, temperature, max_tokens, timeout, on_usage=None):
        response = self._client.chat.completions.create(
            model=model,
            messages=self._messages(system, user),
            temperature=temperature,
            max_completion_tokens=max_tokens,
            timeout=timeout,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in response:
            if getattr(chunk, "usage", None) and on_usage is not None:
                on_usage(_usage_from(chunk))
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


class FakeChatClient:
    """Offline client: builds a schema-valid report that is consistent with the case JSON in the prompt.

    Used when LLM_BACKEND=fake and in tests; ``calls`` records every request for assertions.
    """

    name = "fake"
    CASE_PATTERN = re.compile(r"\[RADIOMICS_CASE_JSON\]\n(.*?)\n\[END_OF_RADIOMICS_CASE_JSON\]", re.S)
    SUMMARY = (
        "분석 요약(FAKE): 분할된 영역의 질감이 불균질하고 분류기는 결핵 가능성을 {level} 수준으로 평가했습니다. "
        "임상 증상과 객담 검사를 함께 고려하시기 바랍니다. 최종 판단은 담당 의사에게 있습니다."
    )

    def __init__(self, summary_override: str | None = None, structured_override: dict[str, Any] | None = None):
        self.calls: list[dict[str, Any]] = []
        self.summary_override = summary_override
        self.structured_override = structured_override

    def _record(self, kind: str, model: str, user: str) -> None:
        self.calls.append({"kind": kind, "model": model, "user": user})

    def _case_from_prompt(self, user: str) -> dict[str, Any]:
        match = self.CASE_PATTERN.search(user)
        if not match:
            return {}
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {}

    def complete(self, system, user, *, model, temperature, max_tokens, timeout):
        self._record("complete", model, user)
        level = "moderate"
        match = re.search(r'"level":\s*"(low|moderate|high)"', user)
        if match:
            level = match.group(1)
        text = self.summary_override or self.SUMMARY.format(level=level)
        return text, Usage(input_tokens=max(1, len(user) // 4), output_tokens=max(1, len(text) // 2))

    def complete_structured(self, system, user, *, schema, model, temperature, max_tokens, timeout):
        self._record("structured", model, user)
        if self.structured_override is not None:
            return dict(self.structured_override), Usage(100, 50)
        case = self._case_from_prompt(user)
        seg = case.get("segmentation", {}) or {}
        ml = case.get("ml_results", {}) or {}
        label = ml.get("primary_label", "tb_suspected")
        no_lesion = label == "no_lesion_detected"
        chunk_ids = re.findall(r"^\[(tb_\d+#\d+)\]", user, flags=re.M)
        evidence = [
            {
                "json_key": "ml_results.primary_label",
                "category": "ml",
                "role": "tb_risk",
                "explanation": f"classifier label {label}",
            },
            {
                "json_key": "segmentation.mask_area_fraction",
                "category": "segmentation",
                "role": "morphology",
                "explanation": "segmented area fraction",
            },
        ]
        if chunk_ids:
            evidence.append(
                {
                    "json_key": chunk_ids[0],
                    "category": "knowledge_base",
                    "role": "tb_risk",
                    "explanation": "knowledge-base pattern",
                }
            )
        report = RadiomicsReport.model_validate(
            {
                "clinical_overview": "FAKE report: "
                + ("no lesion region segmented." if no_lesion else "segmented region with heterogeneous texture."),
                "tb_risk_summary": {"level": ml.get("risk_level", "moderate"), "rationale": "fake rationale"},
                "lesion_summary": {
                    "num_regions": 0 if no_lesion else int(seg.get("num_components", 1) or 0),
                    "zones": [] if no_lesion else list(seg.get("zones", [])),
                    "area_fraction": None if no_lesion else seg.get("mask_area_fraction"),
                    "morphology": "n/a" if no_lesion else "irregular",
                    "texture": "n/a" if no_lesion else "heterogeneous",
                },
                "ml_summary": {
                    "primary_label": label,
                    "tb_probability": ml.get("tb_probability"),
                    "model_confidence": "moderate",
                    "reliability_comment": "fake",
                },
                "recommendations_for_clinician": ["Correlate with symptoms and sputum testing."],
                "supporting_evidence": evidence,
                "limitations": {
                    "data_limitations": ["single-centre data"],
                    "model_limitations": ["in-sample segmentation"],
                    "usage_restrictions": "research only",
                },
            }
        )
        return report.model_dump(mode="json"), Usage(input_tokens=max(1, len(user) // 4), output_tokens=400)

    def stream(self, system, user, *, model, temperature, max_tokens, timeout, on_usage=None):
        text, usage = self.complete(
            system, user, model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout
        )
        self.calls[-1]["kind"] = "stream"
        for start in range(0, len(text), 12):
            yield text[start : start + 12]
        if on_usage is not None:
            on_usage(usage)


def build_chat_client(settings: Settings) -> ChatClient:
    if settings.llm_backend == "fake":
        logger.warning("Using FAKE LLM backend; reports and summaries are synthetic.")
        return FakeChatClient()
    return OpenAIChatClient(settings.openai_api_key)


def build_knowledge_base(settings: Settings) -> KnowledgeBase:
    embedder = (
        HashingEmbedder()
        if settings.llm_backend == "fake"
        else OpenAIEmbedder(settings.openai_api_key, settings.embedding_model)
    )
    return KnowledgeBase.build(Path(settings.kb_dir), embedder)


# ------------------------------------------------------------------------------------- resilience


def _retryable(exc: BaseException) -> bool:
    try:
        import openai
    except ImportError:  # pragma: no cover
        return False
    if isinstance(exc, openai.RateLimitError | openai.APIConnectionError | openai.APITimeoutError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    return False


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
                if not _retryable(exc):
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
