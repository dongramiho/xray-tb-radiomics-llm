"""LLM calls used by the chat graph beyond the stage-1 report and clinician summary (which live in llm_report)."""

from __future__ import annotations

import json
import time
from typing import Any

import guardrails
import llm_report
from graph import prompts
from llm_clients import ChatClient, Usage
from llm_report import LLMTrace, call_with_resilience
from llm_schemas import ConsistencyCheck, FollowUpQuestions, SessionTitle

PATIENT_JARGON = ("radiomics", "GLCM", "GLSZM", "GLRLM", "텍스처 매트릭스", "sigmoid", "U-Net")


def _report_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in report.items() if k not in {"retrieved_chunks", "validation"}}


def explain_for_patient(
    report: dict[str, Any],
    *,
    client: ChatClient,
    model: str,
    fallback_model: str | None,
    doctor_summary: str = "",
    revision_note: str = "",
    timeout: float = 60.0,
    max_retries: int = 2,
) -> tuple[str, LLMTrace]:
    """Patient-facing explanation of the same analysis, guardrailed like the clinician summary."""
    user = "[PATIENT_AUDIENCE]\n" + json.dumps(_report_payload(report), ensure_ascii=False, indent=2)
    if doctor_summary:
        user += f"\n\n[CLINICIAN_SUMMARY]\n{doctor_summary}\n[END_OF_CLINICIAN_SUMMARY]"
    if revision_note:
        user += f"\n\n[REVISION_REQUEST]\n{revision_note}"
    models = [model] + ([fallback_model] if fallback_model and fallback_model != model else [])

    def attempt(model_name: str) -> tuple[str, Usage]:
        return client.complete(
            prompts.PATIENT_SYSTEM_PROMPT, user, model=model_name, temperature=0.3, max_tokens=700, timeout=timeout
        )

    text, trace = call_with_resilience("patient_explanation", models, attempt, max_retries=max_retries)
    guard = guardrails.check_summary(text)
    jargon = [term for term in PATIENT_JARGON if term.lower() in text.lower()]
    if not guard.passed or jargon:
        trace.guardrail_violations = list(guard.violations) + [f"jargon:{t}" for t in jargon]
        trace.regenerated = True
        note = "Remove definitive diagnostic language" if not guard.passed else ""
        if jargon:
            note += (" and " if note else "") + f"avoid technical terms ({', '.join(jargon)})"
        retry_user = user + f"\n\n[REVISION_REQUEST]\n{note}. Keep the physician deferral."
        text2, trace2 = call_with_resilience(
            "patient_explanation",
            [trace.model or model],
            lambda m: client.complete(
                prompts.PATIENT_SYSTEM_PROMPT, retry_user, model=m, temperature=0.3, max_tokens=700, timeout=timeout
            ),
            max_retries=max_retries,
        )
        trace.attempts += trace2.attempts
        trace.input_tokens += trace2.input_tokens
        trace.output_tokens += trace2.output_tokens
        trace.cost_usd = (trace.cost_usd or 0.0) + (trace2.cost_usd or 0.0)
        trace.latency_ms += trace2.latency_ms
        guard = guardrails.check_summary(text2)
        text = (
            guard.text
            if guard.passed
            else guard.text + ("" if guard.appended_disclaimer else "\n\n" + guardrails.DISCLAIMER_KO)
        )
        return text, trace
    return guard.text, trace


def check_consistency(
    report: dict[str, Any],
    doctor_summary: str,
    patient_summary: str,
    *,
    client: ChatClient,
    model: str,
    timeout: float = 30.0,
) -> tuple[dict[str, Any], LLMTrace]:
    user = (
        f"[REPORT_RISK_LEVEL] {report.get('tb_risk_summary', {}).get('level')}\n\n"
        f"[CLINICIAN]\n{doctor_summary}\n\n[PATIENT]\n{patient_summary}"
    )
    started = time.perf_counter()
    decision, usage = client.complete_structured(
        prompts.COMPARE_SYSTEM_PROMPT,
        user,
        schema=ConsistencyCheck,
        model=model,
        temperature=0.0,
        max_tokens=300,
        timeout=timeout,
    )
    decision = ConsistencyCheck.model_validate(decision).model_dump()
    return decision, LLMTrace.from_usage("consistency_check", model, usage, (time.perf_counter() - started) * 1000)


def suggest_follow_ups(
    report: dict[str, Any] | None, question: str, answer: str, *, client: ChatClient, model: str, timeout: float = 30.0
) -> tuple[dict[str, list[str]], LLMTrace]:
    user = (
        f"[REPORT]\n{json.dumps(_report_payload(report), ensure_ascii=False) if report else 'none'}\n\n"
        f"[LATEST_QUESTION]\n{question or '(image analysis request)'}\n\n[LATEST_ANSWER]\n{answer[:1500]}"
    )
    started = time.perf_counter()
    decision, usage = client.complete_structured(
        prompts.FOLLOWUP_SYSTEM_PROMPT,
        user,
        schema=FollowUpQuestions,
        model=model,
        temperature=0.4,
        max_tokens=400,
        timeout=timeout,
    )
    decision = FollowUpQuestions.model_validate(decision).model_dump()
    decision["doctor"] = [q.strip() for q in decision["doctor"] if q.strip()][:3]
    decision["patient"] = [q.strip() for q in decision["patient"] if q.strip()][:2]
    return decision, LLMTrace.from_usage("follow_ups", model, usage, (time.perf_counter() - started) * 1000)


def generate_title(
    report: dict[str, Any], *, client: ChatClient, model: str, timeout: float = 30.0
) -> tuple[str, LLMTrace]:
    user = json.dumps(
        {k: report.get(k) for k in ("clinical_overview", "tb_risk_summary", "lesion_summary")}, ensure_ascii=False
    )
    started = time.perf_counter()
    decision, usage = client.complete_structured(
        prompts.TITLE_SYSTEM_PROMPT,
        user,
        schema=SessionTitle,
        model=model,
        temperature=0.3,
        max_tokens=60,
        timeout=timeout,
    )
    title = SessionTitle.model_validate(decision).title.strip()[:24]
    return title, LLMTrace.from_usage("title", model, usage, (time.perf_counter() - started) * 1000)


def answer_with_context(
    question: str,
    *,
    client: ChatClient,
    model: str,
    fallback_model: str | None,
    context_report: dict[str, Any] | None,
    retrieval: dict[str, Any] | None,
    timeout: float = 60.0,
    max_retries: int = 2,
    patient_audience: bool = False,
) -> tuple[str, LLMTrace]:
    """Non-streaming answer for follow-up / general questions with retrieved context (used by /analyze)."""
    user = build_question_prompt(question, context_report, retrieval, patient_audience=patient_audience)
    return llm_report._guarded_text_call(  # noqa: SLF001 - shared guardrailed call
        "question",
        prompts.PATIENT_QA_SYSTEM_PROMPT if patient_audience else prompts.GENERAL_SYSTEM_PROMPT,
        user,
        client=client,
        model=model,
        fallback_model=fallback_model,
        temperature=0.2,
        max_tokens=800,
        timeout=timeout,
        max_retries=max_retries,
    )


def build_question_prompt(
    question: str,
    context_report: dict[str, Any] | None,
    retrieval: dict[str, Any] | None,
    patient_audience: bool = False,
) -> str:
    parts: list[str] = ["[PATIENT_AUDIENCE]"] if patient_audience else []
    if context_report:
        latest = json.dumps(_report_payload(context_report), ensure_ascii=False)
        parts.append(f"[LATEST_ANALYSIS]\n{latest}\n[END_OF_LATEST_ANALYSIS]")
    if retrieval:
        kb = retrieval.get("kb") or []
        if kb:
            parts.append(
                "[KNOWLEDGE_BASE_CHUNKS]\n"
                + "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in kb)
                + "\n[END_OF_KNOWLEDGE_BASE_CHUNKS]"
            )
        lit = retrieval.get("literature") or []
        if lit:
            parts.append(
                "[LITERATURE]\n"
                + "\n\n".join(
                    f"[PMID {item['pmid']}] {item['title']} ({item['journal']}, {item['year']})\n{item['abstract']}"
                    for item in lit
                )
                + "\n[END_OF_LITERATURE]"
            )
        cases = retrieval.get("similar_cases") or []
        if cases:
            parts.append(
                "[SIMILAR_REFERENCE_CASES]\n"
                + json.dumps(cases, ensure_ascii=False)
                + "\n[END_OF_SIMILAR_REFERENCE_CASES]"
            )
    parts.append(question.strip())
    return "\n\n".join(parts)
