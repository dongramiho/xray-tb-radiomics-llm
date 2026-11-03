"""Chat clients behind one small protocol, plus a router that picks the provider by model name.

Providers
- ``OpenAIChatClient``  gpt-* models (structured outputs, streaming, usage).
- ``GeminiChatClient``  gemini-* models via google-genai; optional fallback provider (needs GOOGLE_API_KEY).
- ``FakeChatClient``    offline, deterministic; returns schema-valid objects for every structured schema the
                        app uses, so tests and UI work need no API key.
- ``LLMRouter``         implements the same protocol and dispatches on the model prefix, so pipeline code can
                        say "primary gpt-4.1, fallback gemini-2.5-flash" without knowing about providers.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


class LLMError(RuntimeError):
    """Raised when a provider refuses or every attempt (including the fallback model) failed."""


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


def is_retryable(exc: BaseException) -> bool:
    """Rate limits, connection problems, timeouts and 5xx responses are worth retrying; 4xx are not."""
    try:
        import openai

        if isinstance(exc, openai.RateLimitError | openai.APIConnectionError | openai.APITimeoutError):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code >= 500
    except ImportError:  # pragma: no cover
        pass
    try:
        from google.genai import errors as genai_errors

        if isinstance(exc, genai_errors.APIError):
            code = getattr(exc, "code", 0) or 0
            return code == 429 or code >= 500
    except ImportError:  # pragma: no cover
        pass
    return isinstance(exc, TimeoutError | ConnectionError)


# ------------------------------------------------------------------------------------- OpenAI


def _openai_usage(response: Any) -> Usage:
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
        return (completion.choices[0].message.content or ""), _openai_usage(completion)

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
        if message.parsed is None:
            raise LLMError("structured output missing")
        return message.parsed.model_dump(mode="json"), _openai_usage(completion)

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
                on_usage(_openai_usage(chunk))
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


# ------------------------------------------------------------------------------------- Gemini (optional)


class GeminiChatClient:
    """Google Gemini via google-genai. Used as the fallback provider when GOOGLE_API_KEY is configured.

    Thinking is kept low (``thinking_level``) for latency, matching the project's fallback design.
    """

    name = "gemini"

    def __init__(self, api_key: str, thinking_level: str = "low"):
        from google import genai

        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set; Gemini fallback is unavailable.")
        self._client = genai.Client(api_key=api_key)
        self.thinking_level = thinking_level

    def _config(self, system: str, temperature: float, max_tokens: int, timeout: float, schema=None):
        from google.genai import types

        kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "http_options": types.HttpOptions(timeout=int(timeout * 1000)),
        }
        if schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = schema
        try:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
        except (TypeError, ValueError):  # older SDK / model without thinking levels
            pass
        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _usage(response: Any) -> Usage:
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return Usage()
        return Usage(
            int(getattr(meta, "prompt_token_count", 0) or 0), int(getattr(meta, "candidates_token_count", 0) or 0)
        )

    def complete(self, system, user, *, model, temperature, max_tokens, timeout):
        response = self._client.models.generate_content(
            model=model, contents=user, config=self._config(system, temperature, max_tokens, timeout)
        )
        return (response.text or ""), self._usage(response)

    def complete_structured(self, system, user, *, schema, model, temperature, max_tokens, timeout):
        response = self._client.models.generate_content(
            model=model, contents=user, config=self._config(system, temperature, max_tokens, timeout, schema)
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, BaseModel):
            return parsed.model_dump(mode="json"), self._usage(response)
        if not response.text:
            raise LLMError("structured output missing")
        return schema.model_validate(json.loads(response.text)).model_dump(mode="json"), self._usage(response)

    def stream(self, system, user, *, model, temperature, max_tokens, timeout, on_usage=None):
        last = None
        for chunk in self._client.models.generate_content_stream(
            model=model, contents=user, config=self._config(system, temperature, max_tokens, timeout)
        ):
            last = chunk
            if chunk.text:
                yield chunk.text
        if last is not None and on_usage is not None:
            on_usage(self._usage(last))


# ------------------------------------------------------------------------------------- Fake (offline)


class FakeChatClient:
    """Offline client that answers every schema the app uses with content consistent with the prompt.

    ``calls`` records every request for assertions; ``summary_override`` / ``structured_override`` let tests
    force a specific output (e.g. definitive language to exercise the guardrails).
    """

    name = "fake"
    CASE_PATTERN = re.compile(r"\[RADIOMICS_CASE_JSON\]\n(.*?)\n\[END_OF_RADIOMICS_CASE_JSON\]", re.S)
    SUMMARY = (
        "분석 요약(FAKE): 분할된 영역의 질감이 불균질하고 분류기는 결핵 가능성을 {level} 수준으로 평가했습니다. "
        "임상 증상과 객담 검사를 함께 고려하시기 바랍니다. 최종 판단은 담당 의사에게 있습니다."
    )
    PATIENT = (
        "환자용 설명(FAKE): 영상에서 확인이 더 필요한 부분이 보였습니다. 이것만으로 병을 확정할 수는 없으며, "
        "담당 의사와 상의해 추가 검사를 받으시면 됩니다."
    )
    BLOCK_WORDS = ("주식", "비트코인", "로또", "욕설", "해킹", "폭탄")

    def __init__(self, summary_override: str | None = None, structured_override: dict[str, Any] | None = None):
        self.calls: list[dict[str, Any]] = []
        self.summary_override = summary_override
        self.structured_override = structured_override

    def _record(self, kind: str, model: str, user: str, schema: str | None = None) -> None:
        self.calls.append({"kind": kind, "model": model, "user": user, "schema": schema})

    def _case_from_prompt(self, user: str) -> dict[str, Any]:
        match = self.CASE_PATTERN.search(user)
        if not match:
            return {}
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _level(user: str) -> str:
        match = re.search(r'"(?:risk_)?level":\s*"(low|moderate|high)"', user)
        return match.group(1) if match else "moderate"

    def complete(self, system, user, *, model, temperature, max_tokens, timeout):
        self._record("complete", model, user)
        if self.summary_override:
            text = self.summary_override
        elif "[PATIENT_AUDIENCE]" in user or "patient-friendly" in system.lower():
            text = self.PATIENT
        else:
            text = self.SUMMARY.format(level=self._level(user))
        return text, Usage(input_tokens=max(1, len(user) // 4), output_tokens=max(1, len(text) // 2))

    def _structured(self, schema: type[BaseModel], user: str) -> dict[str, Any]:
        name = schema.__name__
        if name == "RadiomicsReport":
            return self._fake_report(user)
        if name == "ModerationDecision":
            question = user.split("[USER_MESSAGE]")[-1]
            blocked = any(w in question for w in self.BLOCK_WORDS)
            return {
                "appropriate": not blocked,
                "category": "off_topic" if blocked else "clinical",
                "reason": "fake rule: blocked keyword" if blocked else "fake rule: clinical question",
            }
        if name == "QuestionClass":
            has_image = "[HAS_IMAGE] true" in user
            has_report = "[HAS_PRIOR_REPORT] true" in user
            text = user.split("[USER_MESSAGE]")[-1].strip().lower()
            if has_image:
                category = "analysis_request"
            elif any(g in text for g in ("안녕", "hello", "hi", "고마")):
                category = "chit_chat"
            elif has_report:
                category = "follow_up"
            else:
                category = "general_medical"
            return {
                "category": category,
                "needs_retrieval": category in {"analysis_request", "follow_up", "general_medical"},
                "search_query": "pulmonary tuberculosis chest radiograph radiomics",
            }
        if name == "ConsistencyCheck":
            return {"consistent": True, "issue": ""}
        if name == "FollowUpQuestions":
            return {
                "doctor": [
                    "객담 도말/배양 검사 결과가 있나요?",
                    "이전 흉부 영상과 비교했을 때 변화가 있나요?",
                    "접촉력이나 결핵 과거력이 있나요?",
                ],
                "patient": ["추가 검사는 어떤 것을 받게 되나요?", "결과가 나올 때까지 주의할 점이 있나요?"],
            }
        if name == "SessionTitle":
            level = self._level(user)
            return {
                "title": {"high": "결핵 의심 영역 분석", "moderate": "병변 영역 정밀 확인", "low": "흉부 영상 검토"}[
                    level
                ]
            }
        raise LLMError(f"FakeChatClient has no canned answer for schema {name}")

    def _fake_report(self, user: str) -> dict[str, Any]:
        from llm_schemas import RadiomicsReport

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
        return report.model_dump(mode="json")

    def complete_structured(self, system, user, *, schema, model, temperature, max_tokens, timeout):
        self._record("structured", model, user, schema.__name__)
        if self.structured_override is not None and schema.__name__ == "RadiomicsReport":
            return dict(self.structured_override), Usage(100, 50)
        payload = self._structured(schema, user)
        return payload, Usage(input_tokens=max(1, len(user) // 4), output_tokens=200)

    def stream(self, system, user, *, model, temperature, max_tokens, timeout, on_usage=None):
        text, usage = self.complete(
            system, user, model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout
        )
        self.calls[-1]["kind"] = "stream"
        for start in range(0, len(text), 12):
            yield text[start : start + 12]
        if on_usage is not None:
            on_usage(usage)


# ------------------------------------------------------------------------------------- router


class LLMRouter:
    """Dispatches calls to a provider by model name: gemini-* -> Gemini, everything else -> OpenAI.

    When a fake client is configured every model routes to it (offline mode).
    """

    name = "router"

    def __init__(
        self,
        openai: ChatClient | None = None,
        gemini: ChatClient | None = None,
        fake: ChatClient | None = None,
    ):
        self.openai = openai
        self.gemini = gemini
        self.fake = fake

    @property
    def providers(self) -> list[str]:
        return [
            name for name, client in (("openai", self.openai), ("gemini", self.gemini), ("fake", self.fake)) if client
        ]

    def client_for(self, model: str) -> ChatClient:
        if self.fake is not None:
            return self.fake
        if model.lower().startswith("gemini"):
            if self.gemini is None:
                raise LLMError(f"model {model} needs the Gemini provider (set GOOGLE_API_KEY)")
            return self.gemini
        if self.openai is None:
            raise LLMError(f"model {model} needs the OpenAI provider (set OPENAI_API_KEY)")
        return self.openai

    def complete(self, system, user, *, model, temperature, max_tokens, timeout):
        return self.client_for(model).complete(
            system, user, model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout
        )

    def complete_structured(self, system, user, *, schema, model, temperature, max_tokens, timeout):
        return self.client_for(model).complete_structured(
            system, user, schema=schema, model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout
        )

    def stream(self, system, user, *, model, temperature, max_tokens, timeout, on_usage=None):
        return self.client_for(model).stream(
            system,
            user,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            on_usage=on_usage,
        )


def build_llm_router(settings: Any) -> LLMRouter:
    if settings.llm_backend == "fake":
        logger.warning("Using FAKE LLM backend; all models route to the offline client.")
        return LLMRouter(fake=FakeChatClient())
    gemini = None
    if settings.google_api_key:
        gemini = GeminiChatClient(settings.google_api_key, thinking_level=settings.gemini_thinking_level)
    return LLMRouter(openai=OpenAIChatClient(settings.openai_api_key), gemini=gemini)
