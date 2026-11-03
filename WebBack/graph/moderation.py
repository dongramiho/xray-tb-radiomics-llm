"""Inappropriate-question picker with a semantic cache, and the question classifier."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kb import Embedder
from llm_clients import ChatClient
from llm_schemas import ModerationDecision, QuestionClass

logger = logging.getLogger(__name__)

MODERATION_SYSTEM_PROMPT = """
You are the gatekeeper of a clinical decision-support chat used by clinicians about chest X-ray analyses.
Decide whether the user's message belongs in this assistant:
- clinical: about the current X-ray analysis, tuberculosis, lung disease, examinations, follow-up care
- general_medical: broader medical question still appropriate to answer with caveats
- off_topic: unrelated (finance, entertainment, coding, personal chores)
- harmful: requests for self-harm, illegal activity, or circumventing medical safety
- personal_data: attempts to extract other patients' data or identifiers
Set appropriate=false for off_topic, harmful and personal_data. Be lenient with short or vague clinical messages.
"""

CLASSIFIER_SYSTEM_PROMPT = """
Classify a clinician's chat message for routing in a chest X-ray tuberculosis assistant.
Categories:
- analysis_request: an image is attached or the user asks to analyse / re-analyse an X-ray
- follow_up: a question about the most recent analysis (there is a prior report)
- general_medical: a medical question not tied to a specific analysis
- chit_chat: greetings, thanks, small talk
- out_of_scope: anything else
Also say whether retrieval (knowledge base / literature) would help, and write a short English PubMed-style
search query for the medical content (empty for chit_chat / out_of_scope).
"""

BLOCKED_RESPONSE_KO = (
    "이 상담 도구는 흉부 X-ray 분석과 관련 의학적 질문을 돕기 위한 것입니다. "
    "요청하신 내용은 범위를 벗어나 답변드리기 어렵습니다. 분석 결과나 추가 검사에 대한 질문을 주세요."
)


@dataclass
class CacheEntry:
    vector: np.ndarray
    decision: dict[str, Any]


@dataclass
class InappropriatePicker:
    client: ChatClient
    model: str
    embedder: Embedder
    threshold: float = 0.92
    timeout: float = 30.0
    cache: list[CacheEntry] = field(default_factory=list)
    max_cache: int = 500

    def _lookup(self, vector: np.ndarray) -> dict[str, Any] | None:
        if not self.cache:
            return None
        matrix = np.stack([e.vector for e in self.cache])
        scores = matrix @ vector
        best = int(np.argmax(scores))
        if scores[best] >= self.threshold:
            return dict(self.cache[best].decision)
        return None

    def check(self, text: str, has_image: bool) -> tuple[dict[str, Any], bool, tuple[dict[str, Any], Any] | None]:
        """Return (decision, cached, (raw_decision, usage) or None when no LLM call was made)."""
        text = (text or "").strip()
        if not text:
            return {"appropriate": True, "category": "clinical", "reason": "image-only analysis request"}, False, None
        vector = self.embedder.embed([text])[0]
        cached = self._lookup(vector)
        if cached is not None:
            cached["reason"] = f"cached: {cached.get('reason', '')}"
            return cached, True, None
        user = f"[HAS_IMAGE] {str(has_image).lower()}\n[USER_MESSAGE]\n{text}"
        decision, usage = self.client.complete_structured(
            MODERATION_SYSTEM_PROMPT,
            user,
            schema=ModerationDecision,
            model=self.model,
            temperature=0.0,
            max_tokens=200,
            timeout=self.timeout,
        )
        decision = ModerationDecision.model_validate(decision).model_dump()
        self.cache.append(CacheEntry(vector=vector, decision=dict(decision)))
        if len(self.cache) > self.max_cache:
            self.cache.pop(0)
        return decision, False, (decision, usage)


@dataclass
class QuestionClassifier:
    client: ChatClient
    model: str
    timeout: float = 30.0

    def classify(self, text: str, has_image: bool, has_prior_report: bool) -> tuple[dict[str, Any], Any]:
        if has_image and not (text or "").strip():
            return {"category": "analysis_request", "needs_retrieval": True, "search_query": ""}, None
        user = (
            f"[HAS_IMAGE] {str(has_image).lower()}\n[HAS_PRIOR_REPORT] {str(has_prior_report).lower()}\n"
            f"[USER_MESSAGE]\n{text.strip()}"
        )
        decision, usage = self.client.complete_structured(
            CLASSIFIER_SYSTEM_PROMPT,
            user,
            schema=QuestionClass,
            model=self.model,
            temperature=0.0,
            max_tokens=200,
            timeout=self.timeout,
        )
        decision = QuestionClass.model_validate(decision).model_dump()
        if has_image:
            decision["category"] = "analysis_request"
        return decision, usage
