"""Output guardrails for clinician-facing text and report evidence.

The pipeline is decision *support*: summaries must not read as a definitive diagnosis, and every
evidence item in the structured report must point at something that exists (a case-JSON path or
a knowledge-base chunk id).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Phrases that assert a definitive diagnosis. Matched case-insensitively on the summary text.
DEFINITIVE_PATTERNS: tuple[str, ...] = (
    r"결핵\s*(으로|이라고)?\s*확진",
    r"확진\s*(입니다|되었습니다|됩니다)",
    r"결핵\s*(입니다|이\s*확실|이\s*분명)",
    r"틀림없이\s*결핵",
    r"정상\s*(입니다|이\s*확실)",
    r"\b(definitely|certainly|confirmed)\s+(tb|tuberculosis)\b",
    r"\bdiagnosis is (tb|tuberculosis)\b",
    r"\bdefinitive diagnosis\b",
)
PHYSICIAN_PATTERNS: tuple[str, ...] = (r"담당\s*의", r"의사", r"전문의", r"임상의", r"physician", r"clinician")

DISCLAIMER_KO = "이 분석은 연구용 보조 도구의 출력이며, 최종 임상 판단은 담당 의사에게 있습니다."


@dataclass
class GuardrailResult:
    text: str
    violations: list[str] = field(default_factory=list)
    appended_disclaimer: bool = False

    @property
    def passed(self) -> bool:
        return not self.violations


def check_summary(text: str) -> GuardrailResult:
    """Flag definitive-diagnosis language and make sure the physician-judgment reminder is present."""
    violations = [p for p in DEFINITIVE_PATTERNS if re.search(p, text, flags=re.IGNORECASE)]
    result = GuardrailResult(text=text, violations=[f"definitive_language:{p}" for p in violations])
    if not any(re.search(p, text, flags=re.IGNORECASE) for p in PHYSICIAN_PATTERNS):
        result.text = text.rstrip() + "\n\n" + DISCLAIMER_KO
        result.appended_disclaimer = True
    return result


def _flatten_keys(node: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.add(path)
            keys |= _flatten_keys(value, path)
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            path = f"{prefix}[{idx}]"
            keys.add(path)
            keys |= _flatten_keys(value, path)
    return keys


def validate_evidence(report: dict[str, Any], case_json: dict[str, Any], chunk_ids: set[str]) -> dict[str, Any]:
    """Drop evidence items whose json_key is neither a case-JSON path nor a known knowledge-base chunk id."""
    valid_paths = _flatten_keys(case_json)
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for item in report.get("supporting_evidence", []):
        key = str(item.get("json_key", ""))
        normalized = re.sub(r"\[\d+\]", "", key)
        if key in valid_paths or normalized in valid_paths or key in chunk_ids:
            kept.append(item)
        else:
            dropped.append(key)
    out = dict(report)
    out["supporting_evidence"] = kept
    if dropped:
        out.setdefault("validation", {})["dropped_evidence_keys"] = dropped
    return out
