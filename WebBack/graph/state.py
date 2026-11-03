"""State carried through one chat turn of the LangGraph pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Reducer for keys written by parallel branches (each branch contributes its own sub-key)."""
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class TurnState(TypedDict, total=False):
    # ---- inputs (set by the API for every turn)
    session_id: int
    patient_id: int
    user_id: int
    request_id: str | None
    text: str
    image_path: str | None  # absolute path of the stored upload, when an image was sent
    mask_path: str | None  # CT only: absolute path of the stored NIfTI label map
    modality: str  # "xray" | "ct"
    image_id: str | None
    audience: str  # "doctor" | "patient" (who receives the final response)
    prior_report: dict[str, Any] | None  # latest structured report in the session (follow-up context)
    session_title_is_default: bool
    stream: bool  # whether token streaming is wanted (tokens are emitted as custom stream events)

    # ---- node outputs
    validation: dict[str, Any]
    moderation: dict[str, Any]
    question: dict[str, Any]
    analysis: dict[str, Any] | None  # {"case": ..., "overlay_path": ..., "mask_path": ...}
    retrieval: Annotated[dict[str, Any], merge_dicts]  # {"kb": [...], "similar_cases": [...], "literature": [...]}
    report: dict[str, Any] | None
    doctor_summary: str
    patient_summary: str
    consistency: dict[str, Any]
    safety: dict[str, Any]
    follow_ups: dict[str, list[str]]
    title: str | None
    response: str
    blocked: bool
    traces: Annotated[list[dict[str, Any]], operator.add]
    status_log: Annotated[list[str], operator.add]
