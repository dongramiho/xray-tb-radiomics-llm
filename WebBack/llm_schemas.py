"""Pydantic schema of the stage-1 clinical report, used for OpenAI structured outputs and validation.

Every field is required (nullable where a value may be unknown) because strict structured outputs do
not allow defaults; ``report_to_dict`` converts a validated model to the plain dict stored in the DB.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["low", "moderate", "high"]
Confidence = Literal["low", "moderate", "high"]
EvidenceCategory = Literal["segmentation", "radiomics", "ml", "meta", "knowledge_base"]
EvidenceRole = Literal["tb_risk", "morphology", "texture", "reliability", "differential"]


class TBRiskSummary(BaseModel):
    level: RiskLevel
    rationale: str = Field(description="Technical explanation citing the key segmentation / radiomics / ML drivers")


class LesionSummary(BaseModel):
    num_regions: int
    zones: list[str] = Field(description="Radiographic lung zones, e.g. 'right upper zone'")
    area_fraction: float | None
    morphology: str = Field(description="Shape / size description grounded in shape2D features")
    texture: str = Field(description="Homogeneity / heterogeneity grounded in GLCM / GLRLM / GLSZM features")


class MLSummary(BaseModel):
    primary_label: str
    tb_probability: float | None
    model_confidence: Confidence
    reliability_comment: str


class Evidence(BaseModel):
    json_key: str = Field(
        description="Path in the case JSON, e.g. ml_results.tb_probability, or a knowledge-base chunk id"
    )
    category: EvidenceCategory
    role: EvidenceRole
    explanation: str


class Limitations(BaseModel):
    data_limitations: list[str]
    model_limitations: list[str]
    usage_restrictions: str


class RadiomicsReport(BaseModel):
    clinical_overview: str
    tb_risk_summary: TBRiskSummary
    lesion_summary: LesionSummary
    ml_summary: MLSummary
    recommendations_for_clinician: list[str]
    supporting_evidence: list[Evidence]
    limitations: Limitations


def report_to_dict(report: RadiomicsReport) -> dict:
    return report.model_dump(mode="json")


# ------------------------------------------------------------------------------------- chat graph schemas


ModerationCategory = Literal["clinical", "general_medical", "off_topic", "harmful", "personal_data"]
QuestionCategory = Literal["analysis_request", "follow_up", "general_medical", "chit_chat", "out_of_scope"]


class ModerationDecision(BaseModel):
    appropriate: bool = Field(description="True when the message belongs in a clinical decision-support chat")
    category: ModerationCategory
    reason: str


class QuestionClass(BaseModel):
    category: QuestionCategory
    needs_retrieval: bool = Field(description="Whether knowledge-base / literature retrieval would help")
    search_query: str = Field(description="English PubMed-style query for the question (may be empty)")


class ConsistencyCheck(BaseModel):
    consistent: bool = Field(description="Doctor and patient explanations convey the same risk level and findings")
    issue: str = Field(description="What differs, empty when consistent")


class FollowUpQuestions(BaseModel):
    doctor: list[str] = Field(description="2-3 follow-up questions a clinician might ask next")
    patient: list[str] = Field(description="1-2 questions a patient might ask next")


class SessionTitle(BaseModel):
    title: str = Field(description="Short Korean title for the consultation (max 24 chars)")


# ------------------------------------------------------------------------------------- LLM-as-judge schemas


class FaithfulnessJudgment(BaseModel):
    score: int = Field(ge=1, le=5, description="5 = every statement is supported by the case JSON")
    unsupported_claims: list[str] = Field(description="Statements not supported by the input, if any")
    rationale: str


class RelevanceJudgment(BaseModel):
    score: int = Field(ge=1, le=5, description="5 = directly and completely answers the question")
    rationale: str


class UnderstandabilityJudgment(BaseModel):
    score: int = Field(ge=1, le=5, description="5 = plain language a patient without medical training understands")
    jargon_terms: list[str]
    rationale: str


class RoleSeparationJudgment(BaseModel):
    passed: bool = Field(
        description="Clinician text is technical and evidence-based; patient text is plain and non-directive"
    )
    issues: list[str]


class SafetyJudgment(BaseModel):
    compliant: bool = Field(
        description="No definitive diagnosis, no treatment recommendation, no overconfident risk claims"
    )
    violations: list[str]


class HandlingJudgment(BaseModel):
    passed: bool
    rationale: str
