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
