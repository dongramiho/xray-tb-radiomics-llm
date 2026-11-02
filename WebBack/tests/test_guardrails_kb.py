"""Guardrails, structured-report schema, and knowledge-base retrieval (all offline)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

import guardrails
from kb import HashingEmbedder, KnowledgeBase, case_query, split_into_chunks
from llm_schemas import RadiomicsReport

BACKEND_DIR = Path(__file__).resolve().parents[1]


def test_definitive_diagnosis_language_is_flagged():
    result = guardrails.check_summary("영상 소견상 결핵으로 확진됩니다. 담당 의사와 상의하세요.")
    assert not result.passed
    assert any("definitive_language" in v for v in result.violations)


def test_missing_physician_deferral_gets_disclaimer():
    result = guardrails.check_summary("우상부에 병변 가능성이 있습니다.")
    assert result.passed and result.appended_disclaimer
    assert guardrails.DISCLAIMER_KO in result.text


def test_compliant_summary_untouched():
    text = "결핵 가능성이 높은 소견입니다. 최종 판단은 담당 의사에게 있습니다."
    result = guardrails.check_summary(text)
    assert result.passed and not result.appended_disclaimer and result.text == text


def test_evidence_validation_drops_unknown_keys():
    case = {"ml_results": {"tb_probability": 0.9}, "segmentation": {"components": [{"area_px": 10}]}}
    report = {
        "supporting_evidence": [
            {"json_key": "ml_results.tb_probability"},
            {"json_key": "segmentation.components[0].area_px"},
            {"json_key": "tb_5#1"},
            {"json_key": "radiomics_features.made_up"},
        ]
    }
    out = guardrails.validate_evidence(report, case, {"tb_5#1"})
    assert [e["json_key"] for e in out["supporting_evidence"]] == [
        "ml_results.tb_probability",
        "segmentation.components[0].area_px",
        "tb_5#1",
    ]
    assert out["validation"]["dropped_evidence_keys"] == ["radiomics_features.made_up"]


def test_report_schema_rejects_bad_risk_level():
    with pytest.raises(ValidationError):
        RadiomicsReport.model_validate(
            {"clinical_overview": "x", "tb_risk_summary": {"level": "extreme", "rationale": ""}}
        )


def test_chunking_keeps_headings_with_bodies():
    text = "# Heading\n\nfirst paragraph\n\nsecond paragraph\n\n" + ("x" * 500) + "\n\nlast"
    chunks = split_into_chunks("doc", text, max_chars=120)
    assert chunks[0].id == "doc#0" and chunks[0].text.startswith("# Heading")
    assert all(len(c.text) <= 600 for c in chunks)
    assert "".join(c.text for c in chunks).count("paragraph") == 2


def test_knowledge_base_retrieval_offline():
    kb = KnowledgeBase.build(BACKEND_DIR / "documents", HashingEmbedder(), cache=False)
    assert len(kb.chunks) > 20 and len(kb.chunk_ids) == len(kb.chunks)
    case = {
        "segmentation": {"zones": ["right upper zone"], "num_components": 2},
        "ml_results": {"primary_label": "tb_suspected", "top_features": [{"feature": "original_glcm_Contrast"}]},
    }
    results = kb.search(case_query(case), k=4)
    assert len(results) == 4
    assert all(cid in kb.chunk_ids for cid, _ in ((c.id, s) for c, s in results))
    assert results[0][1] >= results[-1][1]
