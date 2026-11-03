"""Stage 6: quality control, the CT mask-ingestion path, and the LLM-as-judge evaluation plumbing."""

import json
from pathlib import Path

import numpy as np
from cxr_tb.src.ct import CTPipeline, make_synthetic_ct
from cxr_tb.src.pipeline import run_qc, summarize_lesions

from evals.judge import aggregate, judge_case
from llm_clients import FakeChatClient


def blob(cx, cy, r, size=512):
    yy, xx = np.mgrid[0:size, 0:size]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) < r**2


def test_qc_passes_a_normal_analysis():
    gray = np.random.default_rng(0).integers(40, 200, (3000, 2900)).astype(np.uint8)
    mask = blob(150, 140, 45)
    qc = run_qc(gray, summarize_lesions(mask, np.where(mask, 0.9, 0.05).astype(np.float32)), {"a": 1.0})
    assert qc.passed and qc.usable_for_patient and qc.flags == []


def test_qc_flags_tiny_low_confidence_and_nan():
    gray = np.random.default_rng(0).integers(40, 200, (3000, 2900)).astype(np.uint8)
    mask = blob(150, 140, 14)  # ~0.2% of the image: above the detection gate, below the QC size threshold
    qc = run_qc(gray, summarize_lesions(mask, np.where(mask, 0.55, 0.05).astype(np.float32)), {"a": float("nan")})
    assert set(qc.flags) == {"tiny_roi", "low_confidence", "nan_features"}
    assert not qc.passed and not qc.usable_for_patient
    assert len(qc.to_dict()["flag_descriptions"]) == 3


def test_qc_flags_degraded_input_without_blocking_patient_logic_errors():
    qc = run_qc(np.full((100, 120), 128, dtype=np.uint8), summarize_lesions(np.zeros((512, 512), dtype=bool)), None)
    assert {"image_too_small", "low_contrast"} <= set(qc.flags) and not qc.passed


def test_ct_pipeline_on_synthetic_phantom(tmp_path):
    vol, mask = make_synthetic_ct(tmp_path / "ct")
    result = CTPipeline().analyze(vol, mask, image_id="phantom")
    case = result.to_case_json()
    assert case["modality"] == "ct" and case["qc"]["passed"] and case["qc"]["flags"] == []
    tumor = case["segmentation"]["tumor"]
    assert tumor["voxels"] > 100 and tumor["volume_ml"] > 0 and tumor["max_extent_mm"] > 0
    assert case["segmentation"]["labels_present"] == ["liver", "tumor"]
    assert (
        case["ml_results"]["primary_label"] == "ct_lesion_characterized"
        and case["ml_results"]["tb_probability"] is None
    )
    feats = case["radiomics_features"]
    assert any(k.startswith("original_shape_") for k in feats) and any(k.startswith("original_glszm_") for k in feats)
    assert result.overlay_image().size == (64, 64) and 0 <= result.overlay_slice < 48


def test_ct_missing_tumor_label_is_not_no_tumor(tmp_path):
    vol, mask = make_synthetic_ct(tmp_path / "ct", with_tumor=False)
    case = CTPipeline().analyze(vol, mask, image_id="nolabel").to_case_json()
    assert case["ml_results"]["primary_label"] == "ct_tumor_label_missing"
    assert "tumor_label_missing" in case["qc"]["flags"] and not case["qc"]["usable_for_patient"]
    assert "not that there is no tumor" in case["ml_results"]["decision_basis"]
    assert case["radiomics_features"] is None


def test_api_ct_upload_runs_the_ct_path(client, tmp_path):
    pid = client.post("/patients/register", json={"name": "CT환자", "birthDate": "1970-01-01", "gender": "M"}).json()[
        "data"
    ]["id"]
    vol, mask = make_synthetic_ct(tmp_path / "ct")
    with vol.open("rb") as v, mask.open("rb") as m:
        res = client.post(
            "/analyze",
            data={"patient_id": pid, "text": "간 종양 의심"},
            files={
                "file": ("ct_volume.nii.gz", v, "application/gzip"),
                "mask": ("ct_mask.nii.gz", m, "application/gzip"),
            },
        )
    assert res.status_code == 200, res.text
    analysis = res.json()["analysis"]
    assert analysis["modality"] == "ct" and analysis["qc"]["passed"]
    assert analysis["segmentation"]["tumor"]["voxels"] > 0 and analysis["mask_url"] is None and analysis["overlay_url"]
    assert analysis["ml_results"]["primary_label"] == "ct_lesion_characterized"
    assert analysis["report"]["evidence_sources"]["similar_cases"] == []  # no reference cases for CT
    assert res.json()["meta"]["retrieval"]["similar_cases"] == []


def test_api_rejects_ct_volume_without_mask_and_bad_files(client, tmp_path):
    pid = client.post("/patients/register", json={"name": "CT거부", "birthDate": "1970-01-01", "gender": "M"}).json()[
        "data"
    ]["id"]
    vol, _ = make_synthetic_ct(tmp_path / "ct")
    with vol.open("rb") as v:
        assert (
            client.post(
                "/analyze", data={"patient_id": pid}, files={"file": ("ct_volume.nii.gz", v, "application/gzip")}
            ).status_code
            == 400
        )
    with vol.open("rb") as v:
        res = client.post(
            "/analyze",
            data={"patient_id": pid},
            files={
                "file": ("ct_volume.nii.gz", v, "application/gzip"),
                "mask": ("mask.nii.gz", b"not nifti", "application/gzip"),
            },
        )
    assert res.status_code == 400


def test_judge_case_and_aggregate_offline():
    case = json.loads((Path(__file__).resolve().parents[1] / "evals" / "cases" / "CHNCXR_0345_1.json").read_text())[
        "case"
    ]
    client = FakeChatClient()
    report = {"tb_risk_summary": {"level": "high"}, "clinical_overview": "x"}
    good = judge_case(
        case,
        report,
        "분할된 영역의 질감이 불균질합니다. 최종 판단은 담당 의사에게 있습니다.",
        "환자용: 담당 의사와 상의하세요.",
        client=client,
        model="judge-model",
        question="추가 검사는?",
        answer="객담 검사 등을 고려합니다. 담당 의사와 상의하세요.",
    )
    assert good["faithfulness"]["score"] == 4 and good["safety"]["compliant"] and good["role_separation"]["passed"]
    assert good["answer_relevance"]["score"] == 4 and good["usage"]["judge_model"] == "judge-model"
    bad = judge_case(
        case, report, "결핵입니다. 확진.", "환자용: radiomics GLCM 수치가 높습니다.", client=client, model="judge-model"
    )
    assert (
        not bad["safety"]["compliant"]
        and bad["faithfulness"]["score"] == 2
        and bad["understandability"]["jargon_terms"]
    )
    summary = aggregate([good, bad])
    assert (
        summary["faithfulness_mean"] == 3.0
        and summary["safety_pass_rate"] == 0.5
        and summary["answer_relevance_n"] == 1
    )


def test_judge_runs_no_lesion_and_qc_checks_when_applicable():
    client = FakeChatClient()
    case = {"ml_results": {"primary_label": "no_lesion_detected"}, "qc": {"flags": ["tiny_roi"]}, "segmentation": {}}
    out = judge_case(
        case,
        {"tb_risk_summary": {"level": "low"}},
        "병변 영역이 분할되지 않았습니다. 담당 의사의 검토가 필요합니다.",
        "환자용",
        client=client,
        model="m",
    )
    assert out["no_lesion_handling"]["passed"] and out["qc_handling"]["passed"]
