"""Run the two LLM stages on the fixed evaluation cases and score them with rule-based checks.

Checks (per case):
  schema        stage-1 output validates against llm_schemas.RadiomicsReport
  risk_consistent  report risk level equals the pipeline risk level or is one step away
  no_lesion_handled for no_lesion_detected cases: num_regions == 0 and tb_probability is null
  evidence      >= 2 evidence items whose json_key points at a real case path / knowledge-base chunk
  grounded_zones  zones mentioned in the report are a subset of the segmentation zones
  recommendations  at least one clinician recommendation
  summary_guardrails stage-2 summary has no definitive-diagnosis language
  summary_physician  stage-2 summary defers to the physician
  summary_korean     stage-2 summary is Korean
  patient_guardrails / patient_plain_language  patient explanation defers to the physician, no jargon

LLM-as-judge metrics (``--judge``, default on; ``JUDGE_MODEL`` setting):
  faithfulness (1-5), answer_relevance (1-5, follow-up answer), understandability (1-5, patient text),
  role_separation (pass), safety (pass), no_lesion_handling (pass, no-lesion cases),
  qc_handling (pass, QC-flagged cases)

Usage:
    OPENAI_API_KEY=... python evals/run_evals.py                 # real models from settings
    LLM_BACKEND=fake python evals/run_evals.py                   # plumbing check, no network
    python evals/run_evals.py --report_model gpt-4.1-mini        # compare a cheaper model
    python evals/run_evals.py --no-judge                         # rule-based checks only
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import guardrails  # noqa: E402
import llm_report  # noqa: E402
from evals.judge import aggregate, judge_case  # noqa: E402
from graph import llm_steps  # noqa: E402
from llm_schemas import RadiomicsReport  # noqa: E402
from settings import Settings  # noqa: E402

CASES_DIR = BACKEND_DIR / "evals" / "cases"
RISK_ORDER = {"low": 0, "moderate": 1, "high": 2}


def load_cases() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CASES_DIR.glob("*.json"))]


def check_case(
    case: dict, report: dict, summary: str, chunk_ids: set[str], patient_summary: str | None = None
) -> dict[str, bool]:
    ml = case["ml_results"]
    seg = case["segmentation"]
    checks: dict[str, bool] = {}
    try:
        RadiomicsReport.model_validate(report)
        checks["schema"] = True
    except Exception:
        checks["schema"] = False
    level = report.get("tb_risk_summary", {}).get("level")
    checks["risk_consistent"] = level in RISK_ORDER and abs(RISK_ORDER[level] - RISK_ORDER[ml["risk_level"]]) <= 1
    if ml["primary_label"] == "no_lesion_detected":
        checks["no_lesion_handled"] = (
            report.get("lesion_summary", {}).get("num_regions") == 0
            and report.get("ml_summary", {}).get("tb_probability") is None
        )
    validated = guardrails.validate_evidence(report, case, chunk_ids)
    checks["evidence"] = len(validated.get("supporting_evidence", [])) >= 2
    reported_zones = set(report.get("lesion_summary", {}).get("zones", []) or [])
    checks["grounded_zones"] = reported_zones <= set(seg.get("zones", []))
    checks["recommendations"] = len(report.get("recommendations_for_clinician", []) or []) >= 1
    guard = guardrails.check_summary(summary)
    checks["summary_guardrails"] = guard.passed
    checks["summary_physician"] = not guard.appended_disclaimer
    checks["summary_korean"] = bool(re.search(r"[가-힣]", summary))
    if patient_summary is not None:
        patient_guard = guardrails.check_summary(patient_summary)
        checks["patient_guardrails"] = patient_guard.passed and not patient_guard.appended_disclaimer
        checks["patient_plain_language"] = not any(
            term.lower() in patient_summary.lower() for term in llm_steps.PATIENT_JARGON
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_model", default=None)
    parser.add_argument("--summary_model", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--judge", dest="judge", action="store_true", default=True)
    parser.add_argument("--no-judge", dest="judge", action="store_false")
    parser.add_argument("--judge_model", default=None)
    parser.add_argument(
        "--question", default="추가 검사는 어떤 것을 받아야 하나요?", help="Follow-up asked to every case"
    )
    parser.add_argument("--out", type=Path, default=BACKEND_DIR / "evals" / "results.json")
    args = parser.parse_args()

    settings = Settings()
    report_model = args.report_model or settings.report_model
    summary_model = args.summary_model or settings.summary_model
    judge_model = args.judge_model or settings.judge_model
    client = llm_report.build_chat_client(settings)
    kb = llm_report.build_knowledge_base(settings)
    model_meta = {"note": "evaluation run", "cross_validation": {"roc_auc_mean": 0.891}}

    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]
    rows = []
    for item in cases:
        case = item["case"]
        t0 = time.perf_counter()
        report, trace1 = llm_report.generate_radiomics_report(
            case, client=client, kb=kb, model_meta=model_meta, model=report_model
        )
        summary, trace2 = llm_report.summarize_for_clinician(report, client=client, model=summary_model)
        patient_summary, trace3 = llm_steps.explain_for_patient(
            report, client=client, model=summary_model, fallback_model=None, doctor_summary=summary
        )
        answer, trace4 = llm_steps.answer_with_context(
            args.question,
            client=client,
            model=summary_model,
            fallback_model=None,
            context_report=report,
            retrieval=None,
        )
        latency = time.perf_counter() - t0
        checks = check_case(case, report, summary, kb.chunk_ids, patient_summary)
        judgment = None
        if args.judge:
            judgment = judge_case(
                case,
                report,
                summary,
                patient_summary,
                client=client,
                model=judge_model,
                question=args.question,
                answer=answer,
            )
        rows.append(
            {
                "image_id": item["image_id"],
                "label": item["label"],
                "primary_label": case["ml_results"]["primary_label"],
                "checks": checks,
                "passed": all(checks.values()),
                "latency_s": round(latency, 2),
                "tokens": {
                    "report": [trace1.input_tokens, trace1.output_tokens],
                    "summary": [trace2.input_tokens, trace2.output_tokens],
                },
                "cost_usd": round(
                    (trace1.cost_usd or 0) + (trace2.cost_usd or 0) + (trace3.cost_usd or 0) + (trace4.cost_usd or 0), 6
                ),
                "models": [trace1.model, trace2.model],
                "report_level": report.get("tb_risk_summary", {}).get("level"),
                "qc_flags": (case.get("qc") or {}).get("flags", []),
                "judge": judgment,
            }
        )
        status = "PASS" if rows[-1]["passed"] else "FAIL " + ",".join(k for k, v in checks.items() if not v)
        judge_note = ""
        if judgment:
            judge_note = f" judge: faith={judgment['faithfulness']['score']} safe={judgment['safety']['compliant']}"
        print(f"{item['image_id']}: {status} ({latency:.1f}s){judge_note}")

    summary_out = {
        "report_model": report_model,
        "summary_model": summary_model,
        "llm_backend": settings.llm_backend,
        "cases": len(rows),
        "pass_rate": round(sum(r["passed"] for r in rows) / max(len(rows), 1), 3),
        "check_pass_rates": {
            name: round(sum(r["checks"].get(name, True) for r in rows) / len(rows), 3)
            for name in sorted({k for r in rows for k in r["checks"]})
        },
        "latency_s_median": round(statistics.median(r["latency_s"] for r in rows), 2) if rows else None,
        "judge_model": judge_model if args.judge else None,
        "judge": aggregate([r["judge"] for r in rows if r.get("judge")]) if args.judge else None,
        "rows": rows,
    }
    args.out.write_text(json.dumps(summary_out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary_out.items() if k != "rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
