"""Stage-3 API behaviour: streaming, sessions, feedback, telemetry, and the LLM resilience helpers."""

import json

import httpx
import openai
import pytest

import llm_report


def register(client, name, birth="1990-01-01"):
    return client.post("/patients/register", json={"name": name, "birthDate": birth, "gender": "M"}).json()["data"][
        "id"
    ]


def parse_sse(body: str):
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.strip().splitlines()
        if not lines:
            continue
        name = lines[0].removeprefix("event: ")
        data = json.loads("\n".join(line.removeprefix("data: ") for line in lines[1:]))
        events.append((name, data))
    return events


def test_analyze_returns_llm_meta_and_records_calls(client, sample_png):
    pid = register(client, "텔레메트리")
    with sample_png.open("rb") as fh:
        body = client.post("/analyze", data={"patient_id": pid}, files={"file": ("cxr.png", fh, "image/png")}).json()
    llm = body["analysis"]["llm"]
    assert llm["report_model"] == "gpt-4.1" and llm["summary_model"] == "gpt-4.1-mini"
    assert llm["input_tokens"] > 0 and llm["cost_usd"] is not None and llm["prompt_version"]
    report = body["analysis"]["report"]
    assert report["retrieved_chunks"] and all(e["json_key"] for e in report["supporting_evidence"])
    assert any(e["category"] == "knowledge_base" for e in report["supporting_evidence"])

    metrics = client.get("/metrics/llm").json()
    stages = {row["stage"] for row in metrics["by_stage_model"]}
    assert {"report", "summary"} <= stages
    assert metrics["prompt_version"] == llm_report.PROMPT_VERSION


def test_streaming_analysis_emits_ordered_events_and_persists(client, sample_png):
    pid = register(client, "스트리밍")
    with sample_png.open("rb") as fh:
        with client.stream(
            "POST",
            "/analyze/stream",
            data={"patient_id": pid, "text": "기침"},
            files={"file": ("cxr.png", fh, "image/png")},
        ) as res:
            assert res.status_code == 200
            assert res.headers["content-type"].startswith("text/event-stream")
            events = parse_sse(res.read().decode("utf-8"))
    names = [name for name, _ in events]
    assert names[0] == "status" and "analysis" in names and names[-1] == "done"
    assert names.index("analysis") < names.index("delta") < names.index("done")
    deltas = "".join(data["text"] for name, data in events if name == "delta")
    done = events[-1][1]["message"]
    assert done["role"] == "assistant" and done["content"].startswith(deltas[:20])
    assert done["analysis"]["overlay_url"] and done["analysis"]["llm"]["summary_model"] == "gpt-4.1-mini"
    history = client.get(f"/patients/{pid}/messages").json()
    assert [m["role"] for m in history] == ["user", "assistant"] and history[1]["id"] == done["id"]


def test_streaming_follow_up_without_image(client):
    pid = register(client, "스트리밍질문")
    with client.stream("POST", "/analyze/stream", data={"patient_id": pid, "text": "추가 검사는?"}) as res:
        events = parse_sse(res.read().decode("utf-8"))
    names = [name for name, _ in events]
    assert names[0] == "status" and events[0][1]["stage"] == "answering"
    assert "analysis" not in names and names[-1] == "done"
    assert "analysis" not in events[-1][1]["message"]


def test_sessions_are_created_listed_and_used(client, sample_png):
    pid = register(client, "세션환자")
    assert client.get(f"/patients/{pid}/sessions").json() == []
    first = client.post(f"/patients/{pid}/sessions", json={"title": "1차 촬영"}).json()
    second = client.post(f"/patients/{pid}/sessions", json={}).json()
    assert first["title"] == "1차 촬영" and second["title"].startswith("상담 ")
    client.post("/analyze", data={"patient_id": pid, "text": "안녕", "session_id": first["id"]})
    sessions = client.get(f"/patients/{pid}/sessions").json()
    assert [s["id"] for s in sessions] == [second["id"], first["id"]]
    assert next(s for s in sessions if s["id"] == first["id"])["message_count"] == 2
    assert len(client.get(f"/patients/{pid}/messages", params={"session_id": first["id"]}).json()) == 2
    assert client.get(f"/patients/{pid}/messages", params={"session_id": second["id"]}).json() == []
    assert client.post("/analyze", data={"patient_id": pid, "text": "x", "session_id": 999999}).status_code == 404


def test_feedback_roundtrip_and_scoping(client, anon_client, sample_png):
    pid = register(client, "피드백환자")
    with sample_png.open("rb") as fh:
        body = client.post("/analyze", data={"patient_id": pid}, files={"file": ("cxr.png", fh, "image/png")}).json()
    did = body["analysis"]["diagnosis_id"]
    res = client.post(f"/diagnoses/{did}/feedback", json={"agree": False, "note": "경계가 불명확함"})
    assert res.status_code == 200 and res.json() == {
        "diagnosis_id": did,
        "doctor_feedback": False,
        "doctor_note": "경계가 불명확함",
    }
    detail = client.get(f"/diagnoses/{did}").json()
    assert detail["doctor_feedback"] is False and detail["case"]["ml_results"]["primary_label"] == "tb_suspected"
    history = client.get(f"/patients/{pid}/messages").json()
    assert history[-1]["analysis"]["doctor_note"] == "경계가 불명확함"
    assert anon_client.post(f"/diagnoses/{did}/feedback", json={"agree": True}).status_code == 401
    assert client.get("/diagnoses/999999").status_code == 404
    assert client.get("/metrics/llm").json()["doctor_feedback"].get("False", 0) >= 1


def test_guardrail_regeneration_on_definitive_language():
    bad = llm_report.FakeChatClient(summary_override="영상 소견상 결핵으로 확진됩니다.")
    text, trace = llm_report.summarize_for_clinician(
        {"tb_risk_summary": {"level": "high"}}, client=bad, model="gpt-4.1-mini"
    )
    assert trace.regenerated and trace.guardrail_violations
    assert llm_report.guardrails.DISCLAIMER_KO in text
    assert len(bad.calls) == 2 and "[REVISION_REQUEST]" in bad.calls[1]["user"]


class FlakyClient:
    name = "flaky"

    def __init__(self, failures: int, exc_factory):
        self.failures = failures
        self.exc_factory = exc_factory
        self.calls: list[str] = []

    def complete(self, system, user, *, model, temperature, max_tokens, timeout):
        self.calls.append(model)
        if len(self.calls) <= self.failures:
            raise self.exc_factory()
        return "응답입니다. 최종 판단은 담당 의사에게 있습니다.", llm_report.Usage(10, 5)


def rate_limit():
    return openai.RateLimitError(
        "rl", response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=None
    )


def test_retries_then_fallback_model():
    flaky = FlakyClient(failures=2, exc_factory=rate_limit)
    text, trace = llm_report.answer_clinician_question(
        "q", client=flaky, model="gpt-4.1-mini", fallback_model="gpt-4.1", max_retries=1
    )
    assert flaky.calls == ["gpt-4.1-mini", "gpt-4.1-mini", "gpt-4.1"]
    assert trace.fallback_used and trace.attempts == 3 and trace.success and "담당 의사" in text


def test_non_retryable_error_fails_fast():
    def bad_request():
        return openai.BadRequestError(
            "bad", response=httpx.Response(400, request=httpx.Request("POST", "http://x")), body=None
        )

    flaky = FlakyClient(failures=5, exc_factory=bad_request)
    with pytest.raises(llm_report.LLMError):
        llm_report.answer_clinician_question(
            "q", client=flaky, model="gpt-4.1-mini", fallback_model="gpt-4.1", max_retries=2
        )
    assert flaky.calls == ["gpt-4.1-mini", "gpt-4.1"]  # no retries on 400, one try per model


def test_cost_estimate_known_and_unknown_models():
    from llm_pricing import estimate_cost_usd

    assert estimate_cost_usd("gpt-4.1", 1_000_000, 0) == 2.0
    assert estimate_cost_usd("gpt-4.1-mini-2025-04-14", 0, 1_000_000) == 1.6
    assert estimate_cost_usd("mystery-model", 10, 10) is None
