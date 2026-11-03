"""LangGraph chat pipeline: routing, moderation cache, retrievers, and the API surface built on it."""

import json

import pytest

from graph.retrievers import PubMedRetriever, SimilarCaseIndex, literature_query, parse_pubmed_xml


def register(client, name, birth="1990-01-01"):
    return client.post("/patients/register", json={"name": name, "birthDate": birth, "gender": "M"}).json()["data"][
        "id"
    ]


def parse_sse(body: str):
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.strip().splitlines()
        if lines:
            events.append(
                (lines[0].removeprefix("event: "), json.loads("\n".join(ln.removeprefix("data: ") for ln in lines[1:])))
            )
    return events


def test_health_lists_graph_nodes_and_fallbacks(client):
    body = client.get("/health").json()
    assert {
        "validate",
        "moderate",
        "classify",
        "analyze",
        "retrieve_kb",
        "retrieve_similar",
        "retrieve_literature",
    } <= set(body["graph_nodes"])
    assert {
        "generate_report",
        "map_evidence",
        "explain_doctor",
        "explain_patient",
        "compare",
        "safety_check",
        "enrich",
    } <= set(body["graph_nodes"])
    assert body["fallback_models"]["report"] and body["llm_providers"] == ["fake"]
    assert "graph TD" in client.get("/graph").json()["mermaid"]


def test_image_turn_runs_full_graph(client, sample_png):
    pid = register(client, "그래프환자")
    sessions_before = client.get(f"/patients/{pid}/sessions").json()
    assert sessions_before == []
    with sample_png.open("rb") as fh:
        body = client.post(
            "/analyze", data={"patient_id": pid, "text": "기침 2개월"}, files={"file": ("cxr.png", fh, "image/png")}
        ).json()
    assert body["role"] == "assistant" and body["content"]
    meta = body["meta"]
    assert meta["question_category"] == "analysis_request" and not meta["blocked"]
    assert len(meta["follow_ups"]) >= 2 and meta["title"]
    assert (
        meta["retrieval"]["kb"]
        and len(meta["retrieval"]["similar_cases"]) == 3
        and meta["retrieval"]["literature"] == []
    )
    assert "validating" in meta["graph_stages"] and "finalizing" in meta["graph_stages"]
    analysis = body["analysis"]
    assert analysis["patient_summary"] and "담당 의사" in analysis["patient_summary"]
    assert analysis["report"]["evidence_sources"]["similar_cases"] == meta["retrieval"]["similar_cases"]
    stages = {t for t in [row["stage"] for row in client.get("/metrics/llm").json()["by_stage_model"]]}
    assert {
        "moderation",
        "classification",
        "report",
        "summary",
        "patient_explanation",
        "consistency_check",
        "follow_ups",
        "title",
    } <= stages
    # the auto-generated title replaced the default session title
    sessions = client.get(f"/patients/{pid}/sessions").json()
    assert sessions[0]["title"] == meta["title"] and not sessions[0]["title"].startswith("상담 ")


def test_streaming_turn_emits_analysis_before_tokens_and_meta_in_done(client, sample_png):
    pid = register(client, "그래프스트림")
    with sample_png.open("rb") as fh:
        with client.stream(
            "POST", "/analyze/stream", data={"patient_id": pid}, files={"file": ("cxr.png", fh, "image/png")}
        ) as res:
            events = parse_sse(res.read().decode("utf-8"))
    names = [n for n, _ in events]
    assert names[0] == "status" and names[-1] == "done"
    assert names.index("analysis") < names.index("delta")
    stages = [d["stage"] for n, d in events if n == "status"]
    assert stages[:3] == ["validating", "moderating", "classifying"] and "segmenting" in stages and "safety" in stages
    done = events[-1][1]["message"]
    assert done["meta"]["follow_ups"] and done["analysis"]["patient_summary"]


def test_follow_up_uses_prior_report_and_retrieval(client, sample_png):
    pid = register(client, "후속그래프")
    with sample_png.open("rb") as fh:
        client.post("/analyze", data={"patient_id": pid}, files={"file": ("cxr.png", fh, "image/png")})
    body = client.post("/analyze", data={"patient_id": pid, "text": "추가 검사는 뭐가 필요할까?"}).json()
    assert body["meta"]["question_category"] == "follow_up" and "analysis" not in body
    assert body["meta"]["retrieval"]["kb"] and body["meta"]["follow_ups"]
    llm = client.app.state.llm.fake
    question_call = [c for c in llm.calls if c["kind"] in {"complete", "stream"} and "[LATEST_ANALYSIS]" in c["user"]][
        -1
    ]
    assert "[KNOWLEDGE_BASE_CHUNKS]" in question_call["user"]


def test_off_topic_is_blocked_and_cached(client):
    pid = register(client, "차단환자")
    first = client.post("/analyze", data={"patient_id": pid, "text": "비트코인 지금 사야 할까?"}).json()
    assert first["meta"]["blocked"] and first["meta"]["question_category"] is None
    assert "범위" in first["content"] and "analysis" not in first
    second = client.post("/analyze", data={"patient_id": pid, "text": "비트코인 지금 사야 할까?"}).json()
    assert second["meta"]["blocked"] and second["meta"]["moderation_cached"] is True


def test_chit_chat_short_circuits(client):
    pid = register(client, "인사환자")
    body = client.post("/analyze", data={"patient_id": pid, "text": "안녕하세요"}).json()
    assert body["meta"]["question_category"] == "chit_chat" and not body["meta"]["blocked"]
    assert body["content"].startswith("안녕하세요")


def test_similar_case_index_finds_itself(settings):
    index = SimilarCaseIndex.load(settings.similar_cases_csv)
    assert index is not None and len(index.feature_names) == 102
    row = index.frame.iloc[0]
    features = {name: float(row[name]) for name in index.feature_names}
    hits = index.search(features, k=2)
    assert hits[0].image_id == row["image_id"] and hits[0].similarity == pytest.approx(1.0, abs=1e-4)
    assert hits[0].label_name in {"tb", "normal"}
    assert index.search({}, k=2) == []


SAMPLE_XML = """<?xml version="1.0"?><PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>12345</PMID>
<Article><Journal><Title>Chest</Title><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue></Journal>
<ArticleTitle>Radiomics of tuberculosis on chest radiographs</ArticleTitle>
<Abstract><AbstractText Label="BACKGROUND">Texture features</AbstractText>
<AbstractText>predict activity.</AbstractText></Abstract>
</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""


def test_pubmed_xml_parser_and_disabled_retriever():
    items = parse_pubmed_xml(SAMPLE_XML)
    assert len(items) == 1 and items[0].pmid == "12345" and items[0].year == "2021"
    assert items[0].abstract.startswith("Texture features") and items[0].url.endswith("/12345/")
    assert PubMedRetriever(enabled=False).search("tuberculosis") == []
    query = literature_query(
        {"ml_results": {"primary_label": "tb_suspected"}, "segmentation": {"zones": ["right upper zone"]}}, "", ""
    )
    assert "upper lobe" in query and "radiomics" in query
