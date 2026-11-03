"""Build the chat graph: nodes, routing, checkpointing, and a small runner used by the API.

One turn flows as::

    validate -> moderate -> classify -> [analyze] -> retrieve -> {kb | similar cases | literature} -> merge
      -> generate_report -> map_evidence -> {explain_doctor | explain_patient} -> compare -> safety_check
      -> enrich (follow-ups, title) -> respond

Text-only turns skip the analysis/report branch and go retrieve -> answer -> safety_check -> enrich -> respond;
chit-chat, out-of-scope and blocked messages go straight to respond. Parallel groups ({...}) are LangGraph
super-steps. When streaming, the clinician text is emitted token by token on the custom stream channel.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

import guardrails
import llm_report
from graph import llm_steps, prompts
from graph.moderation import BLOCKED_RESPONSE_KO, InappropriatePicker, QuestionClassifier
from graph.retrievers import PubMedRetriever, SimilarCaseIndex, literature_query
from graph.state import TurnState
from inference import InferenceService
from kb import KnowledgeBase, case_query
from llm_clients import ChatClient
from settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_TITLE_PREFIX = "상담 "


@dataclass
class GraphDeps:
    settings: Settings
    llm: ChatClient
    kb: KnowledgeBase
    inference: InferenceService
    picker: InappropriatePicker
    classifier: QuestionClassifier
    similar_index: SimilarCaseIndex | None = None
    pubmed: PubMedRetriever | None = None

    @classmethod
    def build(cls, settings: Settings, llm: ChatClient, kb: KnowledgeBase, inference: InferenceService) -> GraphDeps:
        picker = InappropriatePicker(
            client=llm,
            model=settings.summary_model,
            embedder=kb.embedder,
            threshold=settings.moderation_cache_threshold,
            timeout=settings.llm_timeout_s,
        )
        classifier = QuestionClassifier(client=llm, model=settings.summary_model, timeout=settings.llm_timeout_s)
        similar = SimilarCaseIndex.load(settings.similar_cases_csv)
        pubmed = PubMedRetriever(
            enabled=settings.pubmed_enabled,
            api_key=settings.ncbi_api_key,
            max_results=settings.pubmed_max_results,
            timeout_s=settings.pubmed_timeout_s,
        )
        return cls(settings, llm, kb, inference, picker, classifier, similar, pubmed)


def _emit(payload: dict[str, Any]) -> None:
    """Write a custom stream event if the graph is being streamed; no-op otherwise."""
    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 - not inside a streaming run
        return
    try:
        writer(payload)
    except Exception:  # noqa: BLE001
        pass


def _status(stage: str, message: str) -> dict[str, Any]:
    _emit({"type": "status", "stage": stage, "message": message})
    return {"status_log": [stage]}


def build_graph(deps: GraphDeps, checkpointer: BaseCheckpointSaver | None = None):
    settings = deps.settings
    llm = deps.llm
    timeout = settings.llm_timeout_s
    retries = settings.llm_max_retries

    # ------------------------------------------------------------------ nodes

    def validate(state: TurnState) -> dict[str, Any]:
        out = _status("validating", "입력을 확인하는 중")
        errors: list[str] = []
        text = (state.get("text") or "").strip()
        if not text and not state.get("image_path"):
            errors.append("메시지나 X-ray 이미지가 필요합니다.")
        if len(text) > 4000:
            errors.append("메시지가 너무 깁니다 (4000자 이하).")
        image_path = state.get("image_path")
        if image_path and not Path(image_path).exists():
            errors.append("업로드된 이미지를 찾을 수 없습니다.")
        out["validation"] = {"ok": not errors, "errors": errors}
        if errors:
            out["response"] = " ".join(errors)
            out["blocked"] = True
        return out

    def moderate(state: TurnState) -> dict[str, Any]:
        out = _status("moderating", "질문 범위를 확인하는 중")
        started = time.perf_counter()
        decision, cached, raw = deps.picker.check(state.get("text", ""), bool(state.get("image_path")))
        decision["cached"] = cached
        out["moderation"] = decision
        if raw is not None:
            out["traces"] = [
                llm_report.LLMTrace.from_usage(
                    "moderation", settings.summary_model, raw[1], (time.perf_counter() - started) * 1000
                ).as_dict()
            ]
        elif cached:
            out["traces"] = [
                llm_report.LLMTrace.from_usage("moderation", settings.summary_model, None, 0.0, cached=True).as_dict()
            ]
        if not decision["appropriate"]:
            out["response"] = BLOCKED_RESPONSE_KO
            out["blocked"] = True
        return out

    def classify(state: TurnState) -> dict[str, Any]:
        out = _status("classifying", "질문 유형을 분류하는 중")
        started = time.perf_counter()
        decision, usage = deps.classifier.classify(
            state.get("text", ""), bool(state.get("image_path")), state.get("prior_report") is not None
        )
        out["question"] = decision
        if usage is not None:
            out["traces"] = [
                llm_report.LLMTrace.from_usage(
                    "classification", settings.summary_model, usage, (time.perf_counter() - started) * 1000
                ).as_dict()
            ]
        if decision["category"] == "chit_chat":
            out["response"] = prompts.CHITCHAT_RESPONSE_KO
        elif decision["category"] == "out_of_scope":
            out["response"] = prompts.OUT_OF_SCOPE_RESPONSE_KO
            out["blocked"] = True
        return out

    def analyze(state: TurnState) -> dict[str, Any]:
        out = _status("segmenting", "U-Net 으로 병변을 분할하고 Radiomics 특징을 추출하는 중")
        artifacts = deps.inference.analyze(Path(state["image_path"]), state.get("image_id") or "image")
        out["analysis"] = {
            "case": artifacts.case,
            "overlay_path": artifacts.overlay_path,
            "mask_path": artifacts.mask_path,
        }
        return out

    def retrieve(state: TurnState) -> dict[str, Any]:
        return _status("retrieving", "지식베이스·유사 케이스·문헌을 검색하는 중")

    def _kb_query(state: TurnState) -> str:
        analysis = state.get("analysis")
        if analysis:
            return case_query(analysis["case"])
        question = state.get("question", {})
        return f"{state.get('text', '')} {question.get('search_query', '')}".strip()

    def retrieve_kb(state: TurnState) -> dict[str, Any]:
        results = deps.kb.search(_kb_query(state), k=settings.rag_top_k)
        return {"retrieval": {"kb": [{"id": c.id, "score": round(s, 3), "text": c.text} for c, s in results]}}

    def retrieve_similar(state: TurnState) -> dict[str, Any]:
        analysis = state.get("analysis")
        if deps.similar_index is None or not analysis:
            return {"retrieval": {"similar_cases": []}}
        features = analysis["case"].get("radiomics_features")
        cases = deps.similar_index.search(features, k=settings.similar_cases_k)
        return {"retrieval": {"similar_cases": [c.to_dict() for c in cases]}}

    def retrieve_literature(state: TurnState) -> dict[str, Any]:
        if deps.pubmed is None or not deps.pubmed.enabled:
            return {"retrieval": {"literature": []}}
        analysis = state.get("analysis")
        query = literature_query(
            analysis["case"] if analysis else state.get("prior_report"),
            state.get("text", ""),
            state.get("question", {}).get("search_query", ""),
        )
        return {
            "retrieval": {
                "literature": [item.to_dict() for item in deps.pubmed.search(query)],
                "literature_query": query,
            }
        }

    def merge_retrieval(state: TurnState) -> dict[str, Any]:
        retrieval = state.get("retrieval", {})
        n = (
            len(retrieval.get("kb", []))
            + len(retrieval.get("similar_cases", []))
            + len(retrieval.get("literature", []))
        )
        return _status("retrieved", f"근거 {n}건 확보")

    def generate_report(state: TurnState) -> dict[str, Any]:
        out = _status("reporting", "구조화 리포트를 생성하는 중")
        retrieval = state.get("retrieval", {})
        extra = {}
        if retrieval.get("similar_cases"):
            extra["similar_reference_cases"] = retrieval["similar_cases"]
        if retrieval.get("literature"):
            extra["literature"] = [
                {k: v for k, v in item.items() if k != "abstract"} for item in retrieval["literature"]
            ]
        model_meta = dict(deps.inference.model_metadata())
        if extra:
            model_meta["retrieved_context"] = extra
        report, trace = llm_report.generate_radiomics_report(
            state["analysis"]["case"],
            client=llm,
            kb=deps.kb,
            model_meta=model_meta,
            model=settings.report_model,
            fallback_model=settings.fallback_model_for("report"),
            top_k=settings.rag_top_k,
            timeout=timeout,
            max_retries=retries,
        )
        out["report"] = report
        out["traces"] = [trace.as_dict()]
        return out

    def map_evidence(state: TurnState) -> dict[str, Any]:
        """Attach retrieval provenance to the report (citations were validated in generate_report)."""
        out = _status("mapping", "근거를 리포트에 연결하는 중")
        report = dict(state["report"])
        retrieval = state.get("retrieval", {})
        report["evidence_sources"] = {
            "knowledge_base": [c["id"] for c in retrieval.get("kb", [])],
            "similar_cases": [c["image_id"] for c in retrieval.get("similar_cases", [])],
            "literature": [
                {"pmid": item["pmid"], "title": item["title"], "url": item["url"]}
                for item in retrieval.get("literature", [])
            ],
        }
        out["report"] = report
        analysis = state["analysis"]
        _emit(
            {
                "type": "analysis",
                "analysis": {
                    "overlay_path": analysis["overlay_path"],
                    "mask_path": analysis["mask_path"],
                    "ml_results": analysis["case"].get("ml_results"),
                    "segmentation": analysis["case"].get("segmentation"),
                    "report": report,
                },
            }
        )
        return out

    def explain_doctor(state: TurnState) -> dict[str, Any]:
        out = _status("explaining", "임상의용 요약을 작성하는 중")
        report = state["report"]
        if state.get("stream") and state.get("audience", "doctor") == "doctor":
            final_text, trace = "", None
            for kind, payload in llm_report.stream_summary_for_clinician(
                report, client=llm, model=settings.summary_model, clinician_note=state.get("text", ""), timeout=timeout
            ):
                if kind == "delta":
                    _emit({"type": "token", "text": payload})
                else:
                    final_text, trace = payload
            out["doctor_summary"] = final_text
            out["traces"] = [trace.as_dict()] if trace else []
            return out
        text, trace = llm_report.summarize_for_clinician(
            report,
            client=llm,
            model=settings.summary_model,
            fallback_model=settings.fallback_model_for("summary"),
            clinician_note=state.get("text", ""),
            timeout=timeout,
            max_retries=retries,
        )
        out["doctor_summary"] = text
        out["traces"] = [trace.as_dict()]
        return out

    def explain_patient(state: TurnState) -> dict[str, Any]:
        text, trace = llm_steps.explain_for_patient(
            state["report"],
            client=llm,
            model=settings.summary_model,
            fallback_model=settings.fallback_model_for("summary"),
            timeout=timeout,
            max_retries=retries,
        )
        return {"patient_summary": text, "traces": [trace.as_dict()]}

    def compare(state: TurnState) -> dict[str, Any]:
        out = _status("checking", "의사용·환자용 설명의 일관성을 확인하는 중")
        decision, trace = llm_steps.check_consistency(
            state["report"],
            state.get("doctor_summary", ""),
            state.get("patient_summary", ""),
            client=llm,
            model=settings.summary_model,
            timeout=timeout,
        )
        traces = [trace.as_dict()]
        out["consistency"] = decision
        if not decision["consistent"]:
            text, trace2 = llm_steps.explain_for_patient(
                state["report"],
                client=llm,
                model=settings.summary_model,
                fallback_model=settings.fallback_model_for("summary"),
                doctor_summary=state.get("doctor_summary", ""),
                revision_note=f"Align with the clinician summary: {decision['issue']}",
                timeout=timeout,
                max_retries=retries,
            )
            out["patient_summary"] = text
            out["consistency"]["regenerated"] = True
            traces.append(trace2.as_dict())
        out["traces"] = traces
        return out

    def answer(state: TurnState) -> dict[str, Any]:
        out = _status("answering", "최근 분석과 근거를 참고해 답변하는 중")
        question = state.get("text", "")
        context = state.get("prior_report")
        retrieval = state.get("retrieval") or None
        patient_audience = state.get("audience") == "patient"
        system_prompt = prompts.PATIENT_QA_SYSTEM_PROMPT if patient_audience else prompts.GENERAL_SYSTEM_PROMPT
        answer_key = "patient_summary" if patient_audience else "doctor_summary"
        if state.get("stream"):
            user = llm_steps.build_question_prompt(question, context, retrieval, patient_audience=patient_audience)
            final_text, trace = "", None
            for kind, payload in llm_report.stream_text(
                "question",
                system_prompt,
                user,
                client=llm,
                model=settings.summary_model,
                temperature=0.2,
                max_tokens=800,
                timeout=timeout,
            ):
                if kind == "delta":
                    _emit({"type": "token", "text": payload})
                else:
                    final_text, trace = payload
            out[answer_key] = final_text
            out["traces"] = [trace.as_dict()] if trace else []
            return out
        text, trace = llm_steps.answer_with_context(
            question,
            client=llm,
            model=settings.summary_model,
            fallback_model=settings.fallback_model_for("summary"),
            context_report=context,
            retrieval=retrieval,
            timeout=timeout,
            max_retries=retries,
            patient_audience=patient_audience,
        )
        out[answer_key] = text
        out["traces"] = [trace.as_dict()]
        return out

    def safety_check(state: TurnState) -> dict[str, Any]:
        out = _status("safety", "안전성 검토 중")
        results: dict[str, Any] = {}
        updates: dict[str, Any] = {}
        for key in ("doctor_summary", "patient_summary"):
            text = state.get(key)
            if not text:
                continue
            guard = guardrails.check_summary(text)
            results[key] = {
                "passed": guard.passed,
                "violations": guard.violations,
                "disclaimer_added": guard.appended_disclaimer,
            }
            if guard.text != text:
                updates[key] = guard.text
        out["safety"] = results
        out.update(updates)
        return out

    def enrich(state: TurnState) -> dict[str, Any]:
        out = _status("finalizing", "후속 질문과 제목을 정리하는 중")
        traces: list[dict[str, Any]] = []
        report = state.get("report") or state.get("prior_report")
        try:
            follow_ups, trace = llm_steps.suggest_follow_ups(
                report,
                state.get("text", ""),
                state.get("doctor_summary") or state.get("patient_summary", ""),
                client=llm,
                model=settings.summary_model,
                timeout=timeout,
            )
            out["follow_ups"] = follow_ups
            traces.append(trace.as_dict())
        except Exception as exc:  # noqa: BLE001 - suggestions are optional
            logger.warning("follow-up generation failed: %s", exc)
            out["follow_ups"] = {"doctor": [], "patient": []}
        if state.get("report") and state.get("session_title_is_default"):
            try:
                title, trace = llm_steps.generate_title(
                    state["report"], client=llm, model=settings.summary_model, timeout=timeout
                )
                out["title"] = title
                traces.append(trace.as_dict())
            except Exception as exc:  # noqa: BLE001
                logger.warning("title generation failed: %s", exc)
        out["traces"] = traces
        return out

    def respond(state: TurnState) -> dict[str, Any]:
        if state.get("response"):
            return {"status_log": ["respond"]}
        audience = state.get("audience", "doctor")
        text = (
            state.get("patient_summary")
            if audience == "patient" and state.get("patient_summary")
            else state.get("doctor_summary", "")
        )
        return {"response": text, "status_log": ["respond"]}

    # ------------------------------------------------------------------ routing

    def after_validate(state: TurnState) -> str:
        return "moderate" if state["validation"]["ok"] else "respond"

    def after_moderate(state: TurnState) -> str:
        return "classify" if state["moderation"]["appropriate"] else "respond"

    def after_classify(state: TurnState) -> str:
        if state.get("image_path"):
            return "analyze"
        category = state["question"]["category"]
        if category in {"chit_chat", "out_of_scope"}:
            return "respond"
        return "retrieve" if state["question"].get("needs_retrieval", True) else "answer"

    def after_merge(state: TurnState) -> str:
        return "generate_report" if state.get("analysis") else "answer"

    # ------------------------------------------------------------------ graph

    g = StateGraph(TurnState)
    for name, fn in (
        ("validate", validate),
        ("moderate", moderate),
        ("classify", classify),
        ("analyze", analyze),
        ("retrieve", retrieve),
        ("retrieve_kb", retrieve_kb),
        ("retrieve_similar", retrieve_similar),
        ("retrieve_literature", retrieve_literature),
        ("merge_retrieval", merge_retrieval),
        ("generate_report", generate_report),
        ("map_evidence", map_evidence),
        ("explain_doctor", explain_doctor),
        ("explain_patient", explain_patient),
        ("compare", compare),
        ("answer", answer),
        ("safety_check", safety_check),
        ("enrich", enrich),
        ("respond", respond),
    ):
        g.add_node(name, fn)

    g.add_edge(START, "validate")
    g.add_conditional_edges("validate", after_validate, {"moderate": "moderate", "respond": "respond"})
    g.add_conditional_edges("moderate", after_moderate, {"classify": "classify", "respond": "respond"})
    g.add_conditional_edges(
        "classify",
        after_classify,
        {"analyze": "analyze", "retrieve": "retrieve", "answer": "answer", "respond": "respond"},
    )
    g.add_edge("analyze", "retrieve")
    for node in ("retrieve_kb", "retrieve_similar", "retrieve_literature"):
        g.add_edge("retrieve", node)
    g.add_edge(["retrieve_kb", "retrieve_similar", "retrieve_literature"], "merge_retrieval")
    g.add_conditional_edges("merge_retrieval", after_merge, {"generate_report": "generate_report", "answer": "answer"})
    g.add_edge("generate_report", "map_evidence")
    g.add_edge("map_evidence", "explain_doctor")
    g.add_edge("map_evidence", "explain_patient")
    g.add_edge(["explain_doctor", "explain_patient"], "compare")
    g.add_edge("compare", "safety_check")
    g.add_edge("answer", "safety_check")
    g.add_edge("safety_check", "enrich")
    g.add_edge("enrich", "respond")
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------- checkpointer + runner


def make_checkpointer(settings: Settings) -> BaseCheckpointSaver:
    url = settings.database_url
    if url.startswith("sqlite"):
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = settings.graph_checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver
    if url.startswith("postgresql"):
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row

        dsn = url.replace("postgresql+psycopg://", "postgresql://", 1)
        conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
        saver = PostgresSaver(conn)
        saver.setup()
        return saver
    logger.warning("no persistent checkpointer for %s; using in-memory", url.split(":")[0])
    return MemorySaver()


class ChatGraph:
    """Thin wrapper: one thread per turn (``s<session>-<request>``) so each run is inspectable in the checkpointer."""

    def __init__(self, deps: GraphDeps, checkpointer: BaseCheckpointSaver | None = None):
        self.deps = deps
        self.graph = build_graph(deps, checkpointer)

    @staticmethod
    def thread_id(state: dict[str, Any]) -> str:
        return f"s{state.get('session_id', 0)}-{state.get('request_id') or int(time.time() * 1000)}"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        config = {"configurable": {"thread_id": self.thread_id(state)}}
        return self.graph.invoke({**state, "stream": False}, config)

    def stream(self, state: dict[str, Any]) -> Iterator[tuple[str, Any]]:
        """Yield ("status", payload) / ("token", text) events, then ("final", state)."""
        config = {"configurable": {"thread_id": self.thread_id(state)}}
        for mode, payload in self.graph.stream({**state, "stream": True}, config, stream_mode=["custom"]):
            if mode != "custom" or not isinstance(payload, dict):
                continue
            if payload.get("type") == "token":
                yield "token", payload.get("text", "")
            elif payload.get("type") == "status":
                yield "status", {"stage": payload.get("stage"), "message": payload.get("message")}
            elif payload.get("type") == "analysis":
                yield "analysis", payload.get("analysis", {})
        yield "final", self.graph.get_state(config).values

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()
