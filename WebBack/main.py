"""FastAPI entrypoint: authentication, patient registry, consultation chat, and the X-ray analysis endpoint.

Request flow for ``POST /analyze`` with an image:
    upload -> U-Net segmentation -> lesion summary -> radiomics -> TB classifier   (inference service)
           -> stage-1 structured report -> stage-2 clinician summary               (LLM layer)
           -> persisted as Message + DiagnosisResult, returned with overlay URLs.
"""

from __future__ import annotations

import io
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import func
from sqlalchemy.orm import Session
from starlette.concurrency import iterate_in_threadpool

import auth
import database
import DBTable
import llm_report
from graph.build import ChatGraph, GraphDeps, make_checkpointer
from inference import AnalysisArtifacts, InferenceService, build_inference_service, new_image_id
from logging_config import configure_logging
from schemas import FeedbackIn, PatientCreate, SessionCreate
from settings import Settings, get_settings

logger = logging.getLogger("webback")

ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG"}


# ============================================================ app lifecycle


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    settings.validate_for_environment()
    if settings.auto_migrate:
        database.run_migrations()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(app.state, "inference"):
        app.state.inference = build_inference_service(settings)
    if not hasattr(app.state, "llm"):
        app.state.llm = llm_report.build_chat_client(settings)
    if not hasattr(app.state, "kb"):
        app.state.kb = llm_report.build_knowledge_base(settings)
    if not hasattr(app.state, "graph"):
        deps = GraphDeps.build(settings, app.state.llm, app.state.kb, app.state.inference)
        app.state.graph = ChatGraph(deps, make_checkpointer(settings))
    auth.seed_doctor(settings)
    logger.info(
        "ready: env=%s inference=%s llm=%s report_model=%s summary_model=%s fallback=%s graph_nodes=%d",
        settings.environment,
        app.state.inference.name,
        settings.llm_backend,
        settings.report_model,
        settings.summary_model,
        settings.fallback_model_for("summary"),
        len(app.state.graph.graph.get_graph().nodes),
    )
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(json_logs=settings.log_json)
    app = FastAPI(title=settings.app_name, version="0.3.0", lifespan=lifespan)
    app.state.settings = settings
    settings.static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")

    limiter = Limiter(key_func=get_remote_address, enabled=settings.rate_limit_enabled)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Attach a request id, time the request, and log one access line (no patient data in the line)."""
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        if not request.url.path.startswith("/static"):
            logger.info(
                "%s %s -> %s (%.0f ms)",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
                extra={"request_id": request_id, "status": response.status_code, "duration_ms": round(duration_ms)},
            )
        return response

    register_routes(app, limiter)
    return app


# ============================================================ helpers


def get_db():
    yield from database.get_db()


def get_inference(request: Request) -> InferenceService:
    return request.app.state.inference


def get_llm(request: Request) -> llm_report.ChatClient:
    return request.app.state.llm


def static_url(settings: Settings, relative: str | None) -> str | None:
    return f"{settings.public_base_url.rstrip('/')}/static/{relative}" if relative else None


def message_to_dict(msg: DBTable.Message, settings: Settings) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": msg.id,
        "session_id": msg.session_id,
        "role": msg.role,
        "content": msg.content or "",
        "image_path": static_url(settings, msg.image_path),
        "created_at": msg.created_at.isoformat() if msg.created_at else None,
    }
    if msg.meta:
        payload["meta"] = msg.meta
    diagnosis = msg.diagnosis
    if diagnosis is not None:
        case = diagnosis.model_raw_json or {}
        payload["analysis"] = {
            "diagnosis_id": diagnosis.id,
            "overlay_url": static_url(settings, diagnosis.overlay_path),
            "mask_url": static_url(settings, diagnosis.mask_path),
            "ml_results": case.get("ml_results"),
            "segmentation": case.get("segmentation"),
            "report": diagnosis.report_json,
            "doctor_feedback": diagnosis.doctor_feedback,
            "doctor_note": diagnosis.doctor_note,
            "patient_summary": diagnosis.patient_summary,
            "llm": diagnosis.llm_meta,
        }
    return payload


def session_to_dict(session: DBTable.ChatSession) -> dict[str, Any]:
    messages = session.messages or []
    return {
        "id": session.id,
        "patient_id": session.patient_id,
        "title": session.title,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "message_count": len(messages),
        "last_message_at": messages[-1].created_at.isoformat() if messages and messages[-1].created_at else None,
    }


def record_llm_calls(
    db: Session,
    traces: list[llm_report.LLMTrace],
    *,
    request_id: str | None,
    user_id: int,
    session_id: int | None,
    diagnosis_id: int | None,
) -> None:
    for trace in traces:
        db.add(
            DBTable.LLMCall(
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
                diagnosis_id=diagnosis_id,
                stage=trace.stage,
                requested_model=trace.requested_model,
                model=trace.model or trace.requested_model,
                prompt_version=trace.prompt_version,
                latency_ms=trace.latency_ms,
                input_tokens=trace.input_tokens,
                output_tokens=trace.output_tokens,
                cost_usd=trace.cost_usd,
                attempts=trace.attempts,
                fallback_used=trace.fallback_used,
                regenerated=trace.regenerated,
                success=trace.success,
                error=trace.error,
            )
        )
    db.commit()


def llm_meta_from_traces(traces: list[llm_report.LLMTrace]) -> dict[str, Any]:
    by_stage = {t.stage: t for t in traces}
    cost = [t.cost_usd for t in traces if t.cost_usd is not None]
    return {
        "report_model": by_stage["report"].model if "report" in by_stage else None,
        "summary_model": by_stage["summary"].model if "summary" in by_stage else None,
        "latency_ms": round(sum(t.latency_ms for t in traces), 1),
        "input_tokens": sum(t.input_tokens for t in traces),
        "output_tokens": sum(t.output_tokens for t in traces),
        "cost_usd": round(sum(cost), 6) if cost else None,
        "prompt_version": llm_report.PROMPT_VERSION,
        "fallback_used": any(t.fallback_used for t in traces),
        "regenerated": any(t.regenerated for t in traces),
    }


def resolve_session(db: Session, patient_id: int, session_id: int | None) -> DBTable.ChatSession:
    """Return the requested session (must belong to the patient), else the newest, else a new one."""
    if session_id is not None:
        session = (
            db.query(DBTable.ChatSession)
            .filter(DBTable.ChatSession.id == session_id, DBTable.ChatSession.patient_id == patient_id)
            .first()
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    session = (
        db.query(DBTable.ChatSession)
        .filter(DBTable.ChatSession.patient_id == patient_id)
        .order_by(DBTable.ChatSession.created_at.desc(), DBTable.ChatSession.id.desc())
        .first()
    )
    if not session:
        session = DBTable.ChatSession(patient_id=patient_id, title=f"상담 {datetime.now():%Y-%m-%d %H:%M}")
        db.add(session)
        db.commit()
        db.refresh(session)
    return session


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def validate_and_store_upload(contents: bytes, settings: Settings) -> str:
    """Validate the upload is a real PNG/JPEG under the size limit; return the path relative to static."""
    limit = settings.max_upload_mb * 1024 * 1024
    if len(contents) > limit:
        raise HTTPException(status_code=413, detail=f"Image exceeds {settings.max_upload_mb} MB")
    try:
        with Image.open(io.BytesIO(contents)) as img:
            img_format = (img.format or "").upper()
            img.verify()
    except (UnidentifiedImageError, OSError) as err:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image") from err
    if img_format not in ALLOWED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=400, detail=f"Unsupported image format {img_format or 'unknown'}; use PNG or JPEG"
        )

    ext = "png" if img_format == "PNG" else "jpg"
    relative = f"uploads/{new_image_id()}.{ext}"
    (settings.static_dir / relative).write_bytes(contents)
    return relative


def latest_report_for_session(db: Session, session_id: int) -> dict[str, Any] | None:
    row = (
        db.query(DBTable.DiagnosisResult)
        .join(DBTable.Message, DBTable.Message.id == DBTable.DiagnosisResult.message_id)
        .filter(DBTable.Message.session_id == session_id)
        .order_by(DBTable.DiagnosisResult.id.desc())
        .first()
    )
    return row.report_json if row else None


def persist_turn(
    db: Session,
    session: DBTable.ChatSession,
    text: str,
    image_rel: str | None,
    assistant_text: str,
    artifacts: AnalysisArtifacts | None,
    report: dict[str, Any] | None,
    traces: list[llm_report.LLMTrace],
    *,
    patient_summary: str | None = None,
    meta: dict[str, Any] | None = None,
) -> DBTable.Message:
    """Store the user turn, the assistant turn and (for images) the diagnosis result; return the assistant message."""
    db.add(DBTable.Message(session_id=session.id, role="user", content=text, image_path=image_rel))
    db.commit()
    ai_msg = DBTable.Message(session_id=session.id, role="assistant", content=assistant_text, meta=meta)
    db.add(ai_msg)
    db.commit()
    db.refresh(ai_msg)
    if artifacts is not None:
        db.add(
            DBTable.DiagnosisResult(
                message_id=ai_msg.id,
                model_raw_json=artifacts.case,
                report_json=report,
                gpt_interpretation=assistant_text,
                patient_summary=patient_summary,
                overlay_path=artifacts.overlay_path,
                mask_path=artifacts.mask_path,
                llm_meta=llm_meta_from_traces(traces),
            )
        )
        db.commit()
        db.refresh(ai_msg)
    if meta and meta.get("title") and session_title_is_default(session.title):
        session.title = meta["title"]
        db.commit()
    return ai_msg


def session_title_is_default(title: str | None) -> bool:
    return not title or title.startswith(("상담 ", "Consult"))


def graph_input(
    *,
    session: DBTable.ChatSession,
    patient_id: int,
    user_id: int,
    request_id: str | None,
    text: str,
    image_rel: str | None,
    settings: Settings,
    prior_report: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "session_id": session.id,
        "patient_id": patient_id,
        "user_id": user_id,
        "request_id": request_id,
        "text": text,
        "image_path": str(settings.static_dir / image_rel) if image_rel else None,
        "image_id": Path(image_rel).stem if image_rel else None,
        "audience": "doctor",
        "prior_report": prior_report,
        "session_title_is_default": session_title_is_default(session.title),
    }


def graph_result_to_turn(
    final: dict[str, Any],
) -> tuple[AnalysisArtifacts | None, dict[str, Any] | None, list[llm_report.LLMTrace], dict[str, Any]]:
    """Unpack a finished graph state into what persist_turn needs."""
    analysis = final.get("analysis")
    artifacts = (
        AnalysisArtifacts(case=analysis["case"], overlay_path=analysis["overlay_path"], mask_path=analysis["mask_path"])
        if analysis
        else None
    )
    traces = [llm_report.LLMTrace(**t) for t in final.get("traces", [])]
    question = final.get("question") or {}
    moderation = final.get("moderation") or {}
    meta = {
        "question_category": question.get("category"),
        "follow_ups": (final.get("follow_ups") or {}).get("doctor", []),
        "patient_follow_ups": (final.get("follow_ups") or {}).get("patient", []),
        "blocked": bool(final.get("blocked")),
        "block_reason": moderation.get("reason") if final.get("blocked") else None,
        "moderation_cached": bool(moderation.get("cached")),
        "title": final.get("title"),
        "consistency": final.get("consistency"),
        "safety": final.get("safety"),
        "retrieval": {
            "kb": [c["id"] for c in (final.get("retrieval") or {}).get("kb", [])],
            "similar_cases": [c["image_id"] for c in (final.get("retrieval") or {}).get("similar_cases", [])],
            "literature": [item["pmid"] for item in (final.get("retrieval") or {}).get("literature", [])],
        },
        "graph_stages": final.get("status_log", []),
    }
    return artifacts, final.get("report"), traces, meta


def get_patient_or_404(db: Session, patient_id: int, user: DBTable.User) -> DBTable.Patient:
    patient = (
        db.query(DBTable.Patient).filter(DBTable.Patient.id == patient_id, DBTable.Patient.user_id == user.id).first()
    )
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


# ============================================================ routes


def register_routes(app: FastAPI, limiter: Limiter) -> None:
    @app.get("/")
    def root():
        return {"message": "Radiomics CDS API", "docs": "/docs"}

    @app.get("/health")
    def health(request: Request):
        settings: Settings = request.app.state.settings
        return {
            "status": "ok",
            "environment": settings.environment,
            "inference_backend": request.app.state.inference.name,
            "llm_backend": settings.llm_backend,
            "llm_providers": getattr(request.app.state.llm, "providers", None),
            "fallback_models": {
                "report": settings.fallback_model_for("report"),
                "summary": settings.fallback_model_for("summary"),
            },
            "graph_nodes": sorted(n for n in request.app.state.graph.graph.get_graph().nodes if not n.startswith("__")),
            "models": request.app.state.inference.model_metadata(),
        }

    @app.get("/graph")
    def graph_definition(request: Request):
        """Mermaid definition of the chat graph (for docs and debugging)."""
        return {"mermaid": request.app.state.graph.mermaid()}

    # ---------------------------------------------------------------- auth

    @app.post("/auth/login", response_model=auth.TokenResponse)
    @limiter.limit("10/minute")
    def login(request: Request, body: auth.LoginRequest, db: Session = Depends(get_db)):
        settings: Settings = request.app.state.settings
        user = auth.authenticate(db, body.email, body.password)
        if user is None:
            raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 올바르지 않습니다.")
        return auth.TokenResponse(
            access_token=auth.create_access_token(user, settings),
            user=auth.UserOut(id=user.id, email=user.email, full_name=user.full_name),
        )

    @app.get("/auth/me", response_model=auth.UserOut)
    def me(current_user: auth.CurrentUser):
        return auth.UserOut(id=current_user.id, email=current_user.email, full_name=current_user.full_name)

    # ---------------------------------------------------------------- patients

    @app.post("/patients/register")
    def register_patient(patient: PatientCreate, current_user: auth.CurrentUser, db: Session = Depends(get_db)):
        existing = (
            db.query(DBTable.Patient)
            .filter(
                DBTable.Patient.user_id == current_user.id,
                DBTable.Patient.name == patient.name,
                DBTable.Patient.birthDate == patient.birthDate,
            )
            .first()
        )
        if existing:
            return {"status": "found", "message": "이미 등록된 환자입니다.", "data": existing}

        new_patient = DBTable.Patient(
            name=patient.name, birthDate=patient.birthDate, gender=patient.gender, user_id=current_user.id
        )
        db.add(new_patient)
        db.commit()
        db.refresh(new_patient)
        return {"status": "created", "message": "환자가 등록되었습니다.", "data": new_patient}

    @app.get("/patients")
    def get_patients(current_user: auth.CurrentUser, db: Session = Depends(get_db)):
        return (
            db.query(DBTable.Patient)
            .filter(DBTable.Patient.user_id == current_user.id)
            .order_by(DBTable.Patient.created_at.desc())
            .all()
        )

    @app.get("/patients/{patient_id}")
    def get_patient(patient_id: int, current_user: auth.CurrentUser, db: Session = Depends(get_db)):
        patient = get_patient_or_404(db, patient_id, current_user)
        return {
            "id": patient.id,
            "name": patient.name,
            "birthDate": patient.birthDate,
            "gender": patient.gender,
            "lastVisit": patient.created_at.strftime("%Y-%m-%d") if patient.created_at else "",
        }

    @app.get("/patients/{patient_id}/messages")
    def get_messages(
        patient_id: int,
        request: Request,
        current_user: auth.CurrentUser,
        session_id: int | None = None,
        db: Session = Depends(get_db),
    ):
        settings: Settings = request.app.state.settings
        get_patient_or_404(db, patient_id, current_user)
        query = db.query(DBTable.ChatSession).filter(DBTable.ChatSession.patient_id == patient_id)
        if session_id is not None:
            query = query.filter(DBTable.ChatSession.id == session_id)
        msgs: list[dict[str, Any]] = [message_to_dict(m, settings) for s in query.all() for m in s.messages]
        msgs.sort(key=lambda x: (x["created_at"] or "", x["id"]))
        return msgs

    # ---------------------------------------------------------------- sessions

    @app.get("/patients/{patient_id}/sessions")
    def list_sessions(patient_id: int, current_user: auth.CurrentUser, db: Session = Depends(get_db)):
        get_patient_or_404(db, patient_id, current_user)
        sessions = (
            db.query(DBTable.ChatSession)
            .filter(DBTable.ChatSession.patient_id == patient_id)
            .order_by(DBTable.ChatSession.created_at.desc(), DBTable.ChatSession.id.desc())
            .all()
        )
        return [session_to_dict(s) for s in sessions]

    @app.post("/patients/{patient_id}/sessions")
    def create_session(
        patient_id: int,
        current_user: auth.CurrentUser,
        body: SessionCreate | None = None,
        db: Session = Depends(get_db),
    ):
        get_patient_or_404(db, patient_id, current_user)
        title = (body.title if body and body.title else None) or f"상담 {datetime.now():%Y-%m-%d %H:%M}"
        session = DBTable.ChatSession(patient_id=patient_id, title=title)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session_to_dict(session)

    # ---------------------------------------------------------------- diagnoses / feedback

    def get_diagnosis_or_404(db: Session, diagnosis_id: int, user: DBTable.User) -> DBTable.DiagnosisResult:
        diagnosis = (
            db.query(DBTable.DiagnosisResult)
            .join(DBTable.Message, DBTable.Message.id == DBTable.DiagnosisResult.message_id)
            .join(DBTable.ChatSession, DBTable.ChatSession.id == DBTable.Message.session_id)
            .join(DBTable.Patient, DBTable.Patient.id == DBTable.ChatSession.patient_id)
            .filter(DBTable.DiagnosisResult.id == diagnosis_id, DBTable.Patient.user_id == user.id)
            .first()
        )
        if not diagnosis:
            raise HTTPException(status_code=404, detail="Diagnosis not found")
        return diagnosis

    @app.get("/diagnoses/{diagnosis_id}")
    def get_diagnosis(
        diagnosis_id: int, request: Request, current_user: auth.CurrentUser, db: Session = Depends(get_db)
    ):
        settings: Settings = request.app.state.settings
        diagnosis = get_diagnosis_or_404(db, diagnosis_id, current_user)
        return {
            "diagnosis_id": diagnosis.id,
            "message_id": diagnosis.message_id,
            "case": diagnosis.model_raw_json,
            "report": diagnosis.report_json,
            "summary": diagnosis.gpt_interpretation,
            "overlay_url": static_url(settings, diagnosis.overlay_path),
            "mask_url": static_url(settings, diagnosis.mask_path),
            "doctor_feedback": diagnosis.doctor_feedback,
            "doctor_note": diagnosis.doctor_note,
            "llm": diagnosis.llm_meta,
        }

    @app.post("/diagnoses/{diagnosis_id}/feedback")
    def post_feedback(
        diagnosis_id: int, body: FeedbackIn, current_user: auth.CurrentUser, db: Session = Depends(get_db)
    ):
        diagnosis = get_diagnosis_or_404(db, diagnosis_id, current_user)
        diagnosis.doctor_feedback = body.agree
        diagnosis.doctor_note = (body.note or "").strip() or None
        db.commit()
        return {
            "diagnosis_id": diagnosis.id,
            "doctor_feedback": diagnosis.doctor_feedback,
            "doctor_note": diagnosis.doctor_note,
        }

    # ---------------------------------------------------------------- telemetry

    @app.get("/metrics/llm")
    def llm_metrics(current_user: auth.CurrentUser, db: Session = Depends(get_db)):
        """Aggregate LLM usage (all users): calls, tokens, cost, latency per stage and model."""
        rows = (
            db.query(
                DBTable.LLMCall.stage,
                DBTable.LLMCall.model,
                func.count(DBTable.LLMCall.id),
                func.sum(DBTable.LLMCall.input_tokens),
                func.sum(DBTable.LLMCall.output_tokens),
                func.sum(DBTable.LLMCall.cost_usd),
                func.avg(DBTable.LLMCall.latency_ms),
                func.sum(func.cast(DBTable.LLMCall.fallback_used, DBTable.Integer)),
                func.sum(func.cast(DBTable.LLMCall.regenerated, DBTable.Integer)),
                func.sum(func.cast(DBTable.LLMCall.success, DBTable.Integer)),
            )
            .group_by(DBTable.LLMCall.stage, DBTable.LLMCall.model)
            .all()
        )
        feedback = (
            db.query(DBTable.DiagnosisResult.doctor_feedback, func.count(DBTable.DiagnosisResult.id))
            .group_by(DBTable.DiagnosisResult.doctor_feedback)
            .all()
        )
        return {
            "by_stage_model": [
                {
                    "stage": stage,
                    "model": model,
                    "calls": int(calls),
                    "input_tokens": int(tin or 0),
                    "output_tokens": int(tout or 0),
                    "cost_usd": round(float(cost or 0.0), 6),
                    "latency_ms_avg": round(float(lat or 0.0), 1),
                    "fallback_used": int(fb or 0),
                    "regenerated": int(rg or 0),
                    "success": int(ok or 0),
                }
                for stage, model, calls, tin, tout, cost, lat, fb, rg, ok in rows
            ],
            "doctor_feedback": {str(k): int(v) for k, v in feedback},
            "prompt_version": llm_report.PROMPT_VERSION,
        }

    # ---------------------------------------------------------------- analysis

    @app.post("/analyze")
    @limiter.limit(lambda: app.state.settings.rate_limit_analyze)
    async def analyze(
        request: Request,
        current_user: auth.CurrentUser,
        patient_id: int = Form(...),
        text: str = Form(""),
        session_id: int | None = Form(None),
        file: UploadFile | None = File(None),
        db: Session = Depends(get_db),
        inference: InferenceService = Depends(get_inference),
        llm: llm_report.ChatClient = Depends(get_llm),
    ):
        settings: Settings = request.app.state.settings
        text = (text or "").strip()
        if not text and file is None:
            raise HTTPException(status_code=400, detail="Provide a message, an image, or both")
        get_patient_or_404(db, patient_id, current_user)
        session = resolve_session(db, patient_id, session_id)
        request_id = getattr(request.state, "request_id", None)

        image_rel: str | None = None
        if file is not None:
            contents = await file.read()
            image_rel = validate_and_store_upload(contents, settings)
        prior_report = latest_report_for_session(db, session.id)
        chat_graph: ChatGraph = request.app.state.graph
        try:
            final = await run_in_threadpool(
                chat_graph.run,
                graph_input(
                    session=session,
                    patient_id=patient_id,
                    user_id=current_user.id,
                    request_id=request_id,
                    text=text,
                    image_rel=image_rel,
                    settings=settings,
                    prior_report=prior_report,
                ),
            )
        except llm_report.LLMError as exc:
            logger.error("graph failed: %s", exc, extra={"request_id": request_id})
            raise HTTPException(status_code=503, detail="LLM 호출에 실패했습니다. 잠시 후 다시 시도해주세요.") from exc
        artifacts, report, traces, meta = graph_result_to_turn(final)
        assistant_text = final.get("response", "")
        patient_summary = final.get("patient_summary") or None

        ai_msg = persist_turn(
            db,
            session,
            text,
            image_rel,
            assistant_text,
            artifacts,
            report,
            traces,
            patient_summary=patient_summary,
            meta=meta,
        )
        record_llm_calls(
            db,
            traces,
            request_id=request_id,
            user_id=current_user.id,
            session_id=session.id,
            diagnosis_id=ai_msg.diagnosis.id if ai_msg.diagnosis else None,
        )
        logger.info(
            "analyze patient=%s session=%s image=%s category=%s label=%s cost=%s",
            patient_id,
            session.id,
            bool(image_rel),
            meta.get("question_category"),
            artifacts.ml_results.get("primary_label") if artifacts else None,
            llm_meta_from_traces(traces)["cost_usd"],
            extra={"request_id": request_id, "user_id": current_user.id},
        )
        return message_to_dict(ai_msg, settings)

    @app.post("/analyze/stream")
    @limiter.limit(lambda: app.state.settings.rate_limit_analyze)
    async def analyze_stream(
        request: Request,
        current_user: auth.CurrentUser,
        patient_id: int = Form(...),
        text: str = Form(""),
        session_id: int | None = Form(None),
        file: UploadFile | None = File(None),
        db: Session = Depends(get_db),
        inference: InferenceService = Depends(get_inference),
        llm: llm_report.ChatClient = Depends(get_llm),
    ):
        """Same as /analyze but as Server-Sent Events: status -> analysis -> delta* -> done (or error)."""
        settings: Settings = request.app.state.settings
        text = (text or "").strip()
        if not text and file is None:
            raise HTTPException(status_code=400, detail="Provide a message, an image, or both")
        get_patient_or_404(db, patient_id, current_user)
        session = resolve_session(db, patient_id, session_id)
        contents = await file.read() if file is not None else None
        image_rel = validate_and_store_upload(contents, settings) if contents is not None else None
        request_id = getattr(request.state, "request_id", None)
        user_id = current_user.id
        session_pk = session.id

        chat_graph: ChatGraph = request.app.state.graph

        async def event_stream():
            own_db = database.SessionLocal()
            try:
                session_obj = own_db.get(DBTable.ChatSession, session_pk)
                prior_report = latest_report_for_session(own_db, session_pk)
                state = graph_input(
                    session=session_obj,
                    patient_id=patient_id,
                    user_id=user_id,
                    request_id=request_id,
                    text=text,
                    image_rel=image_rel,
                    settings=settings,
                    prior_report=prior_report,
                )
                final: dict[str, Any] | None = None
                async for kind, payload in iterate_in_threadpool(chat_graph.stream(state)):
                    if kind == "status":
                        yield sse("status", payload)
                    elif kind == "token":
                        yield sse("delta", {"text": payload})
                    elif kind == "analysis":
                        yield sse(
                            "analysis",
                            {
                                "overlay_url": static_url(settings, payload.get("overlay_path")),
                                "mask_url": static_url(settings, payload.get("mask_path")),
                                "ml_results": payload.get("ml_results"),
                                "segmentation": payload.get("segmentation"),
                                "report": payload.get("report"),
                            },
                        )
                    elif kind == "final":
                        final = payload
                if final is None:
                    raise llm_report.LLMError("graph produced no final state")
                artifacts, report, traces, meta = graph_result_to_turn(final)
                ai_msg = persist_turn(
                    own_db,
                    session_obj,
                    text,
                    image_rel,
                    final.get("response", ""),
                    artifacts,
                    report,
                    traces,
                    patient_summary=final.get("patient_summary") or None,
                    meta=meta,
                )
                record_llm_calls(
                    own_db,
                    traces,
                    request_id=request_id,
                    user_id=user_id,
                    session_id=session_pk,
                    diagnosis_id=ai_msg.diagnosis.id if ai_msg.diagnosis else None,
                )
                yield sse("done", {"message": message_to_dict(ai_msg, settings)})
            except HTTPException as exc:
                yield sse("error", {"detail": exc.detail})
            except llm_report.LLMError as exc:
                logger.error("stream failed: %s", exc, extra={"request_id": request_id})
                yield sse("error", {"detail": "LLM 호출에 실패했습니다. 잠시 후 다시 시도해주세요."})
            except Exception:  # noqa: BLE001
                logger.exception("stream failed", extra={"request_id": request_id})
                yield sse("error", {"detail": "분석 중 오류가 발생했습니다."})
            finally:
                own_db.close()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


app = create_app()
