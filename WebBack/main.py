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
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
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
from schemas import (
    AdminUserCreate,
    AdminUserPatch,
    FeedbackIn,
    LinkCreate,
    PatientCreate,
    SessionCreate,
    ShareQuestion,
    VisitIn,
    VisitPatch,
)
from settings import Settings, get_settings

logger = logging.getLogger("webback")

ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG"}
NIFTI_SUFFIXES = (".nii", ".nii.gz")


def looks_like_nifti(filename: str | None, contents: bytes) -> bool:
    name = (filename or "").lower()
    if not name.endswith(NIFTI_SUFFIXES):
        return False
    if contents[:2] == b"\x1f\x8b":  # gzip
        return True
    return len(contents) > 348 and contents[344:348] in (b"n+1\0", b"ni1\0")


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
            "modality": case.get("modality", "xray"),
            "qc": case.get("qc"),
            "image": case.get("image"),
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
        "visit_id": session.visit_id,
        "title": session.title,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "message_count": len(messages),
        "last_message_at": messages[-1].created_at.isoformat() if messages and messages[-1].created_at else None,
    }


def patient_to_dict(patient: DBTable.Patient) -> dict[str, Any]:
    doctor = patient.doctor
    return {
        "id": patient.id,
        "name": patient.name,
        "birthDate": patient.birthDate,
        "gender": patient.gender,
        "patient_code": patient.patient_code,
        "medical_history": patient.medical_history,
        "created_at": patient.created_at.isoformat() if patient.created_at else None,
        "lastVisit": patient.created_at.strftime("%Y-%m-%d") if patient.created_at else "",
        "doctor": {"id": doctor.id, "full_name": doctor.full_name} if doctor else None,
        "hospital_unit_id": patient.hospital_unit_id,
    }


def next_patient_code(db: Session) -> str:
    count = db.query(func.count(DBTable.Patient.id)).scalar() or 0
    while True:
        count += 1
        candidate = f"P-{count:06d}"
        if not db.query(DBTable.Patient.id).filter(DBTable.Patient.patient_code == candidate).first():
            return candidate


def scoped_patients(db: Session, user: DBTable.User, scope: str = "mine"):
    """Doctors see their own patients; administrators may widen to the whole hospital unit."""
    query = db.query(DBTable.Patient)
    if auth.is_admin(user) and scope == "unit" and user.hospital_unit_id is not None:
        return query.filter(DBTable.Patient.hospital_unit_id == user.hospital_unit_id)
    return query.filter(DBTable.Patient.user_id == user.id)


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=f"Invalid datetime: {value}") from err


def diagnosis_brief(diagnosis: DBTable.DiagnosisResult, settings: Settings) -> dict[str, Any]:
    case = diagnosis.model_raw_json or {}
    ml = case.get("ml_results") or {}
    seg = case.get("segmentation") or {}
    message = diagnosis.message
    return {
        "diagnosis_id": diagnosis.id,
        "session_id": message.session_id if message else None,
        "created_at": message.created_at.isoformat() if message and message.created_at else None,
        "risk_level": ml.get("risk_level"),
        "tb_probability": ml.get("tb_probability"),
        "primary_label": ml.get("primary_label"),
        "mask_area_fraction": seg.get("mask_area_fraction"),
        "overlay_url": static_url(settings, diagnosis.overlay_path),
    }


def patient_diagnoses(db: Session, patient_id: int) -> list[DBTable.DiagnosisResult]:
    return (
        db.query(DBTable.DiagnosisResult)
        .join(DBTable.Message, DBTable.Message.id == DBTable.DiagnosisResult.message_id)
        .join(DBTable.ChatSession, DBTable.ChatSession.id == DBTable.Message.session_id)
        .filter(DBTable.ChatSession.patient_id == patient_id)
        .order_by(DBTable.Message.created_at.asc(), DBTable.DiagnosisResult.id.asc())
        .all()
    )


def visit_to_dict(visit: DBTable.Visit, settings: Settings) -> dict[str, Any]:
    analyses = [
        diagnosis_brief(msg.diagnosis, settings)
        for session in (visit.sessions or [])
        for msg in (session.messages or [])
        if msg.diagnosis is not None
    ]
    return {
        "id": visit.id,
        "patient_id": visit.patient_id,
        "visited_at": visit.visited_at.isoformat() if visit.visited_at else None,
        "symptoms": visit.symptoms,
        "treatment": visit.treatment,
        "notes": visit.notes,
        "session_count": len(visit.sessions or []),
        "analyses": analyses,
    }


def link_status(link: DBTable.PatientLink, now: datetime | None = None) -> str:
    now = now or datetime.now()
    if link.revoked_at is not None:
        return "revoked"
    if link.expire_policy in {"time", "both"} and link.expires_at is not None and link.expires_at <= now:
        return "expired"
    return "active"


def link_to_dict(link: DBTable.PatientLink, settings: Settings) -> dict[str, Any]:
    return {
        "id": link.id,
        "token": link.token,
        "url": f"{settings.frontend_base_url.rstrip('/')}/share/{link.token}",
        "patient_id": link.patient_id,
        "session_id": link.session_id,
        "doctor_id": link.doctor_id,
        "expire_policy": link.expire_policy,
        "expires_at": link.expires_at.isoformat() if link.expires_at else None,
        "revoked_at": link.revoked_at.isoformat() if link.revoked_at else None,
        "max_questions": link.max_questions,
        "questions_used": link.questions_used,
        "created_at": link.created_at.isoformat() if link.created_at else None,
        "last_accessed_at": link.last_accessed_at.isoformat() if link.last_accessed_at else None,
        "status": link_status(link),
    }


def log_link_access(
    db: Session, link: DBTable.PatientLink, kind: str, request: Request, question: str | None = None
) -> None:
    link.last_accessed_at = datetime.now()
    db.add(
        DBTable.PatientLinkAccess(
            link_id=link.id,
            kind=kind,
            question=question,
            client_ip=request.client.host if request.client else None,
            request_id=getattr(request.state, "request_id", None),
        )
    )
    db.commit()


def resolve_link(db: Session, token: str) -> DBTable.PatientLink:
    link = db.query(DBTable.PatientLink).filter(DBTable.PatientLink.token == token).first()
    if link is None:
        raise HTTPException(status_code=404, detail="링크를 찾을 수 없습니다.")
    if link_status(link) != "active":
        raise HTTPException(status_code=410, detail="이 링크는 만료되었거나 담당 의사가 종료했습니다.")
    return link


def latest_patient_analysis(session: DBTable.ChatSession) -> DBTable.DiagnosisResult | None:
    for msg in reversed(session.messages or []):
        if msg.diagnosis is not None:
            return msg.diagnosis
    return None


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


def validate_and_store_nifti(contents: bytes, filename: str | None, settings: Settings, kind: str) -> str:
    """Validate a NIfTI volume / label map under the CT size limit; return the path relative to static."""
    limit = settings.max_ct_upload_mb * 1024 * 1024
    if len(contents) > limit:
        raise HTTPException(status_code=413, detail=f"CT {kind} exceeds {settings.max_ct_upload_mb} MB")
    if not looks_like_nifti(filename, contents):
        raise HTTPException(status_code=400, detail=f"CT {kind} must be a NIfTI file (.nii or .nii.gz)")
    ext = ".nii.gz" if contents[:2] == b"\x1f\x8b" else ".nii"
    relative = f"uploads/{new_image_id()}_{kind}{ext}"
    (settings.static_dir / relative).write_bytes(contents)
    return relative


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


async def store_analysis_uploads(
    file: UploadFile | None, mask: UploadFile | None, settings: Settings
) -> tuple[str | None, str | None]:
    """Store the uploaded image (PNG/JPEG) or CT volume + mask (NIfTI); return (image_rel, mask_rel)."""
    if file is None:
        if mask is not None:
            raise HTTPException(status_code=400, detail="A CT mask needs the CT volume as `file`")
        return None, None
    contents = await file.read()
    if mask is not None or (file.filename or "").lower().endswith(NIFTI_SUFFIXES):
        if mask is None:
            raise HTTPException(status_code=400, detail="CT volumes need a NIfTI label map as `mask`")
        image_rel = validate_and_store_nifti(contents, file.filename, settings, "volume")
        mask_rel = validate_and_store_nifti(await mask.read(), mask.filename, settings, "mask")
        return image_rel, mask_rel
    return validate_and_store_upload(contents, settings), None


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
    mask_rel: str | None = None,
) -> dict[str, Any]:
    image_id = None
    if image_rel:
        image_id = Path(image_rel).name.split(".")[0]
    return {
        "session_id": session.id,
        "patient_id": patient_id,
        "user_id": user_id,
        "request_id": request_id,
        "text": text,
        "image_path": str(settings.static_dir / image_rel) if image_rel else None,
        "mask_path": str(settings.static_dir / mask_rel) if mask_rel else None,
        "modality": "ct" if mask_rel else "xray",
        "image_id": image_id,
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
    patient = db.query(DBTable.Patient).filter(DBTable.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    allowed = patient.user_id == user.id or (
        auth.is_admin(user) and user.hospital_unit_id is not None and patient.hospital_unit_id == user.hospital_unit_id
    )
    if not allowed:
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
            access_token=auth.create_access_token(user, settings), user=auth.UserOut.from_user(user)
        )

    @app.get("/auth/me", response_model=auth.UserOut)
    def me(current_user: auth.CurrentUser):
        return auth.UserOut.from_user(current_user)

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
            return {"status": "found", "message": "이미 등록된 환자입니다.", "data": patient_to_dict(existing)}

        new_patient = DBTable.Patient(
            name=patient.name,
            birthDate=patient.birthDate,
            gender=patient.gender,
            user_id=current_user.id,
            hospital_unit_id=current_user.hospital_unit_id,
            patient_code=next_patient_code(db),
        )
        db.add(new_patient)
        db.commit()
        db.refresh(new_patient)
        return {"status": "created", "message": "환자가 등록되었습니다.", "data": patient_to_dict(new_patient)}

    @app.get("/patients")
    def get_patients(
        current_user: auth.CurrentUser,
        q: str | None = None,
        gender: str | None = None,
        scope: str = "mine",
        db: Session = Depends(get_db),
    ):
        query = scoped_patients(db, current_user, scope)
        if q:
            pattern = f"%{q.strip()}%"
            query = query.filter((DBTable.Patient.name.ilike(pattern)) | (DBTable.Patient.patient_code.ilike(pattern)))
        if gender in {"M", "F"}:
            query = query.filter(DBTable.Patient.gender == gender)
        return [patient_to_dict(p) for p in query.order_by(DBTable.Patient.created_at.desc()).all()]

    @app.get("/patients/{patient_id}")
    def get_patient(patient_id: int, current_user: auth.CurrentUser, db: Session = Depends(get_db)):
        return patient_to_dict(get_patient_or_404(db, patient_id, current_user))

    # ---------------------------------------------------------------- visits / timeline

    @app.get("/patients/{patient_id}/visits")
    def list_visits(patient_id: int, request: Request, current_user: auth.CurrentUser, db: Session = Depends(get_db)):
        settings: Settings = request.app.state.settings
        get_patient_or_404(db, patient_id, current_user)
        visits = (
            db.query(DBTable.Visit)
            .filter(DBTable.Visit.patient_id == patient_id)
            .order_by(DBTable.Visit.visited_at.desc(), DBTable.Visit.id.desc())
            .all()
        )
        return [visit_to_dict(v, settings) for v in visits]

    @app.post("/patients/{patient_id}/visits")
    def create_visit(
        patient_id: int, body: VisitIn, request: Request, current_user: auth.CurrentUser, db: Session = Depends(get_db)
    ):
        settings: Settings = request.app.state.settings
        get_patient_or_404(db, patient_id, current_user)
        visit = DBTable.Visit(
            patient_id=patient_id,
            visited_at=parse_datetime(body.visited_at) or datetime.now(),
            symptoms=body.symptoms,
            treatment=body.treatment,
            notes=body.notes,
        )
        db.add(visit)
        db.commit()
        db.refresh(visit)
        return visit_to_dict(visit, settings)

    @app.patch("/visits/{visit_id}")
    def update_visit(
        visit_id: int, body: VisitPatch, request: Request, current_user: auth.CurrentUser, db: Session = Depends(get_db)
    ):
        settings: Settings = request.app.state.settings
        visit = db.query(DBTable.Visit).filter(DBTable.Visit.id == visit_id).first()
        if visit is None:
            raise HTTPException(status_code=404, detail="Visit not found")
        get_patient_or_404(db, visit.patient_id, current_user)
        if body.visited_at is not None:
            visit.visited_at = parse_datetime(body.visited_at)
        for field_name in ("symptoms", "treatment", "notes"):
            value = getattr(body, field_name)
            if value is not None:
                setattr(visit, field_name, value)
        db.commit()
        db.refresh(visit)
        return visit_to_dict(visit, settings)

    @app.get("/patients/{patient_id}/timeline")
    def patient_timeline(
        patient_id: int, request: Request, current_user: auth.CurrentUser, db: Session = Depends(get_db)
    ):
        settings: Settings = request.app.state.settings
        get_patient_or_404(db, patient_id, current_user)
        points = []
        for diagnosis in patient_diagnoses(db, patient_id):
            brief = diagnosis_brief(diagnosis, settings)
            session = diagnosis.message.session if diagnosis.message else None
            points.append(
                {
                    "date": brief["created_at"],
                    "diagnosis_id": brief["diagnosis_id"],
                    "session_id": brief["session_id"],
                    "visit_id": session.visit_id if session else None,
                    "tb_probability": brief["tb_probability"],
                    "risk_level": brief["risk_level"],
                    "primary_label": brief["primary_label"],
                    "mask_area_fraction": brief["mask_area_fraction"],
                }
            )
        visits = (
            db.query(DBTable.Visit)
            .filter(DBTable.Visit.patient_id == patient_id)
            .order_by(DBTable.Visit.visited_at.asc())
            .all()
        )
        return {
            "points": points,
            "visits": [
                {"id": v.id, "visited_at": v.visited_at.isoformat() if v.visited_at else None, "symptoms": v.symptoms}
                for v in visits
            ],
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
        visit_id = body.visit_id if body else None
        if visit_id is not None:
            visit = (
                db.query(DBTable.Visit)
                .filter(DBTable.Visit.id == visit_id, DBTable.Visit.patient_id == patient_id)
                .first()
            )
            if visit is None:
                raise HTTPException(status_code=404, detail="Visit not found")
        session = DBTable.ChatSession(patient_id=patient_id, title=title, visit_id=visit_id)
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

    # ---------------------------------------------------------------- patient share links (doctor side)

    @app.post("/patients/{patient_id}/links")
    def create_link(
        patient_id: int,
        body: LinkCreate,
        request: Request,
        current_user: auth.CurrentUser,
        db: Session = Depends(get_db),
    ):
        settings: Settings = request.app.state.settings
        get_patient_or_404(db, patient_id, current_user)
        session = (
            db.query(DBTable.ChatSession)
            .filter(DBTable.ChatSession.id == body.session_id, DBTable.ChatSession.patient_id == patient_id)
            .first()
        )
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        hours = body.hours or settings.share_link_default_hours
        link = DBTable.PatientLink(
            token=secrets.token_urlsafe(32),
            patient_id=patient_id,
            session_id=session.id,
            doctor_id=current_user.id,
            expire_policy=body.expire_policy,
            expires_at=datetime.now() + timedelta(hours=hours) if body.expire_policy in {"time", "both"} else None,
            max_questions=body.max_questions if body.max_questions is not None else settings.share_link_max_questions,
        )
        db.add(link)
        db.commit()
        db.refresh(link)
        logger.info("share link created patient=%s session=%s policy=%s", patient_id, session.id, link.expire_policy)
        return link_to_dict(link, settings)

    @app.get("/patients/{patient_id}/links")
    def list_links(patient_id: int, request: Request, current_user: auth.CurrentUser, db: Session = Depends(get_db)):
        settings: Settings = request.app.state.settings
        get_patient_or_404(db, patient_id, current_user)
        links = (
            db.query(DBTable.PatientLink)
            .filter(DBTable.PatientLink.patient_id == patient_id)
            .order_by(DBTable.PatientLink.created_at.desc())
            .all()
        )
        return [link_to_dict(link, settings) for link in links]

    def get_link_or_404(db: Session, link_id: int, user: DBTable.User) -> DBTable.PatientLink:
        link = db.query(DBTable.PatientLink).filter(DBTable.PatientLink.id == link_id).first()
        if link is None:
            raise HTTPException(status_code=404, detail="Link not found")
        get_patient_or_404(db, link.patient_id, user)
        return link

    @app.delete("/links/{link_id}")
    def revoke_link(link_id: int, request: Request, current_user: auth.CurrentUser, db: Session = Depends(get_db)):
        settings: Settings = request.app.state.settings
        link = get_link_or_404(db, link_id, current_user)
        if link.revoked_at is None:
            link.revoked_at = datetime.now()
            db.commit()
            db.refresh(link)
        return link_to_dict(link, settings)

    @app.get("/links/{link_id}/accesses")
    def link_accesses(link_id: int, current_user: auth.CurrentUser, db: Session = Depends(get_db)):
        link = get_link_or_404(db, link_id, current_user)
        return [
            {
                "accessed_at": a.accessed_at.isoformat() if a.accessed_at else None,
                "kind": a.kind,
                "question": a.question,
                "client_ip": a.client_ip,
            }
            for a in link.accesses
        ]

    # ---------------------------------------------------------------- public share endpoints (no bearer token)

    def share_messages(session: DBTable.ChatSession) -> list[dict[str, Any]]:
        out = []
        for msg in session.messages or []:
            if msg.role == "patient" or (msg.role == "assistant" and (msg.meta or {}).get("audience") == "patient"):
                out.append(
                    {
                        "role": msg.role,
                        "content": msg.content or "",
                        "created_at": msg.created_at.isoformat() if msg.created_at else None,
                    }
                )
        return out

    @app.get("/share/{token}")
    @limiter.limit(lambda: app.state.settings.rate_limit_share)
    def share_view(token: str, request: Request, db: Session = Depends(get_db)):
        settings: Settings = request.app.state.settings
        link = resolve_link(db, token)
        log_link_access(db, link, "view", request)
        session = link.session
        diagnosis = latest_patient_analysis(session)
        follow_ups: list[str] = []
        for msg in reversed(session.messages or []):
            meta = msg.meta or {}
            if msg.role == "assistant" and meta.get("patient_follow_ups"):
                follow_ups = list(meta["patient_follow_ups"])
                break
        analysis = None
        if diagnosis is not None and diagnosis.patient_summary:
            created = diagnosis.message.created_at if diagnosis.message else None
            analysis = {
                "patient_summary": diagnosis.patient_summary,
                "overlay_url": static_url(settings, diagnosis.overlay_path),
                "created_at": created.isoformat() if created else None,
            }
        return {
            "status": "active",
            "patient": {"name": link.patient.name},
            "doctor": {"full_name": link.doctor.full_name if link.doctor else None},
            "session": {
                "title": session.title,
                "created_at": session.created_at.isoformat() if session.created_at else None,
            },
            "analysis": analysis,
            "follow_ups": follow_ups[:2],
            "questions_remaining": max(0, (link.max_questions or 0) - (link.questions_used or 0)),
            "expires_at": link.expires_at.isoformat() if link.expires_at else None,
            "messages": share_messages(session),
        }

    @app.post("/share/{token}/ask")
    @limiter.limit(lambda: app.state.settings.rate_limit_share)
    async def share_ask(token: str, body: ShareQuestion, request: Request, db: Session = Depends(get_db)):
        settings: Settings = request.app.state.settings
        link = resolve_link(db, token)
        remaining = (link.max_questions or 0) - (link.questions_used or 0)
        if remaining <= 0:
            log_link_access(db, link, "denied", request, body.question)
            raise HTTPException(status_code=429, detail="이 링크로 할 수 있는 질문 횟수를 모두 사용했습니다.")
        session = link.session
        request_id = getattr(request.state, "request_id", None)
        chat_graph: ChatGraph = request.app.state.graph
        state = graph_input(
            session=session,
            patient_id=link.patient_id,
            user_id=link.doctor_id,
            request_id=request_id,
            text=body.question,
            image_rel=None,
            settings=settings,
            prior_report=latest_report_for_session(db, session.id),
        )
        state["audience"] = "patient"
        state["session_title_is_default"] = False
        try:
            final = await run_in_threadpool(chat_graph.run, state)
        except llm_report.LLMError as exc:
            logger.error("share graph failed: %s", exc, extra={"request_id": request_id})
            raise HTTPException(status_code=503, detail="답변 생성에 실패했습니다. 잠시 후 다시 시도해주세요.") from exc
        _, _, traces, meta = graph_result_to_turn(final)
        meta["audience"] = "patient"
        meta["via_link_id"] = link.id
        answer_text = final.get("response", "")
        db.add(DBTable.Message(session_id=session.id, role="patient", content=body.question))
        db.commit()
        reply = DBTable.Message(session_id=session.id, role="assistant", content=answer_text, meta=meta)
        db.add(reply)
        link.questions_used = (link.questions_used or 0) + 1
        db.commit()
        db.refresh(reply)
        record_llm_calls(
            db, traces, request_id=request_id, user_id=link.doctor_id, session_id=session.id, diagnosis_id=None
        )
        log_link_access(db, link, "question", request, body.question)
        return {
            "message": {
                "id": reply.id,
                "role": "assistant",
                "content": answer_text,
                "created_at": reply.created_at.isoformat() if reply.created_at else None,
                "meta": {"follow_ups": meta.get("patient_follow_ups", []), "blocked": meta.get("blocked", False)},
            },
            "questions_remaining": max(0, (link.max_questions or 0) - (link.questions_used or 0)),
        }

    # ---------------------------------------------------------------- dashboard

    @app.get("/dashboard/overview")
    def dashboard_overview(
        request: Request, current_user: auth.CurrentUser, days: int = 30, db: Session = Depends(get_db)
    ):
        settings: Settings = request.app.state.settings
        days = max(1, min(days, 365))
        since = datetime.now() - timedelta(days=days)
        scope = "unit" if auth.is_admin(current_user) else "mine"
        patients = scoped_patients(db, current_user, scope).all()
        patient_ids = [p.id for p in patients]
        diagnoses = (
            db.query(DBTable.DiagnosisResult)
            .join(DBTable.Message, DBTable.Message.id == DBTable.DiagnosisResult.message_id)
            .join(DBTable.ChatSession, DBTable.ChatSession.id == DBTable.Message.session_id)
            .filter(DBTable.ChatSession.patient_id.in_(patient_ids), DBTable.Message.created_at >= since)
            .order_by(DBTable.Message.created_at.desc())
            .all()
            if patient_ids
            else []
        )
        labels = {"tb_suspected": 0, "low_tb_likelihood": 0, "no_lesion_detected": 0}
        latencies: list[float] = []
        cost = 0.0
        feedback = {"agree": 0, "disagree": 0}
        by_doctor: dict[int, dict[str, Any]] = {}
        patient_by_id = {p.id: p for p in patients}
        for diagnosis in diagnoses:
            ml = (diagnosis.model_raw_json or {}).get("ml_results") or {}
            label = ml.get("primary_label")
            if label in labels:
                labels[label] += 1
            llm_meta = diagnosis.llm_meta or {}
            if llm_meta.get("latency_ms") is not None:
                latencies.append(float(llm_meta["latency_ms"]))
            cost += float(llm_meta.get("cost_usd") or 0.0)
            if diagnosis.doctor_feedback is True:
                feedback["agree"] += 1
            elif diagnosis.doctor_feedback is False:
                feedback["disagree"] += 1
            session = diagnosis.message.session if diagnosis.message else None
            patient = patient_by_id.get(session.patient_id) if session else None
            if patient is not None:
                row = by_doctor.setdefault(
                    patient.user_id,
                    {
                        "doctor_id": patient.user_id,
                        "analyses": 0,
                        "llm_cost_usd": 0.0,
                        "latencies": [],
                        "last_active": None,
                    },
                )
                row["analyses"] += 1
                row["llm_cost_usd"] += float(llm_meta.get("cost_usd") or 0.0)
                if llm_meta.get("latency_ms") is not None:
                    row["latencies"].append(float(llm_meta["latency_ms"]))
                created = (
                    diagnosis.message.created_at.isoformat()
                    if diagnosis.message and diagnosis.message.created_at
                    else None
                )
                if created and (row["last_active"] is None or created > row["last_active"]):
                    row["last_active"] = created
        recent = []
        for diagnosis in diagnoses[:10]:
            brief = diagnosis_brief(diagnosis, settings)
            session = diagnosis.message.session if diagnosis.message else None
            patient = patient_by_id.get(session.patient_id) if session else None
            recent.append(
                {
                    **brief,
                    "patient_id": patient.id if patient else None,
                    "patient_name": patient.name if patient else None,
                }
            )
        overview: dict[str, Any] = {
            "period_days": days,
            "patients": len(patients),
            "analyses": len(diagnoses),
            "tb_suspected": labels["tb_suspected"],
            "low_likelihood": labels["low_tb_likelihood"],
            "no_lesion": labels["no_lesion_detected"],
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "llm_cost_usd": round(cost, 6),
            "feedback": feedback,
            "recent": recent,
        }
        if auth.is_admin(current_user):
            doctors = (
                db.query(DBTable.User).filter(DBTable.User.hospital_unit_id == current_user.hospital_unit_id).all()
                if current_user.hospital_unit_id is not None
                else []
            )
            rows = []
            for doctor in doctors:
                row = by_doctor.get(
                    doctor.id, {"analyses": 0, "llm_cost_usd": 0.0, "latencies": [], "last_active": None}
                )
                rows.append(
                    {
                        "doctor_id": doctor.id,
                        "full_name": doctor.full_name,
                        "role": doctor.role,
                        "patients": sum(1 for p in patients if p.user_id == doctor.id),
                        "analyses": row["analyses"],
                        "llm_cost_usd": round(row["llm_cost_usd"], 6),
                        "avg_latency_ms": round(sum(row["latencies"]) / len(row["latencies"]), 1)
                        if row["latencies"]
                        else None,
                        "last_active": row["last_active"],
                    }
                )
            overview["by_doctor"] = sorted(rows, key=lambda r: -r["analyses"])
        return overview

    # ---------------------------------------------------------------- admin: accounts in the unit

    def admin_user_to_dict(user: DBTable.User, db: Session) -> dict[str, Any]:
        return {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role or "doctor",
            "is_active": user.is_active if user.is_active is not None else True,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "patients": db.query(func.count(DBTable.Patient.id)).filter(DBTable.Patient.user_id == user.id).scalar()
            or 0,
        }

    @app.get("/admin/users")
    def admin_list_users(admin: auth.AdminUser, db: Session = Depends(get_db)):
        users = (
            db.query(DBTable.User)
            .filter(DBTable.User.hospital_unit_id == admin.hospital_unit_id)
            .order_by(DBTable.User.created_at.asc())
            .all()
        )
        return [admin_user_to_dict(u, db) for u in users]

    @app.post("/admin/users", status_code=201)
    def admin_create_user(body: AdminUserCreate, admin: auth.AdminUser, db: Session = Depends(get_db)):
        email = body.email.lower().strip()
        if db.query(DBTable.User).filter(DBTable.User.email == email).first():
            raise HTTPException(status_code=409, detail="이미 존재하는 이메일입니다.")
        user = DBTable.User(
            email=email,
            full_name=body.full_name,
            hashed_password=auth.hash_password(body.password),
            role=body.role,
            hospital_unit_id=admin.hospital_unit_id,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return admin_user_to_dict(user, db)

    @app.patch("/admin/users/{user_id}")
    def admin_patch_user(user_id: int, body: AdminUserPatch, admin: auth.AdminUser, db: Session = Depends(get_db)):
        user = (
            db.query(DBTable.User)
            .filter(DBTable.User.id == user_id, DBTable.User.hospital_unit_id == admin.hospital_unit_id)
            .first()
        )
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        if user.id == admin.id and body.is_active is False:
            raise HTTPException(status_code=400, detail="자기 자신을 비활성화할 수 없습니다.")
        if body.is_active is not None:
            user.is_active = body.is_active
        if body.role is not None:
            user.role = body.role
        if body.full_name is not None:
            user.full_name = body.full_name
        db.commit()
        db.refresh(user)
        return admin_user_to_dict(user, db)

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
        mask: UploadFile | None = File(None),
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

        image_rel, mask_rel = await store_analysis_uploads(file, mask, settings)
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
                    mask_rel=mask_rel,
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
        mask: UploadFile | None = File(None),
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
        image_rel, mask_rel = await store_analysis_uploads(file, mask, settings)
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
                    mask_rel=mask_rel,
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
                                "modality": payload.get("modality", "xray"),
                                "qc": payload.get("qc"),
                                "image": payload.get("image"),
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
