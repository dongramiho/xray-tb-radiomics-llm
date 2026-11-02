"""FastAPI entrypoint: authentication, patient registry, consultation chat, and the X-ray analysis endpoint.

Request flow for ``POST /analyze`` with an image:
    upload -> U-Net segmentation -> lesion summary -> radiomics -> TB classifier   (inference service)
           -> stage-1 structured report -> stage-2 clinician summary               (LLM layer)
           -> persisted as Message + DiagnosisResult, returned with overlay URLs.
"""

from __future__ import annotations

import io
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
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

import auth
import database
import DBTable
import llm_report
from inference import AnalysisArtifacts, InferenceService, build_inference_service, new_image_id
from logging_config import configure_logging
from schemas import PatientCreate
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
    app.state.kb_text = llm_report.load_kb_text(settings.kb_dir)
    auth.seed_doctor(settings)
    logger.info(
        "ready: env=%s inference=%s llm=%s report_model=%s summary_model=%s",
        settings.environment,
        app.state.inference.name,
        settings.llm_backend,
        settings.report_model,
        settings.summary_model,
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
        }
    return payload


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
            "models": request.app.state.inference.model_metadata(),
        }

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
    def get_messages(patient_id: int, request: Request, current_user: auth.CurrentUser, db: Session = Depends(get_db)):
        settings: Settings = request.app.state.settings
        get_patient_or_404(db, patient_id, current_user)
        sessions = db.query(DBTable.ChatSession).filter(DBTable.ChatSession.patient_id == patient_id).all()
        msgs: list[dict[str, Any]] = [message_to_dict(m, settings) for s in sessions for m in s.messages]
        msgs.sort(key=lambda x: x["created_at"] or "")
        return msgs

    # ---------------------------------------------------------------- analysis

    @app.post("/analyze")
    @limiter.limit(lambda: app.state.settings.rate_limit_analyze)
    async def analyze(
        request: Request,
        current_user: auth.CurrentUser,
        patient_id: int = Form(...),
        text: str = Form(""),
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

        session = (
            db.query(DBTable.ChatSession)
            .filter(DBTable.ChatSession.patient_id == patient_id)
            .order_by(DBTable.ChatSession.created_at.desc())
            .first()
        )
        if not session:
            session = DBTable.ChatSession(patient_id=patient_id, title=f"Consult {datetime.now():%Y-%m-%d %H:%M}")
            db.add(session)
            db.commit()
            db.refresh(session)

        image_rel: str | None = None
        artifacts: AnalysisArtifacts | None = None
        report: dict[str, Any] | None = None

        if file is not None:
            contents = await file.read()
            image_rel = validate_and_store_upload(contents, settings)
            image_id = Path(image_rel).stem
            artifacts = await run_in_threadpool(inference.analyze, settings.static_dir / image_rel, image_id)
            report = await run_in_threadpool(
                llm_report.generate_radiomics_report,
                artifacts.case,
                client=llm,
                kb_text=request.app.state.kb_text,
                model_meta=inference.model_metadata(),
                model=settings.report_model,
            )
            assistant_text = await run_in_threadpool(
                llm_report.summarize_for_clinician,
                report,
                client=llm,
                model=settings.summary_model,
                clinician_note=text,
            )
        else:
            context = latest_report_for_session(db, session.id)
            assistant_text = await run_in_threadpool(
                llm_report.answer_clinician_question,
                text,
                client=llm,
                model=settings.summary_model,
                context_report=context,
            )

        user_msg = DBTable.Message(session_id=session.id, role="user", content=text, image_path=image_rel)
        db.add(user_msg)
        db.commit()

        ai_msg = DBTable.Message(session_id=session.id, role="assistant", content=assistant_text)
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
                    overlay_path=artifacts.overlay_path,
                    mask_path=artifacts.mask_path,
                )
            )
            db.commit()
            db.refresh(ai_msg)

        logger.info(
            "analyze patient=%s session=%s image=%s label=%s",
            patient_id,
            session.id,
            bool(image_rel),
            artifacts.ml_results.get("primary_label") if artifacts else None,
            extra={"request_id": getattr(request.state, "request_id", None), "user_id": current_user.id},
        )
        return message_to_dict(ai_msg, settings)


app = create_app()
