"""ORM models: doctors, patients, chat sessions, messages, and per-image diagnosis results."""

from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    """A clinician account."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)

    patients = relationship("Patient", back_populates="doctor")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))  # attending doctor
    name = Column(String, index=True)
    birthDate = Column(String)  # YYYY-MM-DD
    gender = Column(String)  # M / F
    medical_history = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    doctor = relationship("User", back_populates="patients")
    sessions = relationship("ChatSession", back_populates="patient")


class ChatSession(Base):
    """One consultation thread for a patient (sidebar entry)."""

    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    title = Column(String)
    created_at = Column(DateTime, default=datetime.now)

    patient = relationship("Patient", back_populates="sessions")
    messages = relationship("Message", back_populates="session", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    role = Column(String)  # 'user' (clinician) or 'assistant' (AI)
    content = Column(Text)
    image_path = Column(String, nullable=True)  # path relative to the static dir, e.g. uploads/<id>.png
    created_at = Column(DateTime, default=datetime.now)

    session = relationship("ChatSession", back_populates="messages")
    diagnosis = relationship("DiagnosisResult", back_populates="message", uselist=False)


class DiagnosisResult(Base):
    """Pipeline + LLM output attached to an assistant message that analysed an image."""

    __tablename__ = "diagnosis_results"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.id"))

    model_raw_json = Column(JSON)  # case JSON: segmentation summary, radiomics features, ml_results
    report_json = Column(JSON, nullable=True)  # stage-1 structured LLM report
    gpt_interpretation = Column(Text)  # stage-2 clinician-facing summary
    overlay_path = Column(String, nullable=True)  # relative to static dir
    mask_path = Column(String, nullable=True)
    doctor_feedback = Column(Boolean, nullable=True)  # clinician agreement, for later model improvement
    doctor_note = Column(Text, nullable=True)  # free-text feedback from the clinician
    llm_meta = Column(JSON, nullable=True)  # models, latency, tokens, cost, prompt version for this analysis

    message = relationship("Message", back_populates="diagnosis")


class LLMCall(Base):
    """One row per LLM request (report / summary / question) for cost and latency telemetry."""

    __tablename__ = "llm_calls"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.now, index=True)
    request_id = Column(String, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=True)
    diagnosis_id = Column(Integer, ForeignKey("diagnosis_results.id"), nullable=True)
    stage = Column(String, index=True)  # report | summary | question
    requested_model = Column(String)
    model = Column(String, index=True)  # model that actually answered (after fallback)
    prompt_version = Column(String)
    latency_ms = Column(Float)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, nullable=True)
    attempts = Column(Integer, default=1)
    fallback_used = Column(Boolean, default=False)
    regenerated = Column(Boolean, default=False)
    success = Column(Boolean, default=True)
    error = Column(Text, nullable=True)
