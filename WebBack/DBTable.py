"""ORM models: doctors, patients, chat sessions, messages, and per-image diagnosis results."""

from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from database import Base


class HospitalUnit(Base):
    """A hospital / department: the tenancy boundary for accounts and patients."""

    __tablename__ = "hospital_units"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, unique=True, index=True)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.now)

    users = relationship("User", back_populates="hospital_unit")
    patients = relationship("Patient", back_populates="hospital_unit")


class User(Base):
    """A clinician (role=doctor) or unit administrator (role=admin) account."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    role = Column(String, default="doctor")  # doctor | admin
    is_active = Column(Boolean, default=True)
    hospital_unit_id = Column(Integer, ForeignKey("hospital_units.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    hospital_unit = relationship("HospitalUnit", back_populates="users")
    patients = relationship("Patient", back_populates="doctor")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))  # attending doctor
    hospital_unit_id = Column(Integer, ForeignKey("hospital_units.id"), nullable=True)
    patient_code = Column(String, unique=True, index=True, nullable=True)  # e.g. P-000042
    name = Column(String, index=True)
    birthDate = Column(String)  # YYYY-MM-DD
    gender = Column(String)  # M / F
    medical_history = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    doctor = relationship("User", back_populates="patients")
    hospital_unit = relationship("HospitalUnit", back_populates="patients")
    sessions = relationship("ChatSession", back_populates="patient")
    visits = relationship("Visit", back_populates="patient", order_by="Visit.visited_at.desc()")
    links = relationship("PatientLink", back_populates="patient")


class Visit(Base):
    """A clinical visit; consultations (sessions) and their analyses hang off a visit."""

    __tablename__ = "visits"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), index=True)
    visited_at = Column(DateTime, default=datetime.now, index=True)
    symptoms = Column(Text, nullable=True)
    treatment = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    patient = relationship("Patient", back_populates="visits")
    sessions = relationship("ChatSession", back_populates="visit")


class ChatSession(Base):
    """One consultation thread for a patient (sidebar entry)."""

    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    visit_id = Column(Integer, ForeignKey("visits.id"), nullable=True)
    title = Column(String)
    created_at = Column(DateTime, default=datetime.now)

    patient = relationship("Patient", back_populates="sessions")
    visit = relationship("Visit", back_populates="sessions")
    messages = relationship("Message", back_populates="session", order_by="Message.created_at")
    links = relationship("PatientLink", back_populates="session")


class PatientLink(Base):
    """A shareable, expiring link that lets the patient read the patient-facing explanation and ask a few questions."""

    __tablename__ = "patient_links"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, unique=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    doctor_id = Column(Integer, ForeignKey("users.id"))
    expire_policy = Column(String, default="time")  # time | manual | both
    expires_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    max_questions = Column(Integer, default=5)
    questions_used = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)
    last_accessed_at = Column(DateTime, nullable=True)

    patient = relationship("Patient", back_populates="links")
    session = relationship("ChatSession", back_populates="links")
    doctor = relationship("User")
    accesses = relationship("PatientLinkAccess", back_populates="link", order_by="PatientLinkAccess.accessed_at")


class PatientLinkAccess(Base):
    """Audit trail of link usage: every view and every question, with time and client address."""

    __tablename__ = "patient_link_accesses"

    id = Column(Integer, primary_key=True, index=True)
    link_id = Column(Integer, ForeignKey("patient_links.id"), index=True)
    accessed_at = Column(DateTime, default=datetime.now)
    kind = Column(String)  # view | question | denied
    question = Column(Text, nullable=True)
    client_ip = Column(String, nullable=True)
    request_id = Column(String, nullable=True)

    link = relationship("PatientLink", back_populates="accesses")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    role = Column(String)  # 'user' (clinician), 'patient' (via share link) or 'assistant' (AI)
    content = Column(Text)
    image_path = Column(String, nullable=True)  # path relative to the static dir, e.g. uploads/<id>.png
    meta = Column(JSON, nullable=True)  # graph outputs: question category, follow-up suggestions, moderation
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
    patient_summary = Column(Text, nullable=True)  # patient-friendly explanation (role-separated output)
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
