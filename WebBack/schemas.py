"""Pydantic request / response schemas."""

from pydantic import BaseModel, EmailStr, Field


class PatientCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    birthDate: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD")
    gender: str = Field(pattern=r"^[MF]$")


class PatientResponse(BaseModel):
    id: int
    name: str
    birthDate: str
    gender: str


class SessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    visit_id: int | None = None


class FeedbackIn(BaseModel):
    agree: bool
    note: str | None = Field(default=None, max_length=2000)


class VisitIn(BaseModel):
    visited_at: str | None = Field(default=None, description="ISO datetime or YYYY-MM-DD; defaults to now")
    symptoms: str | None = Field(default=None, max_length=4000)
    treatment: str | None = Field(default=None, max_length=4000)
    notes: str | None = Field(default=None, max_length=4000)


class VisitPatch(BaseModel):
    visited_at: str | None = None
    symptoms: str | None = Field(default=None, max_length=4000)
    treatment: str | None = Field(default=None, max_length=4000)
    notes: str | None = Field(default=None, max_length=4000)


class LinkCreate(BaseModel):
    session_id: int
    expire_policy: str = Field(default="time", pattern=r"^(time|manual|both)$")
    hours: int | None = Field(default=None, ge=1, le=24 * 30)
    max_questions: int | None = Field(default=None, ge=0, le=50)


class ShareQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class AdminUserCreate(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=8, max_length=128)
    role: str = Field(default="doctor", pattern=r"^(doctor|admin)$")


class AdminUserPatch(BaseModel):
    is_active: bool | None = None
    role: str | None = Field(default=None, pattern=r"^(doctor|admin)$")
    full_name: str | None = Field(default=None, min_length=1, max_length=100)
