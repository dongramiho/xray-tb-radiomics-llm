"""Pydantic request / response schemas."""

from pydantic import BaseModel, Field


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


class FeedbackIn(BaseModel):
    agree: bool
    note: str | None = Field(default=None, max_length=2000)
