"""Authentication: bcrypt password hashing, JWT access tokens, and the current-user dependency.

Clinician accounts are provisioned by an operator (seeded from settings); there is deliberately
no self-service sign-up for a clinical tool.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

import database
import DBTable
from settings import Settings

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)
ALGORITHM = "HS256"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class HospitalUnitOut(BaseModel):
    id: int
    code: str
    name: str


class UserOut(BaseModel):
    id: int
    email: str
    full_name: str | None = None
    role: str = "doctor"
    hospital_unit: HospitalUnitOut | None = None

    @classmethod
    def from_user(cls, user: DBTable.User) -> UserOut:
        unit = user.hospital_unit
        return cls(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role or "doctor",
            hospital_unit=HospitalUnitOut(id=unit.id, code=unit.code, name=unit.name) if unit else None,
        )


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str | None) -> bool:
    if not hashed or not hashed.startswith("$2"):
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(user: DBTable.User, settings: Settings) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.access_token_expire_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def authenticate(db: Session, email: str, password: str) -> DBTable.User | None:
    user = db.query(DBTable.User).filter(DBTable.User.email == email.lower().strip()).first()
    if user is None or not verify_password(password, user.hashed_password):
        return None
    if user.is_active is False:
        return None
    return user


def get_db():
    yield from database.get_db()


def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Session = Depends(get_db),
) -> DBTable.User:
    settings: Settings = request.app.state.settings
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = jwt.decode(credentials.credentials, settings.jwt_secret, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as err:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired") from err
    except jwt.InvalidTokenError as err:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from err
    user = db.query(DBTable.User).filter(DBTable.User.id == int(payload["sub"])).first()
    if user is None or user.is_active is False:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists or is disabled")
    return user


CurrentUser = Annotated[DBTable.User, Depends(get_current_user)]


def require_admin(current_user: CurrentUser) -> DBTable.User:
    if (current_user.role or "doctor") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator role required")
    return current_user


AdminUser = Annotated[DBTable.User, Depends(require_admin)]


def is_admin(user: DBTable.User) -> bool:
    return (user.role or "doctor") == "admin"


def ensure_hospital_unit(db: Session, code: str, name: str) -> DBTable.HospitalUnit:
    unit = db.query(DBTable.HospitalUnit).filter(DBTable.HospitalUnit.code == code).first()
    if unit is None:
        unit = DBTable.HospitalUnit(code=code, name=name)
        db.add(unit)
        db.commit()
        db.refresh(unit)
    return unit


def upsert_account(
    db: Session, *, email: str, password: str, full_name: str, role: str, unit: DBTable.HospitalUnit
) -> DBTable.User:
    email = email.lower().strip()
    user = db.query(DBTable.User).filter(DBTable.User.email == email).first()
    if user is None:
        user = DBTable.User(email=email, full_name=full_name, hashed_password=hash_password(password), role=role)
        db.add(user)
        logger.info("seeded %s account %s", role, email)
    elif not verify_password(password, user.hashed_password):
        user.hashed_password = hash_password(password)
        logger.info("updated password for seeded account %s", email)
    if user.hospital_unit_id is None:
        user.hospital_unit_id = unit.id
    if not user.role:
        user.role = role
    db.commit()
    db.refresh(user)
    return user


def seed_doctor(settings: Settings) -> None:
    """Create or update the operator-provisioned accounts (hospital unit, clinician, optional admin)."""
    if settings.environment == "production" and settings.seed_doctor_password == "changeme":
        raise RuntimeError("SEED_DOCTOR_PASSWORD must be changed from the default in production")
    db = database.SessionLocal()
    try:
        unit = ensure_hospital_unit(db, settings.seed_hospital_code, settings.seed_hospital_name)
        if settings.seed_doctor_email and settings.seed_doctor_password:
            upsert_account(
                db,
                email=settings.seed_doctor_email,
                password=settings.seed_doctor_password,
                full_name=settings.seed_doctor_name,
                role="doctor",
                unit=unit,
            )
        if settings.seed_admin_email and settings.seed_admin_password:
            upsert_account(
                db,
                email=settings.seed_admin_email,
                password=settings.seed_admin_password,
                full_name=settings.seed_admin_name,
                role="admin",
                unit=unit,
            )
        # patients created before hospital units existed inherit their doctor's unit
        orphans = db.query(DBTable.Patient).filter(DBTable.Patient.hospital_unit_id.is_(None)).all()
        for patient in orphans:
            if patient.doctor is not None and patient.doctor.hospital_unit_id is not None:
                patient.hospital_unit_id = patient.doctor.hospital_unit_id
        if orphans:
            db.commit()
    finally:
        db.close()
