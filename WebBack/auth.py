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


class UserOut(BaseModel):
    id: int
    email: str
    full_name: str | None = None


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
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists")
    return user


CurrentUser = Annotated[DBTable.User, Depends(get_current_user)]


def seed_doctor(settings: Settings) -> None:
    """Create or update the operator-provisioned clinician account from settings."""
    if not settings.seed_doctor_email or not settings.seed_doctor_password:
        return
    if settings.environment == "production" and settings.seed_doctor_password == "changeme":
        raise RuntimeError("SEED_DOCTOR_PASSWORD must be changed from the default in production")
    db = database.SessionLocal()
    try:
        email = settings.seed_doctor_email.lower().strip()
        user = db.query(DBTable.User).filter(DBTable.User.email == email).first()
        if user is None:
            db.add(
                DBTable.User(
                    email=email,
                    full_name=settings.seed_doctor_name,
                    hashed_password=hash_password(settings.seed_doctor_password),
                )
            )
            logger.info("seeded clinician account %s", email)
        elif not verify_password(settings.seed_doctor_password, user.hashed_password):
            user.hashed_password = hash_password(settings.seed_doctor_password)
            logger.info("updated password for seeded account %s", email)
        db.commit()
    finally:
        db.close()
