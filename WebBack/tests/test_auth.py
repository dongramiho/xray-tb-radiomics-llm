"""Authentication, authorization scoping, and request hygiene."""

from conftest import SEED_EMAIL, SEED_PASSWORD


def test_login_returns_token_and_user(anon_client):
    res = anon_client.post("/auth/login", json={"email": SEED_EMAIL, "password": SEED_PASSWORD})
    assert res.status_code == 200
    body = res.json()
    assert body["token_type"] == "bearer" and body["access_token"]
    assert body["user"]["email"] == SEED_EMAIL


def test_login_rejects_bad_password(anon_client):
    res = anon_client.post("/auth/login", json={"email": SEED_EMAIL, "password": "nope"})
    assert res.status_code == 401


def test_protected_routes_require_token(anon_client):
    assert anon_client.get("/patients").status_code == 401
    assert anon_client.get("/auth/me").status_code == 401
    assert anon_client.post("/analyze", data={"patient_id": 1, "text": "hi"}).status_code == 401
    assert anon_client.get("/patients", headers={"Authorization": "Bearer not-a-jwt"}).status_code == 401


def test_me_returns_current_user(client):
    body = client.get("/auth/me").json()
    assert body["email"] == SEED_EMAIL and "id" in body


def test_patients_are_scoped_to_the_doctor(client, anon_client, app):
    import auth
    import database
    import DBTable

    # another clinician provisioned directly in the DB
    db = database.SessionLocal()
    other = DBTable.User(email="other@test.com", full_name="Dr. Other", hashed_password=auth.hash_password("pw"))
    db.add(other)
    db.commit()
    db.close()
    other_token = anon_client.post("/auth/login", json={"email": "other@test.com", "password": "pw"}).json()[
        "access_token"
    ]
    other_headers = {"Authorization": f"Bearer {other_token}"}

    pid = client.post("/patients/register", json={"name": "범위환자", "birthDate": "1991-02-02", "gender": "F"}).json()[
        "data"
    ]["id"]
    assert any(p["id"] == pid for p in client.get("/patients").json())
    assert all(p["id"] != pid for p in anon_client.get("/patients", headers=other_headers).json())
    assert anon_client.get(f"/patients/{pid}", headers=other_headers).status_code == 404
    assert anon_client.get(f"/patients/{pid}/messages", headers=other_headers).status_code == 404
    assert (
        anon_client.post("/analyze", data={"patient_id": pid, "text": "hi"}, headers=other_headers).status_code == 404
    )


def test_request_id_header_is_set_and_echoed(client):
    res = client.get("/health")
    assert res.headers["X-Request-ID"]
    res = client.get("/health", headers={"X-Request-ID": "trace-123"})
    assert res.headers["X-Request-ID"] == "trace-123"


def test_upload_size_limit(client, tmp_path):
    pid = client.post("/patients/register", json={"name": "큰파일", "birthDate": "1992-03-03", "gender": "M"}).json()[
        "data"
    ]["id"]
    big = b"\x89PNG" + b"0" * (2 * 1024 * 1024)  # 2 MB > MAX_UPLOAD_MB=1 in tests
    res = client.post("/analyze", data={"patient_id": pid}, files={"file": ("big.png", big, "image/png")})
    assert res.status_code == 413


def test_production_settings_are_validated():
    import pytest

    from settings import Settings

    unsafe = Settings(environment="production", jwt_secret="dev-only-change-me", llm_backend="fake")
    with pytest.raises(RuntimeError):
        unsafe.validate_for_environment()
    Settings(environment="development", jwt_secret="dev-only-change-me").validate_for_environment()
