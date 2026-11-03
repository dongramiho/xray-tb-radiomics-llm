"""Stage 5 product features: hospital units and roles, visits, share links, dashboard, admin accounts."""

from datetime import datetime, timedelta

from conftest import ADMIN_EMAIL, SEED_EMAIL


def register(client, name, birth="1990-01-01", gender="M"):
    return client.post("/patients/register", json={"name": name, "birthDate": birth, "gender": gender}).json()["data"]


def analyze_image(client, pid, sample_png, text="기침"):
    with sample_png.open("rb") as fh:
        return client.post(
            "/analyze", data={"patient_id": pid, "text": text}, files={"file": ("cxr.png", fh, "image/png")}
        ).json()


def test_me_reports_role_and_unit(client, admin_client):
    me = client.get("/auth/me").json()
    assert me["role"] == "doctor" and me["hospital_unit"]["code"] == "default"
    admin = admin_client.get("/auth/me").json()
    assert admin["role"] == "admin" and admin["hospital_unit"]["id"] == me["hospital_unit"]["id"]


def test_patient_code_and_scoping_between_doctor_and_admin(client, admin_client):
    patient = register(client, "스코프환자", "1985-03-03", "F")
    assert patient["patient_code"].startswith("P-") and patient["doctor"]["full_name"]
    # the doctor's own list contains it; the admin sees it only with the unit scope
    assert any(p["id"] == patient["id"] for p in client.get("/patients").json())
    assert all(p["id"] != patient["id"] for p in admin_client.get("/patients").json())
    unit_list = admin_client.get("/patients", params={"scope": "unit"}).json()
    assert any(p["id"] == patient["id"] and p["doctor"]["full_name"] for p in unit_list)
    assert admin_client.get(f"/patients/{patient['id']}").status_code == 200
    # search by name and by code, gender filter
    assert [p["id"] for p in client.get("/patients", params={"q": "스코프"}).json()] == [patient["id"]]
    assert client.get("/patients", params={"q": patient["patient_code"]}).json()[0]["id"] == patient["id"]
    assert all(p["gender"] == "F" for p in client.get("/patients", params={"gender": "F"}).json())


def test_doctor_cannot_use_admin_endpoints(client):
    assert client.get("/admin/users").status_code == 403
    assert (
        client.post(
            "/admin/users", json={"email": "x@test.com", "full_name": "x", "password": "longenough1"}
        ).status_code
        == 403
    )


def test_admin_manages_unit_accounts(admin_client, anon_client):
    users = admin_client.get("/admin/users").json()
    assert {u["email"] for u in users} >= {SEED_EMAIL, ADMIN_EMAIL}
    created = admin_client.post(
        "/admin/users",
        json={"email": "newdoc@test.com", "full_name": "Dr. New", "password": "strong-pass-1", "role": "doctor"},
    )
    assert created.status_code == 201 and created.json()["role"] == "doctor"
    uid = created.json()["id"]
    assert (
        admin_client.post(
            "/admin/users", json={"email": "newdoc@test.com", "full_name": "x", "password": "strong-pass-1"}
        ).status_code
        == 409
    )
    assert (
        anon_client.post("/auth/login", json={"email": "newdoc@test.com", "password": "strong-pass-1"}).status_code
        == 200
    )
    disabled = admin_client.patch(f"/admin/users/{uid}", json={"is_active": False}).json()
    assert disabled["is_active"] is False
    assert (
        anon_client.post("/auth/login", json={"email": "newdoc@test.com", "password": "strong-pass-1"}).status_code
        == 401
    )
    me = admin_client.get("/auth/me").json()
    assert admin_client.patch(f"/admin/users/{me['id']}", json={"is_active": False}).status_code == 400


def test_visits_sessions_and_timeline(client, sample_png):
    pid = register(client, "방문환자")["id"]
    assert client.get(f"/patients/{pid}/visits").json() == []
    visit = client.post(f"/patients/{pid}/visits", json={"visited_at": "2026-10-01", "symptoms": "기침 3주"}).json()
    assert visit["visited_at"].startswith("2026-10-01") and visit["session_count"] == 0
    session = client.post(f"/patients/{pid}/sessions", json={"title": "1차", "visit_id": visit["id"]}).json()
    assert session["visit_id"] == visit["id"]
    assert client.post(f"/patients/{pid}/sessions", json={"visit_id": 999999}).status_code == 404
    with sample_png.open("rb") as fh:
        client.post(
            "/analyze",
            data={"patient_id": pid, "session_id": session["id"]},
            files={"file": ("cxr.png", fh, "image/png")},
        )
    visits = client.get(f"/patients/{pid}/visits").json()
    assert visits[0]["session_count"] == 1 and len(visits[0]["analyses"]) == 1
    assert visits[0]["analyses"][0]["primary_label"] == "tb_suspected" and visits[0]["analyses"][0]["overlay_url"]
    patched = client.patch(f"/visits/{visit['id']}", json={"treatment": "추적 검사 예약"}).json()
    assert patched["treatment"] == "추적 검사 예약" and patched["symptoms"] == "기침 3주"
    timeline = client.get(f"/patients/{pid}/timeline").json()
    assert len(timeline["points"]) == 1 and timeline["points"][0]["visit_id"] == visit["id"]
    assert timeline["points"][0]["tb_probability"] == 0.81 and timeline["visits"][0]["id"] == visit["id"]


def test_share_link_lifecycle_and_patient_questions(client, anon_client, sample_png):
    pid = register(client, "링크환자")["id"]
    analyze_image(client, pid, sample_png)
    session_id = client.get(f"/patients/{pid}/sessions").json()[0]["id"]
    link = client.post(
        f"/patients/{pid}/links",
        json={"session_id": session_id, "expire_policy": "both", "hours": 48, "max_questions": 2},
    ).json()
    assert link["status"] == "active" and link["url"] == f"http://front.test/share/{link['token']}"
    assert link["expires_at"] and link["questions_used"] == 0 and link["max_questions"] == 2
    assert client.get(f"/patients/{pid}/links").json()[0]["id"] == link["id"]

    # public view: no bearer token
    view = anon_client.get(f"/share/{link['token']}").json()
    assert view["status"] == "active" and view["patient"]["name"] == "링크환자" and view["doctor"]["full_name"]
    assert view["analysis"]["patient_summary"] and view["analysis"]["overlay_url"]
    assert view["questions_remaining"] == 2 and len(view["follow_ups"]) <= 2 and view["messages"] == []

    first = anon_client.post(f"/share/{link['token']}/ask", json={"question": "추가 검사는 어떤 걸 받게 되나요?"})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["questions_remaining"] == 1 and body["message"]["role"] == "assistant"
    assert "담당 의사" in body["message"]["content"] and not body["message"]["meta"]["blocked"]
    second = anon_client.post(
        f"/share/{link['token']}/ask", json={"question": "결과가 나올 때까지 주의할 점은?"}
    ).json()
    assert second["questions_remaining"] == 0
    assert anon_client.post(f"/share/{link['token']}/ask", json={"question": "하나 더"}).status_code == 429

    # the doctor's transcript shows the patient's questions and the link audit trail
    history = client.get(f"/patients/{pid}/messages", params={"session_id": session_id}).json()
    roles = [m["role"] for m in history]
    assert roles.count("patient") == 2 and roles[-1] == "assistant" and history[-1]["meta"]["audience"] == "patient"
    view2 = anon_client.get(f"/share/{link['token']}").json()
    assert [m["role"] for m in view2["messages"]] == ["patient", "assistant", "patient", "assistant"]
    accesses = client.get(f"/links/{link['id']}/accesses").json()
    assert [a["kind"] for a in accesses] == ["view", "question", "question", "denied", "view"]
    assert accesses[1]["question"].startswith("추가 검사")

    # manual revoke -> 410 for the public endpoints
    revoked = client.delete(f"/links/{link['id']}").json()
    assert revoked["status"] == "revoked"
    assert anon_client.get(f"/share/{link['token']}").status_code == 410
    assert anon_client.post(f"/share/{link['token']}/ask", json={"question": "x"}).status_code == 410
    assert anon_client.get("/share/not-a-real-token").status_code == 404


def test_time_expired_link(client, anon_client, sample_png, app):
    import database
    import DBTable

    pid = register(client, "만료환자")["id"]
    analyze_image(client, pid, sample_png)
    session_id = client.get(f"/patients/{pid}/sessions").json()[0]["id"]
    link = client.post(
        f"/patients/{pid}/links", json={"session_id": session_id, "expire_policy": "time", "hours": 1}
    ).json()
    db = database.SessionLocal()
    row = db.get(DBTable.PatientLink, link["id"])
    row.expires_at = datetime.now() - timedelta(minutes=1)
    db.commit()
    db.close()
    assert anon_client.get(f"/share/{link['token']}").status_code == 410
    assert client.get(f"/patients/{pid}/links").json()[0]["status"] == "expired"
    manual = client.post(f"/patients/{pid}/links", json={"session_id": session_id, "expire_policy": "manual"}).json()
    assert manual["expires_at"] is None and manual["status"] == "active"


def test_patient_audience_question_is_moderated(client, anon_client, sample_png):
    pid = register(client, "링크차단")["id"]
    analyze_image(client, pid, sample_png)
    session_id = client.get(f"/patients/{pid}/sessions").json()[0]["id"]
    link = client.post(
        f"/patients/{pid}/links", json={"session_id": session_id, "expire_policy": "manual", "max_questions": 3}
    ).json()
    res = anon_client.post(f"/share/{link['token']}/ask", json={"question": "비트코인 사야 할까요?"}).json()
    assert res["message"]["meta"]["blocked"] is True and "범위" in res["message"]["content"]


def test_dashboard_overview_for_doctor_and_admin(client, admin_client, sample_png):
    pid = register(client, "대시보드환자")["id"]
    body = analyze_image(client, pid, sample_png)
    client.post(f"/diagnoses/{body['analysis']['diagnosis_id']}/feedback", json={"agree": True})
    overview = client.get("/dashboard/overview", params={"days": 7}).json()
    assert overview["period_days"] == 7 and overview["patients"] >= 1 and overview["analyses"] >= 1
    assert overview["tb_suspected"] >= 1 and overview["feedback"]["agree"] >= 1
    assert overview["llm_cost_usd"] > 0 and overview["avg_latency_ms"] is not None
    assert overview["recent"][0]["patient_name"] and overview["recent"][0]["primary_label"]
    assert "by_doctor" not in overview
    admin_view = admin_client.get("/dashboard/overview").json()
    assert admin_view["patients"] >= overview["patients"]
    doctors = {row["full_name"]: row for row in admin_view["by_doctor"]}
    assert any(row["analyses"] >= 1 and row["llm_cost_usd"] > 0 for row in doctors.values())
