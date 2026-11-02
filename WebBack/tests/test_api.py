"""API tests against the fake inference + fake LLM backends."""


def register(client, name="테스트환자", birth="1990-01-01"):
    res = client.post("/patients/register", json={"name": name, "birthDate": birth, "gender": "M"})
    assert res.status_code == 200, res.text
    return res.json()["data"]["id"]


def test_health_reports_backends(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["inference_backend"] == "fake" and body["llm_backend"] == "fake"


def test_register_patient_is_idempotent(client):
    first = client.post("/patients/register", json={"name": "홍길동", "birthDate": "1980-05-05", "gender": "F"}).json()
    second = client.post("/patients/register", json={"name": "홍길동", "birthDate": "1980-05-05", "gender": "F"}).json()
    assert first["status"] == "created" and second["status"] == "found"
    assert first["data"]["id"] == second["data"]["id"]


def test_register_patient_validates_input(client):
    assert (
        client.post("/patients/register", json={"name": "", "birthDate": "1980-05-05", "gender": "F"}).status_code
        == 422
    )
    assert (
        client.post("/patients/register", json={"name": "x", "birthDate": "05/05/1980", "gender": "F"}).status_code
        == 422
    )


def test_analyze_requires_text_or_image(client):
    pid = register(client, "빈요청")
    assert client.post("/analyze", data={"patient_id": pid, "text": "  "}).status_code == 400


def test_analyze_rejects_non_image_upload(client):
    pid = register(client, "가짜파일")
    res = client.post("/analyze", data={"patient_id": pid}, files={"file": ("x.png", b"not an image", "image/png")})
    assert res.status_code == 400


def test_analyze_with_image_returns_message_with_analysis(client, sample_png):
    pid = register(client, "영상환자")
    with sample_png.open("rb") as fh:
        res = client.post(
            "/analyze", data={"patient_id": pid, "text": "기침 2개월"}, files={"file": ("cxr.png", fh, "image/png")}
        )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["role"] == "assistant" and body["content"]
    analysis = body["analysis"]
    assert analysis["overlay_url"].startswith("http://testserver/static/results/")
    assert analysis["ml_results"]["primary_label"] == "tb_suspected"
    assert analysis["segmentation"]["detected"] is True
    assert analysis["report"]["tb_risk_summary"]["level"] in {"low", "moderate", "high"}
    assert "spoofed_user_input" not in body

    # the overlay file really exists and is served
    overlay = client.get(analysis["overlay_url"].replace("http://testserver", ""))
    assert overlay.status_code == 200 and overlay.headers["content-type"] == "image/png"

    # history contains the user turn (with the uploaded image) and the assistant turn (with analysis)
    history = client.get(f"/patients/{pid}/messages").json()
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == "기침 2개월" and history[0]["image_path"].startswith(
        "http://testserver/static/uploads/"
    )
    assert history[1]["analysis"]["diagnosis_id"] == analysis["diagnosis_id"]


def test_follow_up_question_uses_latest_report_as_context(client, sample_png):
    pid = register(client, "후속질문")
    with sample_png.open("rb") as fh:
        client.post("/analyze", data={"patient_id": pid}, files={"file": ("cxr.png", fh, "image/png")})
    llm = client.app.state.llm
    before = len(llm.calls)
    res = client.post("/analyze", data={"patient_id": pid, "text": "추가 검사는 뭐가 필요해?"})
    assert res.status_code == 200
    assert "analysis" not in res.json()
    last_call = llm.calls[-1]
    assert len(llm.calls) == before + 1
    assert "[LATEST_ANALYSIS]" in last_call["user"] and "추가 검사는" in last_call["user"]


def test_unknown_patient(client):
    assert client.post("/analyze", data={"patient_id": 999999, "text": "hi"}).status_code == 404
    assert client.get("/patients/999999").status_code == 404
