from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import shutil
import os
import uuid
import json
from typing import Optional, Dict, Any

# DB
import DBTable
import database

# Radiomics
from radiomics_llm_report2 import generate_radiomics_report, _create_client

# ============================================================
# 초기 설정
# ============================================================

DBTable.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
def startup_event():
    db = database.SessionLocal()
    if not db.query(DBTable.User).first():
        db.add(DBTable.User(
            email="doctor@test.com",
            full_name="Dr. Kim",
            hashed_password="dummy"
        ))
        db.commit()
    db.close()

# ============================================================
# Pydantic
# ============================================================

class PatientCreate(BaseModel):
    name: str
    birthDate: str
    gender: str

# ============================================================
# 환자 API
# ============================================================

@app.post("/patients/register")
def register_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    existing = db.query(DBTable.Patient).filter(
        DBTable.Patient.name == patient.name,
        DBTable.Patient.birthDate == patient.birthDate
    ).first()

    if existing:
        return {"status": "found", "data": existing}

    new_patient = DBTable.Patient(
        name=patient.name,
        birthDate=patient.birthDate,
        gender=patient.gender,
        user_id=1
    )
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    return {"status": "created", "data": new_patient}

@app.get("/patients")
def get_patients(db: Session = Depends(get_db)):
    return db.query(DBTable.Patient).order_by(DBTable.Patient.created_at.desc()).all()

@app.get("/patients/{patient_id}")
def get_patient(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(DBTable.Patient).filter(DBTable.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    return {
        "id": patient.id,
        "name": patient.name,
        "birthDate": patient.birthDate,
        "gender": patient.gender,
        "lastVisit": patient.created_at.strftime("%Y-%m-%d") if patient.created_at else ""
    }

@app.get("/patients/{patient_id}/messages")
def get_messages(patient_id: int, db: Session = Depends(get_db)):
    sessions = db.query(DBTable.ChatSession).filter(
        DBTable.ChatSession.patient_id == patient_id
    ).all()

    msgs = []
    for s in sessions:
        for m in s.messages:
            msgs.append({
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "image_path": m.image_path,
                "created_at": m.created_at
            })

    msgs.sort(key=lambda x: x["created_at"])
    return msgs

# ============================================================
# GPT
# ============================================================

client = _create_client()

SYSTEM_PROMPT = """
You are a medical AI assistant supporting a clinician.  
The input will be an AI-generated analysis in JSON format from real diagnosis data.  
Based on this input, provide an interpretation-focused summary covering:  
- JSON content interpretation  
- Risk level assessment  
- Recommendations for clinicians  

Requirements:  
- Do NOT discuss or mention the model’s performance.  
- Do NOT provide definitive diagnoses.  
- Focus on likelihoods, patterns, risk indications, and clinical considerations.  
- Clearly state that the final clinical judgment belongs to the physician shortly.  
- The output must be written in Korean.  


"""

def call_medical_chat(user_input: str) -> str:
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ],
        temperature=0.2,
        max_tokens=1000,
    )
    return completion.choices[0].message.content

# ============================================================
# Radiomics DEMO
# ============================================================

def run_radiomics_pipeline(image_path: str) -> dict:
    return {
        "image": {"path": image_path},
        "ml_results": {
            "label": "tb_suspected",
            "probability": 0.82,
            "risk_score": 0.78
        }
    }

# ============================================================
# Root
# ============================================================

@app.get("/")
def root():
    return {"message": "Radiomics CDS API"}

# ============================================================
# ANALYZE 
# ============================================================

@app.post("/analyze")
async def analyze(
    patient_id: int = Form(...),
    text: str = Form(""),
    file: UploadFile = File(None),
    db: Session = Depends(get_db),
):
    # 세션
    session = db.query(DBTable.ChatSession)\
        .filter(DBTable.ChatSession.patient_id == patient_id)\
        .order_by(DBTable.ChatSession.created_at.desc())\
        .first()

    if not session:
        session = DBTable.ChatSession(
            patient_id=patient_id,
            title=f"Consult ({uuid.uuid4().hex[:6]})"
        )
        db.add(session)
        db.commit()
        db.refresh(session)

    # 이미지 저장
    image_path = None
    image_url = None

    if file:
        fname = f"{uuid.uuid4()}_{file.filename}"
        image_path = f"static/uploads/{fname}"
        with open(image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        image_url = f"http://127.0.0.1:8000/{image_path}"

    diagnosis = None

    # =========================
    # 이미지 → Radiomics → JSON → user input
    # =========================
    if image_path:
        case = run_radiomics_pipeline(image_path)
        report = generate_radiomics_report(case)

        report_json_str = json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        )

        # 👇 눈속임 핵심
        user_input = report_json_str
        user_content_for_db = report_json_str

    else:
        user_input = text.strip()
        user_content_for_db = text.strip()

    # user 메시지 저장
    user_msg = DBTable.Message(
        session_id=session.id,
        role="user",
        content="",
        image_path=image_url
    )
    db.add(user_msg)
    db.commit()

    # GPT
    ai_text = call_medical_chat(user_input)

    ai_msg = DBTable.Message(
        session_id=session.id,
        role="assistant",
        content=ai_text
    )
    db.add(ai_msg)
    db.commit()
    db.refresh(ai_msg)

    # Radiomics 원본 저장
    if image_path:
        db.add(DBTable.DiagnosisResult(
            message_id=ai_msg.id,
            model_raw_json=report,
            gpt_interpretation=ai_text
        ))
        db.commit()

    return {
        "session_id": session.id,
        "assistant_response": ai_text,
        "spoofed_user_input": user_input
    }
