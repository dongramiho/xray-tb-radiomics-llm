# WebBack

FastAPI 백엔드. 실행 방법과 환경 변수는 루트 [README](../README.md)를 참고.

| 파일 | 역할 |
| --- | --- |
| `main.py` | API 라우트, `/analyze` 처리, 2차 LLM(임상의용 한국어 요약) 호출 |
| `radiomics_llm_report2.py` | 1차 LLM: Radiomics + ML 결과 + TB 지식베이스 → 구조화 JSON 리포트 |
| `DBTable.py`, `database.py` | SQLAlchemy 모델(User / Patient / ChatSession / Message / DiagnosisResult), SQLite |
| `schemas.py` | Pydantic 스키마 |
| `documents/` | TB 지식베이스 (`tb_1.txt` ~ `tb_8.txt`) |

```bash
python -m venv .venv && source .venv/bin/activate
pip install "numpy<2" versioneer && pip install --no-build-isolation -e "../model[inference]"
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
uvicorn main:app --reload --port 8000   # http://127.0.0.1:8000/docs
```
