# WebBack

FastAPI 백엔드. 실행 방법과 환경 변수는 루트 [README](../README.md), 배포는 [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md).

| 파일 | 역할 |
| --- | --- |
| `main.py` | 앱 팩토리(`create_app`), 라우트, `/analyze` 흐름, 요청 ID·접근 로그 미들웨어, 속도 제한 |
| `auth.py` | bcrypt 비밀번호, JWT 발급/검증, `CurrentUser` 의존성, 운영자 발급 계정 시드 |
| `inference.py` | `cxr_tb` 파이프라인 서빙 래퍼, 오버레이/마스크 저장, `INFERENCE_BACKEND=fake` |
| `graph/` | **LangGraph 챗 파이프라인** — `build.py`(노드·라우팅·체크포인터·러너), `state.py`, `moderation.py`(부적절 질문 차단 + 시맨틱 캐시, 질문 분류), `retrievers.py`(유사 케이스, PubMed), `llm_steps.py`(환자용 설명, 일관성 검사, 후속 질문, 제목), `prompts.py` |
| `llm_clients.py` | OpenAI / Gemini(선택) / Fake 클라이언트와 모델명 기반 라우터 |
| `llm_report.py` | 1차 구조화 리포트(structured outputs + RAG) · 임상의용 요약 · 스트리밍, 재시도/폴백 |
| `kb.py` | 지식베이스 청킹·임베딩·검색 (OpenAI 임베딩 / 오프라인 해싱 임베더) |
| `llm_schemas.py` · `guardrails.py` · `llm_pricing.py` | 리포트 pydantic 스키마, 확정 진단 표현·근거 검증, 비용 단가표 |
| `evals/` | 고정 케이스 17건 + `run_evals.py` (규칙 기반 채점) — [docs/LLM.md](../docs/LLM.md) |
| `settings.py` | pydantic-settings, production 설정 검증 |
| `database.py` · `DBTable.py` · `schemas.py` | 엔진/세션, ORM 모델, 요청 스키마 |
| `alembic/` | 스키마 마이그레이션 (기동 시 자동 적용 또는 엔트리포인트) |
| `logging_config.py` | 텍스트 / JSON 로그 |
| `documents/` | TB 지식베이스 (`tb_1.txt` ~ `tb_8.txt`) |
| `tests/` | pytest (fake 백엔드, TensorFlow/OpenAI 불필요) |

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install "numpy<2" versioneer && pip install --no-build-isolation -e "../model[inference]"
pip install -r requirements.txt
cp .env.example .env            # OPENAI_API_KEY 등
uvicorn main:app --reload --port 8000   # http://127.0.0.1:8000/docs
pytest
```
