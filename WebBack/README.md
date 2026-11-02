# WebBack

FastAPI 백엔드. 실행 방법과 환경 변수는 루트 [README](../README.md), 배포는 [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md).

| 파일 | 역할 |
| --- | --- |
| `main.py` | 앱 팩토리(`create_app`), 라우트, `/analyze` 흐름, 요청 ID·접근 로그 미들웨어, 속도 제한 |
| `auth.py` | bcrypt 비밀번호, JWT 발급/검증, `CurrentUser` 의존성, 운영자 발급 계정 시드 |
| `inference.py` | `cxr_tb` 파이프라인 서빙 래퍼, 오버레이/마스크 저장, `INFERENCE_BACKEND=fake` |
| `llm_report.py` | 1차 구조화 리포트 · 2차 요약 · 후속 질문, OpenAI / `FakeChatClient` |
| `settings.py` | pydantic-settings, production 설정 검증 |
| `database.py` · `DBTable.py` · `schemas.py` | 엔진/세션, ORM 모델, 요청 스키마 |
| `alembic/` | 스키마 마이그레이션 (기동 시 자동 적용 또는 엔트리포인트) |
| `logging_config.py` | 텍스트 / JSON 로그 |
| `documents/` | TB 지식베이스 (`tb_1.txt` ~ `tb_8.txt`) |
| `tests/` | pytest (fake 백엔드, TensorFlow/OpenAI 불필요) |

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install "numpy<2" && pip install --no-build-isolation -e "../model[inference]"
grep -vE '^-e' requirements.txt | pip install -r /dev/stdin
cp .env.example .env            # OPENAI_API_KEY 등
uvicorn main:app --reload --port 8000   # http://127.0.0.1:8000/docs
pytest
```
