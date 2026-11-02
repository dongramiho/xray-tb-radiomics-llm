# 배포 가이드

## 구성 요소

| 서비스 | 이미지 | 역할 | 포트 |
| --- | --- | --- | --- |
| `backend` | `WebBack/Dockerfile` (python:3.11-slim, TensorFlow CPU, PyRadiomics) | FastAPI API, 모델 추론, LLM 호출, 정적 파일(업로드·오버레이) | 8000 |
| `frontend` | `WebFront/Dockerfile` (node:20-alpine, Next.js standalone) | 웹 UI | 3000 |
| `db` | `postgres:16-alpine` | 환자·세션·메시지·진단 결과 | 5432 (내부) |

`docker compose up --build` 가 세 서비스를 띄웁니다. 백엔드 컨테이너의 엔트리포인트는 (1) U-Net 가중치가 없으면 GitHub Release 에서 내려받고, (2) `alembic upgrade head` 로 스키마를 맞추고, (3) uvicorn 을 실행합니다. 이미지는 non-root 사용자로 실행되며 `HEALTHCHECK` 가 `/health` 를 확인합니다.

## 환경 변수 (백엔드)

전체 목록과 기본값은 `WebBack/.env.example`, 구현은 `WebBack/settings.py`.

| 변수 | 설명 | production 요구사항 |
| --- | --- | --- |
| `ENVIRONMENT` | `development` / `staging` / `production` | `production` 이면 아래 검증이 기동 시 수행됨 |
| `OPENAI_API_KEY` | OpenAI 키 | 필수 (`LLM_BACKEND=fake` 불가) |
| `REPORT_MODEL`, `SUMMARY_MODEL` | 1차 리포트 / 2차 요약 모델 | |
| `JWT_SECRET` | HS256 서명 키 | 32자 이상 무작위 문자열 |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 토큰 만료 (기본 720) | |
| `SEED_DOCTOR_EMAIL` / `_PASSWORD` / `_NAME` | 운영자가 발급하는 임상의 계정 | 기본 비밀번호(`changeme`)면 기동 거부 |
| `DATABASE_URL` | SQLAlchemy URL (`postgresql+psycopg://…`, `sqlite:///…`) | PostgreSQL 권장 |
| `AUTO_MIGRATE` | 기동 시 Alembic 적용 (기본 true) | 멀티 워커면 `false` 로 두고 엔트리포인트에서 1회 적용 |
| `PUBLIC_BASE_URL` | 클라이언트가 접근하는 API 주소 (오버레이 URL 생성에 사용) | 실제 도메인 |
| `CORS_ORIGINS` | JSON 배열 | `*` 금지 |
| `STATIC_DIR` | 업로드·결과 PNG 저장 경로 | 볼륨 마운트 |
| `MAX_UPLOAD_MB` | 업로드 상한 (기본 20) | |
| `RATE_LIMIT_ANALYZE` / `RATE_LIMIT_ENABLED` | `/analyze` IP 당 제한 (기본 30/분) | |
| `SEGMENTATION_MODEL_PATH`, `CLASSIFIER_PATH` | 모델 파일 경로 | |
| `INFERENCE_BACKEND`, `LLM_BACKEND` | `real` / `fake` | `fake` 불가 |
| `LOG_JSON` | JSON 로그 (컨테이너) | `true` 권장 |

프론트엔드는 빌드 시 `NEXT_PUBLIC_API_BASE_URL` 하나만 필요합니다 (compose 의 `PUBLIC_BASE_URL` 과 같은 값).

## 모델 아티팩트

| 파일 | 크기 | 위치 | 배포 방법 |
| --- | --- | --- | --- |
| `best_model.h5` (U-Net) | 89MB | `model/outputs/checkpoints/` | GitHub Release `models-v1` 자산. `scripts/download_models.sh` 또는 컨테이너 엔트리포인트가 내려받음. 레포가 private 인 동안은 `GITHUB_TOKEN`(repo 읽기) 또는 `gh` 로그인이 필요 |
| `tb_classifier.joblib` | 1.2MB | `model/outputs/classifier/` | git 에 포함, 이미지에 복사됨 |

가중치를 새로 학습하면 Release 태그를 올리고 `MODEL_RELEASE_TAG` 를 바꿉니다. 분류기는 `python -m cxr_tb.src.train_classifier` 로 재생성해 커밋합니다 (scikit-learn 버전 고정 필요).

## 데이터베이스

- 스키마는 Alembic (`WebBack/alembic/`) 으로 관리합니다. 모델을 바꾸면 `alembic revision --autogenerate -m "..."` 로 리비전을 만들고 커밋합니다.
- CI 는 빈 SQLite 에 `alembic upgrade head` 를 적용해 마이그레이션이 깨지지 않았는지 확인합니다.
- 환자 식별 정보(이름·생년월일)가 저장되므로 디스크 암호화와 접근 통제가 있는 DB 를 사용하고, 백업에도 같은 통제를 적용하세요.

## 운영 체크리스트

- [ ] `ENVIRONMENT=production`, 무작위 `JWT_SECRET`, 기본 비밀번호 변경
- [ ] HTTPS 종단 (Nginx / Caddy / 로드밸런서) 뒤에 배치, `CORS_ORIGINS` 와 `PUBLIC_BASE_URL` 을 실제 도메인으로
- [ ] `STATIC_DIR` 과 PostgreSQL 데이터를 영속 볼륨에, 정기 백업
- [ ] 로그(JSON, `X-Request-ID`)를 수집기로 전송. 접근 로그에는 환자 정보가 포함되지 않지만 LLM 요청 본문은 OpenAI 로 전송됨 — 데이터 처리 계약 확인
- [ ] 리소스: 백엔드 컨테이너 메모리 2GB 이상 (TensorFlow + 모델 로드 시 ~1.2GB), CPU 2 vCPU 이상
- [ ] 모델 변경 시 `docs/evaluation.md` 갱신, Release 태그 증가

## 로컬에서 production 설정 점검

```bash
ENVIRONMENT=production JWT_SECRET=$(openssl rand -hex 32) SEED_DOCTOR_PASSWORD=strong-password \
  OPENAI_API_KEY=sk-... docker compose up --build
```
