# LLM 레이어 설계

`WebBack/graph/`, `llm_clients.py`, `llm_report.py`, `kb.py`, `llm_schemas.py`, `guardrails.py`, `llm_pricing.py`, `evals/`

## LangGraph 챗 파이프라인 (한 턴)

```mermaid
flowchart TD
    S([START]) --> V[validate]
    V -->|ok| M[moderate<br/>부적절 질문 차단 + 시맨틱 캐시]
    V -->|error| R[respond]
    M -->|blocked| R
    M --> C[classify<br/>analysis / follow_up / general / chit_chat / out_of_scope]
    C -->|image| A[analyze<br/>U-Net → 게이트 → Radiomics → 분류기]
    C -->|chit-chat · out of scope| R
    C -->|question| RT[retrieve]
    A --> RT
    RT --> K[retrieve_kb<br/>지식베이스 top-k]
    RT --> SC[retrieve_similar<br/>유사 케이스 (Shenzhen 617)]
    RT --> L[retrieve_literature<br/>PubMed E-utilities]
    K --> MG[merge_retrieval]
    SC --> MG
    L --> MG
    MG -->|analysis| G[generate_report<br/>structured outputs]
    MG -->|no analysis| AN[answer<br/>근거 포함 답변]
    G --> E[map_evidence<br/>인용 검증 · 출처 연결]
    E --> D[explain_doctor<br/>임상의용 · 스트리밍]
    E --> P[explain_patient<br/>환자용]
    D --> CP[compare<br/>의사/환자 일관성]
    P --> CP
    CP --> SF[safety_check]
    AN --> SF
    SF --> EN[enrich<br/>후속 질문 · 세션 제목]
    EN --> R
    R --> END([END])
```

| 노드 | 역할 | LLM 호출 |
| --- | --- | --- |
| `validate` | 입력 유무·길이·이미지 존재 | 없음 |
| `moderate` | 범위 밖/유해/개인정보 요청 차단. 질문 임베딩으로 **시맨틱 캐시**(코사인 ≥ 0.92 면 이전 판정 재사용) | 구조화 1회 (캐시 적중 시 0) |
| `classify` | 질문 유형과 검색 필요 여부, PubMed 질의문 | 구조화 1회 |
| `analyze` | 기존 파이프라인 (`cxr_tb`) | 없음 |
| `retrieve_kb` / `retrieve_similar` / `retrieve_literature` | **병렬** super-step. 지식베이스 청크, Radiomics 벡터 최근접 참조 케이스, PubMed 초록 | 임베딩 1회 |
| `generate_report` | 1차 구조화 리포트 (RAG 청크 + 유사 케이스 + 문헌 메타) | 구조화 1회 (+폴백) |
| `map_evidence` | 인용 검증, `evidence_sources` 기록, UI 용 `analysis` 이벤트 발행 | 없음 |
| `explain_doctor` / `explain_patient` | **병렬**. 의사용 요약(스트리밍) · 환자용 설명(쉬운 말, 전문용어·확정 표현 금지) | 각 1회 (+재생성) |
| `compare` | 두 설명의 위험도·소견 일관성. 불일치 시 환자용 재생성 | 구조화 1회 |
| `safety_check` | 확정 진단 표현·의사 판단 귀속 검사, 면책 문구 | 없음 |
| `enrich` | 후속 질문 제안(의사 2–3, 환자 1–2), 첫 분석 시 세션 제목 | 구조화 1–2회 |
| `respond` | 대상(의사/환자)에 맞는 최종 응답 선택 | 없음 |

체크포인트: 턴마다 `s<세션>-<요청id>` 스레드로 SQLite(`GRAPH_CHECKPOINT_PATH`) 또는 PostgreSQL(`DATABASE_URL`)에 저장되어 각 실행의 상태를 추적·재현할 수 있습니다. 비스트리밍 `/analyze` 는 `graph.invoke`, 스트리밍 `/analyze/stream` 은 `graph.stream(stream_mode="custom")` 으로 노드 상태·토큰 이벤트를 SSE 로 중계합니다. `GET /graph` 가 Mermaid 정의를 돌려줍니다.

### 모델 라우팅과 폴백

`llm_clients.LLMRouter` 가 모델명으로 제공자를 고릅니다(`gemini-*` → Gemini, 그 외 → OpenAI). 발표 설계대로 **GPT 기본 + Gemini 폴백(thinking low)** 으로 쓰려면 `GOOGLE_API_KEY` 와 `FALLBACK_PROVIDER=gemini` 만 설정하면 되고, 키가 없으면 OpenAI 대체 모델로 폴백합니다. `LLM_BACKEND=fake` 는 모든 모델을 오프라인 클라이언트로 보내며, 이 클라이언트는 앱이 쓰는 모든 구조화 스키마에 대해 케이스와 일관된 답을 만듭니다.

## 왜 2단계인가

| 단계 | 입력 | 출력 | 모델 (기본) |
| --- | --- | --- | --- |
| 1차 **구조화 리포트** | 케이스 JSON(분할 통계·102 특징·분류 결과) + 분류기 학습 메타데이터 + 검색된 지식베이스 청크 | `RadiomicsReport` (pydantic 스키마) | gpt-4.1, temperature 0.1 |
| 2차 **임상의용 요약** | 1차 리포트 (+ 임상의 메모) | 한국어 요약 150–250 단어 | gpt-4.1-mini, temperature 0.2 |
| 후속 질문 | 질문 + 세션의 최근 리포트 | 한국어 답변 | gpt-4.1-mini |

분석(1차)과 서술(2차)을 분리하면 1차 출력을 **스키마로 강제**하고 근거(`supporting_evidence`)를 기계적으로 검증할 수 있으며, 2차는 어조·길이·언어만 책임집니다. UI 는 1차 JSON 을 구조화된 카드로, 2차 텍스트를 채팅으로 보여 줍니다.

## 구조화 출력 (Structured Outputs)

- `llm_schemas.RadiomicsReport` 를 `client.chat.completions.parse(response_format=...)` 에 그대로 전달합니다. strict 모드라 모든 필드가 필수이며, 값이 없을 수 있는 필드는 nullable 로 선언했습니다.
- 모델이 거부(refusal)하거나 파싱이 실패하면 `LLMError` 로 처리합니다. JSON 문자열을 직접 파싱하던 이전 방식의 "마지막 `}` 까지 잘라서 재시도" 같은 복구 코드는 더 이상 필요 없습니다.

## 지식베이스 RAG

- `WebBack/documents/tb_1~8.txt` (질병 개요·증상·치료·영상 소견·패턴 추론·감별·분할 지표) 를 문단 단위로 묶어 **32개 청크**(최대 700자)로 나눕니다. 청크 id 는 `tb_5#0` 형식으로 안정적입니다.
- 임베딩: `text-embedding-3-small` (fake 백엔드에서는 문자 n-gram 해싱 임베더). 인덱스는 문서 내용 해시로 지문을 찍어 `documents/.kb_index.json` 에 캐시합니다 (문서가 바뀌면 자동 재생성).
- 케이스 JSON → 질의문(`kb.case_query`: 라벨·구역·성분 수·상위 특징군을 한국어 키워드로) → 코사인 상위 k=6 청크만 프롬프트에 주입합니다. 이전에는 8개 문서 전체(약 16K 자)를 매 호출 넣었습니다.
- 모델은 근거 항목의 `json_key` 에 케이스 경로(`ml_results.tb_probability`) 또는 청크 id 를 적습니다. `guardrails.validate_evidence` 가 **존재하지 않는 경로·id 를 삭제**하고 삭제 목록을 `report.validation` 에 남깁니다.

## 신뢰성: 타임아웃 · 재시도 · 폴백

`call_with_resilience(stage, [primary, fallback], attempt)`:

1. 요청마다 `LLM_TIMEOUT_S`(기본 60초).
2. `RateLimitError`, `APIConnectionError`, `APITimeoutError`, 5xx 는 지수 백오프(0.5s·2ⁿ + jitter, 최대 8s)로 `LLM_MAX_RETRIES`(기본 2)회 재시도. 4xx 는 즉시 다음 모델로.
3. 기본 모델이 소진되면 폴백 모델(`REPORT_FALLBACK_MODEL`=gpt-4.1-mini, `SUMMARY_FALLBACK_MODEL`=gpt-4.1)로 같은 절차.
4. 모두 실패하면 `LLMError` → API 는 502 가 아닌 명확한 메시지로 응답하고, 스트리밍에서는 `error` 이벤트.

스트리밍 응답은 중간에 재시도할 수 없으므로 기본 모델만 사용하고, 완료 후 가드레일을 전체 텍스트에 적용한 결과를 `done` 이벤트로 보내 UI 가 초안을 교체합니다.

## 가드레일

- **확정 진단 표현 금지**: `guardrails.DEFINITIVE_PATTERNS` (예: "결핵으로 확진", "definitely TB"). 2차 요약이 걸리면 위반 패턴을 명시한 `[REVISION_REQUEST]` 를 붙여 **한 번 재생성**하고, 그래도 남으면 면책 문구를 덧붙이고 경고 로그를 남깁니다.
- **의사 판단 귀속**: "담당 의사/임상의/physician" 언급이 없으면 면책 문구를 자동으로 추가합니다.
- **근거 검증**: 위 RAG 절의 `validate_evidence`.
- **구역 → 엽 승격 금지**, `no_lesion_detected` 케이스의 병변 서술 금지는 프롬프트 규칙 + 평가 규칙으로 검사합니다.

## 텔레메트리와 비용

호출마다 `LLMTrace` (요청/실제 모델, 지연, 입·출력 토큰, 추정 비용, 시도 횟수, 폴백 여부, 재생성 여부, 프롬프트 버전) 를 만들고:

- `llm_calls` 테이블에 1행씩 저장 (`request_id`, 사용자, 세션, 진단 id 와 연결),
- 진단 결과에는 합산 `llm_meta` 를 저장해 UI 가 "gpt-4.1 → gpt-4.1-mini · 3.2s · 5,120 tokens · $0.004" 를 표시,
- `GET /metrics/llm` 이 단계·모델별 호출 수, 토큰, 비용, 평균 지연, 폴백/재생성 횟수와 의사 피드백 분포를 집계합니다.

비용은 `llm_pricing.py` 의 목록 단가로 추정합니다 (gpt-4.1 $2/$8, gpt-4.1-mini $0.4/$1.6 per 1M tokens). 1차 리포트는 케이스당 입력 약 3–4K 토큰(RAG 적용 후), 출력 약 0.5K 토큰입니다.

프롬프트는 `PROMPT_VERSION` (`2026.11.02-r1`) 으로 버전을 매기고 모든 트레이스·진단 결과에 기록됩니다. 프롬프트를 바꾸면 버전을 올리고 아래 평가를 다시 돌립니다.

## 평가 (evals)

`WebBack/evals/cases/` 에 실제 Shenzhen 영상 17장을 파이프라인에 통과시킨 **고정 케이스 JSON** (TB 8 · 정상 8 · 병변 미검출 1) 이 있습니다. `evals/run_evals.py` 가 두 단계를 실행하고 규칙으로 채점합니다.

| 검사 | 통과 조건 |
| --- | --- |
| `schema` | 1차 출력이 `RadiomicsReport` 스키마를 만족 |
| `risk_consistent` | 리포트 위험도가 파이프라인 위험도와 같거나 한 단계 차이 |
| `no_lesion_handled` | 미검출 케이스: `num_regions = 0`, `tb_probability = null` |
| `evidence` | 검증을 통과한 근거 2개 이상 |
| `grounded_zones` | 리포트 구역 ⊆ 분할 구역 |
| `recommendations` | 임상의 권고 1개 이상 |
| `summary_guardrails` / `summary_physician` / `summary_korean` | 확정 표현 없음 · 의사 판단 귀속 · 한국어 |

### LLM-as-judge 지표 (발표의 평가 지표)

규칙 검사 외에 `evals/judge.py` 가 구조화 출력으로 채점합니다 (`JUDGE_MODEL`, 기본 gpt-4.1-mini; `--no-judge` 로 끌 수 있음).

| 지표 | 대상 | 산출 |
| --- | --- | --- |
| Faithfulness | 의사용 요약 vs 케이스 JSON | 1–5 점 + 근거 없는 주장 목록 |
| Answer Relevance | 후속 질문("추가 검사는?")에 대한 답변 | 1–5 점 |
| Patient Understandability | 환자용 설명 | 1–5 점 + 전문용어 목록 |
| Role Separation Accuracy | 의사용 vs 환자용 | pass/fail + 문제점 |
| Safety Compliance | 두 설명 모두 | pass/fail + 위반 문장 (확정 진단, 치료 권고, 확률의 단정) |
| Missing-lesion handling | `no_lesion_detected` 케이스 | "정상"이라 단정하지 않고 검토를 권고하는지 |
| QC-failure handling | QC 플래그가 있는 케이스 | 신뢰도 저하를 명시하고 검토를 권고하는지 |

케이스 18건 중 QC 실패 케이스 1건(160px 로 축소한 영상)과 미검출 케이스 1건이 이 두 처리 지표를 검사합니다.

```bash
cd WebBack
LLM_BACKEND=fake python evals/run_evals.py          # 배선 점검 (네트워크 없음, judge 는 결정적 가짜 판정)
OPENAI_API_KEY=sk-... python evals/run_evals.py      # 실제 모델 + judge, evals/results.json 생성
python evals/run_evals.py --report_model gpt-4.1-mini  # 저렴한 모델과 비교
python evals/run_evals.py --no-judge                 # 규칙 검사만
```

결과 파일에는 케이스별 통과 여부, 지연, 토큰이 남으므로 모델·프롬프트 변경 전후를 비교할 수 있습니다.
