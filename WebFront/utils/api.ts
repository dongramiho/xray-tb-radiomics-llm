import {
  Patient,
  PatientDoctor,
  PatientListParams,
  Message,
  AuthUser,
  LoginResponse,
  Session,
  FeedbackResult,
  StreamHandlers,
  StreamStatus,
  Analysis,
  Visit,
  VisitInput,
  Timeline,
  PatientLink,
  CreateLinkInput,
  LinkAccess,
  ShareView,
  ShareAskResult,
  DashboardOverview,
  AdminUser,
  CreateAdminUserInput,
  PatchAdminUserInput,
} from "@/types/index";
import { clearAuth, getToken, setStoredUser, setToken } from "@/utils/auth";

export const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

const LOGIN_PATH = "/login";

type ApiFetchOptions = {
  /** Treat 401 as an expired session (clear token, go to /login). Off only for the login call itself. */
  redirectOn401?: boolean;
};

// Single entry point for every backend request: attaches the bearer token and handles expired sessions.
export async function apiFetch(
  path: string,
  init: RequestInit = {},
  { redirectOn401 = true }: ApiFetchOptions = {},
): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const res = await fetch(`${BASE_URL}${path}`, { ...init, headers });

  if (res.status === 401 && redirectOn401) {
    clearAuth();
    if (typeof window !== "undefined" && window.location.pathname !== LOGIN_PATH) {
      window.location.assign(LOGIN_PATH);
    }
    throw new Error("로그인이 만료되었습니다. 다시 로그인해주세요.");
  }
  return res;
}

async function parseError(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    if (detail) return JSON.stringify(detail);
  } catch {
    /* non-JSON error body */
  }
  return fallback;
}

// Backend may hand back root-relative media paths (/static/...); point those at the API host.
export function resolveMediaUrl(url: string): string {
  return url.startsWith("/") ? `${BASE_URL}${url}` : url;
}

type RawPatient = {
  id: number;
  name: string;
  birthDate: string;
  gender: string;
  created_at?: string;
  lastVisit?: string;
  patient_code?: string | null;
  doctor?: PatientDoctor | null;
};

function toPatient(p: RawPatient): Patient {
  return {
    id: p.id,
    name: p.name,
    birthDate: p.birthDate,
    gender: p.gender,
    lastVisit: p.lastVisit ?? (p.created_at ? new Date(p.created_at).toLocaleDateString() : ""),
    patient_code: p.patient_code ?? null,
    doctor: p.doctor ?? null,
  };
}

const JSON_HEADERS = { "Content-Type": "application/json" };

// 0-1. 로그인 → 토큰 + 사용자 정보 저장
export async function login(email: string, password: string): Promise<LoginResponse> {
  const res = await apiFetch(
    "/auth/login",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    },
    { redirectOn401: false }, // 401 here means bad credentials, not an expired session
  );
  if (!res.ok) throw new Error(await parseError(res, "이메일 또는 비밀번호가 올바르지 않습니다."));
  const data: LoginResponse = await res.json();
  setToken(data.access_token);
  setStoredUser(data.user);
  return data;
}

// 0-2. 현재 로그인 사용자 (토큰 유효성 확인 겸용)
export async function fetchMe(): Promise<AuthUser> {
  const res = await apiFetch("/auth/me");
  if (!res.ok) throw new Error(await parseError(res, "사용자 정보를 불러오지 못했습니다."));
  const user: AuthUser = await res.json();
  setStoredUser(user);
  return user;
}

// 0-3. 로그아웃 (로컬 저장소만 정리)
export function logout(): void {
  clearAuth();
}

// 1. 환자 목록 (q: 이름 / 환자번호 검색, gender 필터, scope: 관리자의 병원 단위 조회)
export async function fetchPatients(params: PatientListParams = {}): Promise<Patient[]> {
  const search = new URLSearchParams();
  if (params.q?.trim()) search.set("q", params.q.trim());
  if (params.gender) search.set("gender", params.gender);
  if (params.scope) search.set("scope", params.scope);
  const query = search.toString();
  const res = await apiFetch(`/patients${query ? `?${query}` : ""}`);
  if (!res.ok) throw new Error(await parseError(res, "환자 목록을 불러오지 못했습니다."));
  const data: RawPatient[] = await res.json();
  return data.map(toPatient);
}

// 2. 특정 환자의 대화 이력 (sessionId 생략 시 전체)
export async function fetchMessages(patientId: number, sessionId?: number): Promise<Message[]> {
  const query = sessionId != null ? `?session_id=${sessionId}` : "";
  const res = await apiFetch(`/patients/${patientId}/messages${query}`);
  if (!res.ok) throw new Error(await parseError(res, "대화 이력을 불러오지 못했습니다."));
  return await res.json();
}

// 3. 환자 등록
export async function registerPatient(formData: { name: string; birthDate: string; gender: string }) {
  const res = await apiFetch("/patients/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(formData),
  });
  if (!res.ok) throw new Error(await parseError(res, "환자 등록에 실패했습니다."));
  return await res.json(); // { status, message, data }
}

// Multipart body shared by /analyze and /analyze/stream.
//   X-ray: `file` = PNG/JPEG.
//   CT:    `file` = NIfTI volume (.nii / .nii.gz) + `mask` = NIfTI label map (0 background, 1 liver, 2 tumor).
function buildAnalyzeForm(
  patientId: number,
  text: string,
  file: File | null,
  sessionId?: number,
  mask?: File | null,
): FormData {
  const formData = new FormData();
  formData.append("patient_id", patientId.toString());
  formData.append("text", text);
  if (file) formData.append("file", file);
  if (mask) formData.append("mask", mask);
  if (sessionId != null) formData.append("session_id", sessionId.toString());
  return formData;
}

// 4. 메시지 / X-ray / CT 전송 → 어시스턴트 메시지(분석 결과 포함) 반환
export async function sendMessage(
  patientId: number,
  text: string,
  file: File | null,
  sessionId?: number,
  mask?: File | null,
): Promise<Message> {
  // Content-Type is left to the browser (multipart boundary); apiFetch only adds Authorization.
  const res = await apiFetch("/analyze", {
    method: "POST",
    body: buildAnalyzeForm(patientId, text, file, sessionId, mask),
  });
  if (!res.ok) throw new Error(await parseError(res, "분석 요청에 실패했습니다."));
  return await res.json();
}

// 5. 환자 상세
export async function fetchPatientDetails(patientId: number): Promise<Patient> {
  const res = await apiFetch(`/patients/${patientId}`);
  if (!res.ok) throw new Error(await parseError(res, "환자 정보를 불러오지 못했습니다."));
  return toPatient(await res.json());
}

// 6. 상담 세션 목록 (최신순)
export async function fetchSessions(patientId: number): Promise<Session[]> {
  const res = await apiFetch(`/patients/${patientId}/sessions`);
  if (!res.ok) throw new Error(await parseError(res, "상담 목록을 불러오지 못했습니다."));
  return await res.json();
}

// 7. 새 상담 세션 생성 (visitId: 방문 기록에 연결, 선택)
export async function createSession(patientId: number, title?: string, visitId?: number): Promise<Session> {
  const body: { title?: string; visit_id?: number } = {};
  if (title) body.title = title;
  if (visitId != null) body.visit_id = visitId;
  const res = await apiFetch(`/patients/${patientId}/sessions`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await parseError(res, "새 상담을 만들지 못했습니다."));
  return await res.json();
}

// 8. 진단 결과에 대한 의사 피드백 (동의 / 비동의 + 메모)
export async function sendFeedback(diagnosisId: number, agree: boolean, note?: string): Promise<FeedbackResult> {
  const res = await apiFetch(`/diagnoses/${diagnosisId}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(note ? { agree, note } : { agree }),
  });
  if (!res.ok) throw new Error(await parseError(res, "피드백 저장에 실패했습니다."));
  return await res.json();
}

// ---- 9. 스트리밍 분석 (POST /analyze/stream, text/event-stream) ----

/** Thrown when the stream endpoint did not start a stream at all (404 on an older backend, non-SSE reply, ...).
 *  Callers can safely fall back to the non-streaming `sendMessage` in that case. */
export class StreamUnavailableError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "StreamUnavailableError";
    this.status = status;
  }
}

type SseEvent = { event: string; data: string };

// One SSE block ("event: x\ndata: {...}") → { event, data }. Comments / id / retry lines are ignored.
function parseSseBlock(block: string): SseEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  if (data.length === 0) return null;
  return { event, data: data.join("\n") };
}

// Dispatches one event to the handlers. Returns true when the stream is finished (done / error).
function dispatchSseEvent(ev: SseEvent, handlers: StreamHandlers): boolean {
  let payload: unknown;
  try {
    payload = JSON.parse(ev.data);
  } catch {
    console.warn("[stream] malformed event payload", ev);
    return false;
  }
  const body = (payload && typeof payload === "object" ? payload : {}) as Record<string, unknown>;

  switch (ev.event) {
    case "status": {
      // Stage names are open-ended (the pipeline keeps adding them); only `message` is rendered.
      const status: StreamStatus = {
        stage: typeof body.stage === "string" ? body.stage : "",
        message: typeof body.message === "string" ? body.message : "",
      };
      handlers.onStatus(status);
      return false;
    }
    case "analysis":
      handlers.onAnalysis(body as unknown as Analysis);
      return false;
    case "delta":
      if (typeof body.text === "string") handlers.onDelta(body.text);
      return false;
    case "done":
      if (body.message && typeof body.message === "object") handlers.onDone(body.message as Message);
      else handlers.onError("서버 응답(done)에 메시지가 없습니다.");
      return true;
    case "error":
      handlers.onError(typeof body.detail === "string" ? body.detail : "분석 중 오류가 발생했습니다.");
      return true;
    default:
      return false;
  }
}

/**
 * Streams an analysis over SSE. Resolves after the `done` / `error` event (or when the connection drops,
 * which is reported through `onError`). Rejects with `StreamUnavailableError` only when no stream was
 * started, so the caller can fall back to `sendMessage`.
 */
export async function streamAnalysis(
  patientId: number,
  text: string,
  file: File | null,
  sessionId: number | undefined,
  handlers: StreamHandlers,
  mask?: File | null,
): Promise<void> {
  const res = await apiFetch("/analyze/stream", {
    method: "POST",
    headers: { Accept: "text/event-stream" },
    body: buildAnalyzeForm(patientId, text, file, sessionId, mask),
  });
  if (!res.ok) {
    throw new StreamUnavailableError(res.status, await parseError(res, "스트리밍 분석 요청에 실패했습니다."));
  }
  const contentType = res.headers.get("content-type") ?? "";
  if (!res.body || !contentType.includes("text/event-stream")) {
    throw new StreamUnavailableError(res.status, "서버가 스트리밍 응답을 지원하지 않습니다.");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;

  const consume = (chunk: string, flush: boolean) => {
    // Normalise CRLF after concatenation so a "\r" | "\n" split across chunks is still handled.
    buffer = (buffer + chunk).replace(/\r\n/g, "\n");
    let sep = buffer.indexOf("\n\n");
    while (sep !== -1 && !finished) {
      const block = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const ev = parseSseBlock(block);
      if (ev) finished = dispatchSseEvent(ev, handlers);
      sep = buffer.indexOf("\n\n");
    }
    if (flush && !finished && buffer.trim()) {
      const ev = parseSseBlock(buffer);
      buffer = "";
      if (ev) finished = dispatchSseEvent(ev, handlers);
    }
  };

  try {
    while (!finished) {
      const { value, done } = await reader.read();
      if (done) {
        consume(decoder.decode(), true);
        break;
      }
      consume(decoder.decode(value, { stream: true }), false);
    }
  } catch (err) {
    // Transport failure (connection dropped, network error) after the stream started.
    console.warn("[stream] connection failed", err);
    if (!finished) {
      finished = true;
      handlers.onError("스트리밍 연결이 끊어졌습니다. 다시 시도해주세요.");
    }
    return;
  } finally {
    reader.cancel().catch(() => {
      /* already closed */
    });
  }

  if (!finished) handlers.onError("응답이 완료되기 전에 스트림이 종료되었습니다.");
}

// ---- 10. 방문 기록 ----

// 방문 목록 (최신순)
export async function fetchVisits(patientId: number): Promise<Visit[]> {
  const res = await apiFetch(`/patients/${patientId}/visits`);
  if (!res.ok) throw new Error(await parseError(res, "방문 기록을 불러오지 못했습니다."));
  return await res.json();
}

export async function createVisit(patientId: number, input: VisitInput): Promise<Visit> {
  const res = await apiFetch(`/patients/${patientId}/visits`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error(await parseError(res, "방문 기록을 추가하지 못했습니다."));
  return await res.json();
}

export async function updateVisit(visitId: number, input: VisitInput): Promise<Visit> {
  const res = await apiFetch(`/visits/${visitId}`, {
    method: "PATCH",
    headers: JSON_HEADERS,
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error(await parseError(res, "방문 기록을 수정하지 못했습니다."));
  return await res.json();
}

// ---- 11. 변화 추이 (분석 결과 타임라인, 오래된 순) ----
export async function fetchTimeline(patientId: number): Promise<Timeline> {
  const res = await apiFetch(`/patients/${patientId}/timeline`);
  if (!res.ok) throw new Error(await parseError(res, "변화 추이를 불러오지 못했습니다."));
  return await res.json();
}

// ---- 12. 환자 공유 링크 (의사 측) ----

export async function createLink(patientId: number, input: CreateLinkInput): Promise<PatientLink> {
  const res = await apiFetch(`/patients/${patientId}/links`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error(await parseError(res, "공유 링크를 만들지 못했습니다."));
  return await res.json();
}

export async function fetchLinks(patientId: number): Promise<PatientLink[]> {
  const res = await apiFetch(`/patients/${patientId}/links`);
  if (!res.ok) throw new Error(await parseError(res, "공유 링크 목록을 불러오지 못했습니다."));
  return await res.json();
}

export async function revokeLink(linkId: number): Promise<PatientLink> {
  const res = await apiFetch(`/links/${linkId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await parseError(res, "공유 링크를 종료하지 못했습니다."));
  return await res.json();
}

export async function fetchLinkAccesses(linkId: number): Promise<LinkAccess[]> {
  const res = await apiFetch(`/links/${linkId}/accesses`);
  if (!res.ok) throw new Error(await parseError(res, "접속 기록을 불러오지 못했습니다."));
  return await res.json();
}

// ---- 13. 공개 공유 페이지 (환자 측, 토큰 없음) ----

/** Error from the public share endpoints; `status` tells the page which state to show (404 / 410 / 429). */
export class ShareError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ShareError";
    this.status = status;
  }
}

// Plain fetch on purpose: no bearer token, and a 401 / 410 must never redirect the patient to /login.
async function shareFetch<T>(path: string, init: RequestInit, fallback: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, init);
  } catch {
    throw new ShareError(0, "서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.");
  }
  if (!res.ok) throw new ShareError(res.status, await parseError(res, fallback));
  return (await res.json()) as T;
}

export function fetchShare(token: string): Promise<ShareView> {
  return shareFetch<ShareView>(`/share/${encodeURIComponent(token)}`, { method: "GET" }, "공유 링크를 불러오지 못했습니다.");
}

export function askShare(token: string, question: string): Promise<ShareAskResult> {
  return shareFetch<ShareAskResult>(
    `/share/${encodeURIComponent(token)}/ask`,
    { method: "POST", headers: JSON_HEADERS, body: JSON.stringify({ question }) },
    "질문을 보내지 못했습니다.",
  );
}

// ---- 14. 대시보드 ----
export async function fetchDashboard(days: number): Promise<DashboardOverview> {
  const res = await apiFetch(`/dashboard/overview?days=${days}`);
  if (!res.ok) throw new Error(await parseError(res, "대시보드를 불러오지 못했습니다."));
  return await res.json();
}

// ---- 15. 관리자: 계정 관리 (admin 전용, 그 외 403) ----

export async function fetchAdminUsers(): Promise<AdminUser[]> {
  const res = await apiFetch("/admin/users");
  if (!res.ok) throw new Error(await parseError(res, "계정 목록을 불러오지 못했습니다."));
  return await res.json();
}

export async function createAdminUser(input: CreateAdminUserInput): Promise<AdminUser> {
  const res = await apiFetch("/admin/users", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error(await parseError(res, "계정을 만들지 못했습니다."));
  return await res.json();
}

export async function patchAdminUser(userId: number, input: PatchAdminUserInput): Promise<AdminUser> {
  const res = await apiFetch(`/admin/users/${userId}`, {
    method: "PATCH",
    headers: JSON_HEADERS,
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error(await parseError(res, "계정을 수정하지 못했습니다."));
  return await res.json();
}
