import { Patient, Message, AuthUser, LoginResponse } from "@/types/index";
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

type RawPatient = { id: number; name: string; birthDate: string; gender: string; created_at?: string; lastVisit?: string };

function toPatient(p: RawPatient): Patient {
  return {
    id: p.id,
    name: p.name,
    birthDate: p.birthDate,
    gender: p.gender,
    lastVisit: p.lastVisit ?? (p.created_at ? new Date(p.created_at).toLocaleDateString() : ""),
  };
}

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

// 1. 환자 목록
export async function fetchPatients(): Promise<Patient[]> {
  const res = await apiFetch("/patients");
  if (!res.ok) throw new Error(await parseError(res, "환자 목록을 불러오지 못했습니다."));
  const data: RawPatient[] = await res.json();
  return data.map(toPatient);
}

// 2. 특정 환자의 대화 이력
export async function fetchMessages(patientId: number): Promise<Message[]> {
  const res = await apiFetch(`/patients/${patientId}/messages`);
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

// 4. 메시지 / X-ray 전송 → 어시스턴트 메시지(분석 결과 포함) 반환
export async function sendMessage(patientId: number, text: string, file: File | null): Promise<Message> {
  const formData = new FormData();
  formData.append("patient_id", patientId.toString());
  formData.append("text", text);
  if (file) formData.append("file", file);

  // Content-Type is left to the browser (multipart boundary); apiFetch only adds Authorization.
  const res = await apiFetch("/analyze", { method: "POST", body: formData });
  if (!res.ok) throw new Error(await parseError(res, "분석 요청에 실패했습니다."));
  return await res.json();
}

// 5. 환자 상세
export async function fetchPatientDetails(patientId: number): Promise<Patient> {
  const res = await apiFetch(`/patients/${patientId}`);
  if (!res.ok) throw new Error(await parseError(res, "환자 정보를 불러오지 못했습니다."));
  return toPatient(await res.json());
}
