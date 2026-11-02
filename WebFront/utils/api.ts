import { Patient, Message } from "@/types/index";

export const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

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

// 1. 환자 목록
export async function fetchPatients(): Promise<Patient[]> {
  const res = await fetch(`${BASE_URL}/patients`);
  if (!res.ok) throw new Error(await parseError(res, "환자 목록을 불러오지 못했습니다."));
  const data: RawPatient[] = await res.json();
  return data.map(toPatient);
}

// 2. 특정 환자의 대화 이력
export async function fetchMessages(patientId: number): Promise<Message[]> {
  const res = await fetch(`${BASE_URL}/patients/${patientId}/messages`);
  if (!res.ok) throw new Error(await parseError(res, "대화 이력을 불러오지 못했습니다."));
  return await res.json();
}

// 3. 환자 등록
export async function registerPatient(formData: { name: string; birthDate: string; gender: string }) {
  const res = await fetch(`${BASE_URL}/patients/register`, {
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

  const res = await fetch(`${BASE_URL}/analyze`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(await parseError(res, "분석 요청에 실패했습니다."));
  return await res.json();
}

// 5. 환자 상세
export async function fetchPatientDetails(patientId: number): Promise<Patient> {
  const res = await fetch(`${BASE_URL}/patients/${patientId}`);
  if (!res.ok) throw new Error(await parseError(res, "환자 정보를 불러오지 못했습니다."));
  return toPatient(await res.json());
}
