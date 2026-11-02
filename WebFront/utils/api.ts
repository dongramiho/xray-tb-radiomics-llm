import { Patient, Message } from "@/types/index";

const BASE_URL = "http://127.0.0.1:8000";

// 1. 환자 목록 가져오기
export async function fetchPatients(): Promise<Patient[]> {
  const res = await fetch(`${BASE_URL}/patients`);
  if (!res.ok) throw new Error("Failed to fetch patients");
  
  const data = await res.json();
  // 백엔드(snake_case) -> 프론트(camelCase) 변환
  return data.map((p: any) => ({
    id: p.id,
    name: p.name,
    birthDate: p.birthDate,
    gender: p.gender,
    lastVisit: new Date(p.created_at).toLocaleDateString(),
  }));
}

// 2. 특정 환자의 메시지 가져오기
export async function fetchMessages(patientId: number): Promise<Message[]> {
  const res = await fetch(`${BASE_URL}/patients/${patientId}/messages`);
  if (!res.ok) throw new Error("Failed to fetch messages");
  return await res.json();
}

// 3. 환자 등록하기
export async function registerPatient(formData: { name: string; birthDate: string; gender: string }) {
  const res = await fetch(`${BASE_URL}/patients/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: formData.name,
      birthDate: formData.birthDate, // 백엔드 형식에 맞춤
      gender: formData.gender,
    }),
  });

  const result = await res.json();
  if (!res.ok) {
    const msg =
      result.detail?.message ||
      result.detail ||
      "Failed to register";

    throw new Error(
      typeof msg === "string" ? msg : JSON.stringify(msg)
    );
  }
  return result; // { status, message, data } 반환
}

// 4. 메시지 및 이미지 전송하기
export async function sendMessage(patientId: number, text: string, file: File | null): Promise<Message> {
  const formData = new FormData();
  formData.append("patient_id", patientId.toString());
  formData.append("text", text);
  if (file) {
    formData.append("file", file);
  }

  const res = await fetch(`${BASE_URL}/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) throw new Error("Failed to send message");
  return await res.json(); // AI 응답 메시지 반환
}

// 5. 환자 상세 정보 가져오기
export async function fetchPatientDetails(patientId: number): Promise<Patient> {
  const res = await fetch(`${BASE_URL}/patients/${patientId}`);
  if (!res.ok) throw new Error("Failed to fetch patient details");

  const data = await res.json();
  // 백엔드(snake_case) -> 프론트(camelCase) 변환
  return {
    id: data.id,
    name: data.name,
    birthDate: data.birthDate,
    gender: data.gender,
    lastVisit: data.lastVisit || new Date(data.created_at).toLocaleDateString(),  // 최근 방문일
  };
}
