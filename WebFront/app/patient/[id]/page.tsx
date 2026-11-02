"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation"; // 👈 [수정 1] params 가져오는 훅 변경
import ChatWindow from "@/components/ChatWindow";
import InputBar from "@/components/InputBar";
import { fetchMessages, fetchPatientDetails, sendMessage } from "@/utils/api";
import { Patient, Message } from "@/types";
import styles from "../../page.module.css"; // 스타일 재사용 (경로 확인 필요)

export default function PatientChatPage() {
  const params = useParams();
  const id = Number(params.id); // URL에서 ID 추출

  const [messages, setMessages] = useState<Message[]>([]);
  const [patient, setPatient] = useState<Patient | null>(null);
  const [loading, setLoading] = useState(true);

  // 1. 초기 데이터 로드 (환자 목록 + 현재 환자 상세 + 메시지)
  useEffect(() => {
    if (id) {
      Promise.all([
        fetchMessages(id),        // 대화 내용
        fetchPatientDetails(id)   // 현재 환자 정보
      ])
      .then(([msgs, pInfo]) => {
        setMessages(msgs);
        setPatient(pInfo);
      })
      .catch((err) => console.error("데이터 로드 실패:", err))
      .finally(() => setLoading(false));
    }
  }, [id]);

  // 2. 메시지 전송 핸들러
  const handleSendMessage = async (text: string, file: File | null) => {
    if (!patient) return;
    
    // 낙관적 업데이트 (화면에 먼저 표시)
    const tempMsg: Message = {
        id: Date.now(),
        role: "user",
        content: text,
        image_path: file ? URL.createObjectURL(file) : undefined,
        created_at: new Date().toISOString() // 타입 에러 방지용
    };
    setMessages((prev) => [...prev, tempMsg]);

    try {
        const aiMsg = await sendMessage(patient.id, text, file);
        setMessages((prev) => [...prev, aiMsg]);
    } catch (e) {
        console.error(e);
        alert("메시지 전송 실패");
    }
  };

  if (loading) return <div>데이터를 불러오는 중...</div>;
  if (!patient) return <div>환자 정보를 찾을 수 없습니다.</div>;

  return (
    <div className={styles.layout}>
      {/* 사이드바를 여기서도 렌더링하여 ChatGPT처럼 유지 */}
      {/*<Sidebar 
        patients={patientsList} 
        activePatientId={id} 
        onSelectPatient={() => {}} // 내부에서 router.push 하므로 빈 함수 처리
        onOpenModal={() => {}} // 필요 시 구현
      />*/}
      
      <main className={styles.main}>
        <ChatWindow messages={messages} activePatient={patient} />
        <InputBar onSendMessage={handleSendMessage} disabled={false} />
      </main>
    </div>
  );
}