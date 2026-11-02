"use client";

import { useState, useEffect } from "react";
import styles from "./page.module.css";
import { UserPlus } from "lucide-react";

import Sidebar from "@/components/Sidebar";
import InputBar from "@/components/InputBar";
import ChatWindow from "@/components/ChatWindow";
import PatientModal from "@/components/PatientModal";

import { fetchPatients, fetchMessages, registerPatient, sendMessage } from "@/utils/api";
import { Patient, Message } from "@/types";

export default function Home() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [activePatientId, setActivePatientId] = useState<number | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const activePatient = patients.find((p) => p.id === activePatientId);

  // 초기 로드
  useEffect(() => {
    fetchPatients()
      .then(setPatients)
      .catch((err) => console.error(err));
  }, []);

  // 환자 선택 → 메시지 로드
  useEffect(() => {
    if (!activePatientId) return;
    fetchMessages(activePatientId)
      .then(setMessages)
      .catch((err) => console.error(err));
  }, [activePatientId]);

  // 환자 등록
  const handleRegister = async (formData: { name: string; birthDate: string; gender: string }) => {
    try {
      const result = await registerPatient(formData);
      const pData = result.data;

      const newPatient: Patient = {
        id: pData.id,
        name: pData.name,
        birthDate: pData.birthDate,
        gender: pData.gender,
        lastVisit: new Date().toLocaleDateString(),
      };

      if (result.status === "created") {
        setPatients((prev) => [newPatient, ...prev]);
      }

      setActivePatientId(pData.id);
      setIsModalOpen(false);
      alert(result.message);
    } catch (e: any) {
      console.error(e);
      alert(e.message || "환자 등록 실패");
    }
  };

  // 메시지 전송
  const handleSend = async (text: string, file: File | null) => {
    if (!activePatientId) return;

    const tempMsg: Message = {
      id: Date.now(),
      role: "user",
      content: text,
      image_path: file ? URL.createObjectURL(file) : undefined,
    };

    setMessages((prev) => [...prev, tempMsg]);

    try {
      const aiMsg = await sendMessage(activePatientId, text, file);
      setMessages((prev) => [...prev, aiMsg]);
    } catch (e) {
      console.error(e);
      alert("전송 실패");
    }
  };

  return (
  
    <div className={styles.layout}>
      {/*<Sidebar
        patients={patients}
        activePatientId={activePatientId}
        onSelectPatient={setActivePatientId}
        onOpenModal={() => setIsModalOpen(true)}
      />*/}

      <main className={styles.main}>
        {activePatientId && activePatient ? (
          <>
            <ChatWindow messages={messages} activePatient={activePatient} />
            <InputBar onSendMessage={handleSend} disabled={false} />
          </>
        ) : (
          <div className={styles.placeholder}>
            <UserPlus size={48} className={styles.placeholderIcon} />
            <p>환자를 선택하거나 등록해주세요.</p>
          </div>
        )}
      </main>

      {isModalOpen && (
        <PatientModal
          onClose={() => setIsModalOpen(false)}
          onSubmit={handleRegister}
        />
      )}
    </div>
  );
}