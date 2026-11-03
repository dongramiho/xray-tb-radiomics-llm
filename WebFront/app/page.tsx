"use client";

import { useState, useEffect } from "react";
import styles from "./page.module.css";
import { UserPlus } from "lucide-react";

import PatientChat from "@/components/PatientChat";
import PatientModal from "@/components/PatientModal";

import { fetchPatients, registerPatient } from "@/utils/api";
import { Patient } from "@/types";

export default function Home() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [activePatientId, setActivePatientId] = useState<number | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const activePatient = patients.find((p) => p.id === activePatientId);

  // 초기 로드
  useEffect(() => {
    fetchPatients()
      .then(setPatients)
      .catch((err) => console.error(err));
  }, []);

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
        patient_code: pData.patient_code ?? null,
        doctor: pData.doctor ?? null,
      };

      if (result.status === "created") {
        setPatients((prev) => [newPatient, ...prev]);
      }

      setActivePatientId(pData.id);
      setIsModalOpen(false);
      alert(result.message);
    } catch (e: unknown) {
      console.error(e);
      alert(e instanceof Error ? e.message : "환자 등록 실패");
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
        {activePatient ? (
          // 세션 선택 / 스트리밍 전송은 PatientChat 이 담당 (key 로 환자 전환 시 상태 초기화)
          <PatientChat key={activePatient.id} patient={activePatient} />
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
