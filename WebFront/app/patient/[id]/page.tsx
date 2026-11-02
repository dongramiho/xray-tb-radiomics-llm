"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import PatientChat from "@/components/PatientChat";
import { fetchPatientDetails } from "@/utils/api";
import { Patient } from "@/types";
import styles from "../../page.module.css"; // 스타일 재사용

export default function PatientChatPage() {
  const params = useParams();
  const id = Number(params.id); // URL에서 ID 추출

  const [patient, setPatient] = useState<Patient | null>(null);
  const [loadError, setLoadError] = useState<{ id: number; message: string } | null>(null);

  // 환자 상세 로드 (세션 / 메시지는 PatientChat 이 담당)
  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    fetchPatientDetails(id)
      .then((info) => {
        if (!cancelled) setPatient(info);
      })
      .catch((err: unknown) => {
        console.error("환자 정보 로드 실패:", err);
        if (!cancelled) {
          setLoadError({ id, message: err instanceof Error ? err.message : "환자 정보를 찾을 수 없습니다." });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  // 다른 환자로 이동하는 동안 이전 환자의 화면이 남지 않도록 현재 id 와 맞는 데이터만 사용
  const current = patient && patient.id === id ? patient : null;

  if (!current) {
    const message = loadError && loadError.id === id ? loadError.message : "데이터를 불러오는 중...";
    return (
      <div className={styles.layout}>
        <main className={styles.main}>
          <div className={styles.placeholder}>
            <p>{message}</p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className={styles.layout}>
      {/* 사이드바는 AuthGate 가 전역으로 렌더링 */}
      <main className={styles.main}>
        <PatientChat key={current.id} patient={current} />
      </main>
    </div>
  );
}
