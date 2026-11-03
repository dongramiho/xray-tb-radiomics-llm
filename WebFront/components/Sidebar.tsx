"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import styles from "./Sidebar.module.css";
import { UserPlus, Search, LogOut } from "lucide-react";
// 👇 [중요] registerPatient 함수를 import 해야 합니다!
import { fetchPatients, registerPatient, logout } from "@/utils/api"; 
import { useStoredUser } from "@/utils/auth";
import { Patient } from "@/types";
import PatientModal from "./PatientModal";

export default function Sidebar() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [isModalOpen, setIsModalOpen] = useState(false);

  const router = useRouter();
  const params = useParams();
  const user = useStoredUser();

  const activeId = params?.id ? Number(params.id) : null;

  const loadPatients = useCallback(() => {
    fetchPatients()
      .then(setPatients)
      .catch(console.error);
  }, []);

  // 초기 데이터 로드
  useEffect(() => {
    loadPatients();
  }, [loadPatients]);

  const handleLogout = () => {
    logout();
    router.push("/login");
  };

  return (
    <>
      <aside className={styles.sidebar}>
        <div className={styles.header}>
          <button
            className={styles.newPatientButton}
            onClick={() => setIsModalOpen(true)}
          >
            <UserPlus size={18} /> 새 환자 등록
          </button>
          <div className={styles.searchWrapper}>
            <Search size={16} className={styles.searchIcon} />
            <input
              type="text"
              placeholder="환자 검색..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className={styles.searchInput}
            />
          </div>
        </div>

        <div className={styles.patientList}>
          {patients
            .filter((p) => p.name.includes(searchTerm))
            .map((p) => (
              <Link
                key={p.id}
                href={`/patient/${p.id}`}
                className={`${styles.patientItem} ${
                  activeId === p.id ? styles.activePatient : ""
                }`}
              >
                <div className={styles.patientName}>{p.name}</div>
                <div className={styles.patientSub}>
                  {p.birthDate} ({p.gender})
                </div>
              </Link>
            ))}
        </div>

        {/* 로그인한 의사 정보 + 로그아웃 */}
        <div className={styles.footer}>
          <div className={styles.userInfo}>
            <div className={styles.userName}>{user?.full_name ?? "의료진"}</div>
            <div className={styles.userEmail}>{user?.email ?? ""}</div>
          </div>
          <button
            className={styles.logoutButton}
            onClick={handleLogout}
            title="로그아웃"
          >
            <LogOut size={16} /> 로그아웃
          </button>
        </div>
      </aside>

      {/* 모달 로직 */}
      {isModalOpen && (
        <PatientModal
          onClose={() => setIsModalOpen(false)}
          onSubmit={async (formData) => {
            try {
              // 1. API 호출
              const response = await registerPatient(formData);
              
              // 2. 모달 닫기
              setIsModalOpen(false);

              // 3. 목록 다시 불러오기 (즉시 갱신)
              loadPatients();

              // 4. 방금 등록한 환자 채팅방으로 바로 이동
              if (response && response.data && response.data.id) {
                router.push(`/patient/${response.data.id}`);
              }
              
            } catch (error) {
              console.error("환자 등록 실패:", error);
              alert("등록에 실패했습니다. 다시 시도해주세요.");
            }
          }}
        />
      )}
    </>
  );
}
