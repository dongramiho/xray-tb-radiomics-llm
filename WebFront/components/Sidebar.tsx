"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Link from "next/link";
import { useParams, usePathname, useRouter } from "next/navigation";
import styles from "./Sidebar.module.css";
import { UserPlus, Search, LogOut, LayoutDashboard } from "lucide-react";
import { fetchPatients, registerPatient, logout } from "@/utils/api";
import { useStoredUser } from "@/utils/auth";
import { Patient, PatientListParams } from "@/types";
import PatientModal from "./PatientModal";

const SEARCH_DEBOUNCE_MS = 300;
const DASHBOARD_PATH = "/dashboard";

type Scope = NonNullable<PatientListParams["scope"]>;

const SCOPE_OPTIONS: { key: Scope; label: string }[] = [
  { key: "mine", label: "내 환자" },
  { key: "unit", label: "병원 전체" },
];

export default function Sidebar() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [scope, setScope] = useState<Scope>("mine");
  const [isModalOpen, setIsModalOpen] = useState(false);
  // Sequence number so a slow search response cannot overwrite a newer one.
  const requestSeq = useRef(0);

  const router = useRouter();
  const pathname = usePathname();
  const params = useParams();
  const user = useStoredUser();
  const isAdmin = user?.role === "admin";
  // Doctors never send scope (the backend only knows "mine" for them).
  const effectiveScope: Scope | undefined = isAdmin ? scope : undefined;

  const activeId = params?.id ? Number(params.id) : null;

  // Server-side search; an empty query falls back to the full list.
  const loadPatients = useCallback((q: string, scopeParam: Scope | undefined) => {
    const seq = ++requestSeq.current;
    fetchPatients({ q: q || undefined, scope: scopeParam })
      .then((list) => {
        if (seq === requestSeq.current) setPatients(list);
      })
      .catch(console.error);
  }, []);

  // Initial load + debounced search (300ms); empty query loads immediately.
  useEffect(() => {
    const q = searchTerm.trim();
    if (!q) {
      loadPatients("", effectiveScope);
      return;
    }
    const timer = setTimeout(() => loadPatients(q, effectiveScope), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [searchTerm, effectiveScope, loadPatients]);

  const reloadPatients = () => loadPatients(searchTerm.trim(), effectiveScope);

  const handleLogout = () => {
    logout();
    router.push("/login");
  };

  return (
    <>
      <aside className={styles.sidebar}>
        <div className={styles.header}>
          <Link
            href={DASHBOARD_PATH}
            className={`${styles.navLink} ${pathname === DASHBOARD_PATH ? styles.navActive : ""}`}
          >
            <LayoutDashboard size={18} /> 대시보드
          </Link>
          <button
            className={styles.newPatientButton}
            onClick={() => setIsModalOpen(true)}
          >
            <UserPlus size={18} /> 새 환자 등록
          </button>
          <div className={styles.searchWrapper}>
            <Search size={16} className={styles.searchIcon} />
            <input
              type="search"
              placeholder="환자 검색 (이름 / 환자번호)"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className={styles.searchInput}
              aria-label="환자 검색"
            />
          </div>
          {isAdmin && (
            <div className={styles.scopeToggle} role="group" aria-label="환자 범위">
              {SCOPE_OPTIONS.map((opt) => (
                <button
                  key={opt.key}
                  type="button"
                  className={`${styles.scopeButton} ${scope === opt.key ? styles.scopeActive : ""}`}
                  aria-pressed={scope === opt.key}
                  onClick={() => setScope(opt.key)}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className={styles.patientList}>
          {patients.length === 0 && searchTerm.trim() && (
            <p className={styles.emptyState}>검색 결과가 없습니다.</p>
          )}
          {patients.map((p) => (
            <Link
              key={p.id}
              href={`/patient/${p.id}`}
              className={`${styles.patientItem} ${
                activeId === p.id ? styles.activePatient : ""
              }`}
            >
              <div className={styles.patientName}>
                {p.name}
                {p.patient_code && <span className={styles.patientCode}>{p.patient_code}</span>}
              </div>
              <div className={styles.patientSub}>
                {p.birthDate} ({p.gender})
                {effectiveScope === "unit" && p.doctor && ` · ${p.doctor.full_name}`}
              </div>
            </Link>
          ))}
        </div>

        {/* 로그인한 의사 정보 + 로그아웃 */}
        <div className={styles.footer}>
          <div className={styles.userInfo}>
            <div className={styles.userName}>
              {user?.full_name ?? "의료진"}
              {isAdmin && <span className={styles.roleBadge}>관리자</span>}
            </div>
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
              reloadPatients();

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
