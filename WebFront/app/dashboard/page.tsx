"use client";

import { useEffect, useId, useState, type FormEvent } from "react";
import Link from "next/link";
import { LayoutDashboard, LoaderCircle, RefreshCw, UserCog, UserPlus } from "lucide-react";
import styles from "./dashboard.module.css";
import { RISK_KO, labelKo } from "@/components/AnalysisCard";
import { createAdminUser, fetchAdminUsers, fetchDashboard, patchAdminUser } from "@/utils/api";
import { useStoredUser } from "@/utils/auth";
import { formatDateTime, formatLatency, formatPercent, formatUsd } from "@/utils/format";
import type { AdminUser, DashboardDoctorUsage, DashboardOverview, DashboardRecent, RiskLevel, UserRole } from "@/types";

const PERIODS = [7, 30, 90];
const DEFAULT_PERIOD = 30;

const ROLE_KO: Record<UserRole, string> = { doctor: "의사", admin: "관리자" };

type Loaded = { days: number; data: DashboardOverview };
type Failed = { days: number; message: string };

function riskClass(level: RiskLevel): string {
  if (level === "high") return styles.riskHigh;
  if (level === "moderate") return styles.riskModerate;
  return styles.riskLow;
}

function StatTile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className={styles.tile}>
      <span className={styles.tileLabel}>{label}</span>
      <span className={styles.tileValue}>{value}</span>
      {sub && <span className={styles.tileSub}>{sub}</span>}
    </div>
  );
}

function RecentTable({ rows }: { rows: DashboardRecent[] }) {
  if (rows.length === 0) return <p className={styles.muted}>이 기간에 수행된 분석이 없습니다.</p>;
  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th scope="col">환자</th>
            <th scope="col">일시</th>
            <th scope="col">분류</th>
            <th scope="col" className={styles.num}>
              결핵 확률
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.diagnosis_id}>
              <td>
                <Link href={`/patient/${r.patient_id}`} className={styles.link}>
                  {r.patient_name}
                </Link>
              </td>
              <td className={styles.num}>{formatDateTime(r.created_at)}</td>
              <td>
                <span className={`${styles.riskBadge} ${riskClass(r.risk_level)}`} title={`위험도 ${RISK_KO[r.risk_level] ?? r.risk_level}`}>
                  {labelKo(r.primary_label)}
                </span>
              </td>
              <td className={styles.num}>{formatPercent(r.tb_probability)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DoctorUsageTable({ rows }: { rows: DashboardDoctorUsage[] }) {
  if (rows.length === 0) return <p className={styles.muted}>이 기간에 활동한 의사가 없습니다.</p>;
  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th scope="col">의사</th>
            <th scope="col" className={styles.num}>
              환자
            </th>
            <th scope="col" className={styles.num}>
              분석
            </th>
            <th scope="col" className={styles.num}>
              LLM 비용
            </th>
            <th scope="col" className={styles.num}>
              평균 지연
            </th>
            <th scope="col">마지막 활동</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((d) => (
            <tr key={d.doctor_id}>
              <td>{d.full_name}</td>
              <td className={styles.num}>{d.patients}</td>
              <td className={styles.num}>{d.analyses}</td>
              <td className={styles.num}>{formatUsd(d.llm_cost_usd)}</td>
              <td className={styles.num}>{formatLatency(d.avg_latency_ms)}</td>
              <td className={styles.num}>{d.last_active ? formatDateTime(d.last_active) : "–"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Admin only: account list with activate / deactivate, and the inline "의사 계정 추가" form.
function AdminUsersSection() {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [email, setEmail] = useState("");
  const [fullName, setFullName] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<UserRole>("doctor");
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [patchingId, setPatchingId] = useState<number | null>(null);
  const baseId = useId();

  useEffect(() => {
    let cancelled = false;
    fetchAdminUsers()
      .then((list) => {
        if (!cancelled) setUsers(list);
      })
      .catch((err: unknown) => {
        if (!cancelled) setLoadError(err instanceof Error ? err.message : "계정 목록을 불러오지 못했습니다.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleCreate = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (submitting) return;
    if (!email.trim() || !fullName.trim() || !password) {
      setFormError("이메일, 이름, 비밀번호를 모두 입력해주세요.");
      return;
    }
    setSubmitting(true);
    setFormError(null);
    setNotice(null);
    try {
      const user = await createAdminUser({ email: email.trim(), full_name: fullName.trim(), password, role });
      setUsers((prev) => [user, ...prev.filter((u) => u.id !== user.id)]);
      setEmail("");
      setFullName("");
      setPassword("");
      setRole("doctor");
      setNotice(`${user.full_name} (${user.email}) 계정을 만들었습니다.`);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : "계정을 만들지 못했습니다.");
    } finally {
      setSubmitting(false);
    }
  };

  const toggleActive = async (user: AdminUser) => {
    if (patchingId !== null) return;
    const next = !user.is_active;
    if (!next && !window.confirm(`${user.full_name} 계정을 비활성화할까요? 로그인할 수 없게 됩니다.`)) return;
    setPatchingId(user.id);
    try {
      const updated = await patchAdminUser(user.id, { is_active: next });
      setUsers((prev) => prev.map((u) => (u.id === user.id ? { ...u, ...updated } : u)));
    } catch (err) {
      alert(err instanceof Error ? err.message : "계정을 수정하지 못했습니다.");
    } finally {
      setPatchingId(null);
    }
  };

  return (
    <section className={styles.section} aria-labelledby={`${baseId}-title`}>
      <h2 id={`${baseId}-title`} className={styles.sectionTitle}>
        <UserCog size={16} aria-hidden="true" /> 계정 관리
      </h2>

      <form className={styles.inlineForm} onSubmit={handleCreate} noValidate>
        <span className={styles.formTitle}>
          <UserPlus size={14} aria-hidden="true" /> 의사 계정 추가
        </span>
        <label className={styles.formField}>
          <span className={styles.formLabel}>이메일</span>
          <input
            type="email"
            autoComplete="off"
            className={styles.input}
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="doctor@hospital.org"
            disabled={submitting}
          />
        </label>
        <label className={styles.formField}>
          <span className={styles.formLabel}>이름</span>
          <input
            className={styles.input}
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            placeholder="홍길동"
            disabled={submitting}
          />
        </label>
        <label className={styles.formField}>
          <span className={styles.formLabel}>비밀번호</span>
          <input
            type="password"
            autoComplete="new-password"
            className={styles.input}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="초기 비밀번호"
            disabled={submitting}
          />
        </label>
        <label className={`${styles.formField} ${styles.formFieldNarrow}`}>
          <span className={styles.formLabel}>역할</span>
          <select className={styles.input} value={role} onChange={(e) => setRole(e.target.value as UserRole)} disabled={submitting}>
            <option value="doctor">의사</option>
            <option value="admin">관리자</option>
          </select>
        </label>
        <button type="submit" className={styles.primaryButton} disabled={submitting}>
          {submitting ? <LoaderCircle size={14} className={styles.spinner} aria-hidden="true" /> : <UserPlus size={14} aria-hidden="true" />}
          추가
        </button>
        {formError && (
          <p className={styles.error} role="alert">
            {formError}
          </p>
        )}
        {notice && <p className={styles.notice}>{notice}</p>}
      </form>

      {loading && <p className={styles.muted}>계정 목록을 불러오는 중...</p>}
      {loadError && <p className={styles.error}>{loadError}</p>}
      {!loading && !loadError && (
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th scope="col">이름</th>
                <th scope="col">이메일</th>
                <th scope="col">역할</th>
                <th scope="col" className={styles.num}>
                  환자
                </th>
                <th scope="col">상태</th>
                <th scope="col">가입일</th>
                <th scope="col"></th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id} className={u.is_active ? "" : styles.inactiveRow}>
                  <td>{u.full_name}</td>
                  <td className={styles.mono}>{u.email}</td>
                  <td>{ROLE_KO[u.role] ?? u.role}</td>
                  <td className={styles.num}>{u.patients}</td>
                  <td>
                    <span className={`${styles.stateBadge} ${u.is_active ? styles.stateActive : styles.stateInactive}`}>
                      {u.is_active ? "활성" : "비활성"}
                    </span>
                  </td>
                  <td className={styles.num}>{formatDateTime(u.created_at)}</td>
                  <td>
                    <button
                      type="button"
                      className={styles.smallButton}
                      onClick={() => toggleActive(u)}
                      disabled={patchingId === u.id}
                    >
                      {patchingId === u.id ? <LoaderCircle size={12} className={styles.spinner} aria-hidden="true" /> : null}
                      {u.is_active ? "비활성화" : "활성화"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export default function DashboardPage() {
  const [days, setDays] = useState(DEFAULT_PERIOD);
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [failed, setFailed] = useState<Failed | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const user = useStoredUser();
  const isAdmin = user?.role === "admin";

  useEffect(() => {
    let cancelled = false;
    fetchDashboard(days)
      .then((data) => {
        if (!cancelled) setLoaded({ days, data });
      })
      .catch((err: unknown) => {
        if (!cancelled) setFailed({ days, message: err instanceof Error ? err.message : "대시보드를 불러오지 못했습니다." });
      });
    return () => {
      cancelled = true;
    };
  }, [days, reloadKey]);

  // Only the response for the selected period is shown; anything else means "loading".
  const data = loaded && loaded.days === days ? loaded.data : null;
  const error = failed && failed.days === days && !data ? failed.message : null;
  const loading = !data && !error;

  const feedbackTotal = data ? data.feedback.agree + data.feedback.disagree : 0;

  return (
    <div className={styles.page}>
      <div className={styles.container}>
        <header className={styles.header}>
          <h1 className={styles.title}>
            <LayoutDashboard size={20} aria-hidden="true" /> 대시보드
          </h1>
          <div className={styles.headerActions}>
            <div className={styles.periodToggle} role="group" aria-label="집계 기간">
              {PERIODS.map((p) => (
                <button
                  key={p}
                  type="button"
                  className={`${styles.periodButton} ${p === days ? styles.periodActive : ""}`}
                  aria-pressed={p === days}
                  onClick={() => setDays(p)}
                >
                  {p}일
                </button>
              ))}
            </div>
            <button
              type="button"
              className={styles.iconButton}
              onClick={() => setReloadKey((k) => k + 1)}
              title="새로고침"
              aria-label="새로고침"
              disabled={loading}
            >
              <RefreshCw size={15} className={loading ? styles.spinner : undefined} />
            </button>
          </div>
        </header>

        {error && (
          <p className={styles.error} role="alert">
            {error}
          </p>
        )}
        {loading && (
          <p className={styles.muted}>
            <LoaderCircle size={14} className={styles.spinner} aria-hidden="true" /> 최근 {days}일 데이터를 불러오는 중...
          </p>
        )}

        {data && (
          <>
            <section className={styles.tiles} aria-label={`최근 ${data.period_days}일 요약`}>
              <StatTile label="환자" value={data.patients.toLocaleString("ko-KR")} sub={`최근 ${data.period_days}일`} />
              <StatTile label="분석" value={data.analyses.toLocaleString("ko-KR")} sub="X-ray 분석 건수" />
              <StatTile label="결핵 의심" value={data.tb_suspected.toLocaleString("ko-KR")} sub={data.analyses > 0 ? formatPercent(data.tb_suspected / data.analyses, 0) : undefined} />
              <StatTile label="가능성 낮음" value={data.low_likelihood.toLocaleString("ko-KR")} sub={data.analyses > 0 ? formatPercent(data.low_likelihood / data.analyses, 0) : undefined} />
              <StatTile label="병변 미검출" value={data.no_lesion.toLocaleString("ko-KR")} sub={data.analyses > 0 ? formatPercent(data.no_lesion / data.analyses, 0) : undefined} />
              <StatTile label="평균 지연" value={formatLatency(data.avg_latency_ms)} sub="분석당 LLM 응답" />
              <StatTile label="LLM 비용" value={formatUsd(data.llm_cost_usd)} sub="기간 합계" />
              <StatTile
                label="피드백 동의 / 비동의"
                value={`${data.feedback.agree} / ${data.feedback.disagree}`}
                sub={feedbackTotal > 0 ? `동의율 ${formatPercent(data.feedback.agree / feedbackTotal, 0)}` : "피드백 없음"}
              />
            </section>

            <section className={styles.section} aria-label="최근 분석">
              <h2 className={styles.sectionTitle}>최근 분석</h2>
              <RecentTable rows={data.recent ?? []} />
            </section>

            {isAdmin && (
              <section className={styles.section} aria-label="의사별 사용 현황">
                <h2 className={styles.sectionTitle}>의사별 사용 현황</h2>
                <DoctorUsageTable rows={data.by_doctor ?? []} />
              </section>
            )}
          </>
        )}

        {isAdmin && <AdminUsersSection />}
      </div>
    </div>
  );
}
