"use client";

import { useId, useState, type FormEvent } from "react";
import { CalendarDays, CalendarPlus, ChevronDown, ChevronRight, LoaderCircle, TrendingUp } from "lucide-react";
import styles from "./VisitsPanel.module.css";
import TrendChart from "./TrendChart";
import { labelKo, RISK_KO } from "./AnalysisCard";
import { formatDate, formatDateTime, formatPercent, todayIsoDate } from "@/utils/format";
import type { RiskLevel, Timeline, Visit, VisitInput } from "@/types";

interface VisitsPanelProps {
  visits: Visit[];
  loading?: boolean;
  timeline: Timeline | null;
  timelineLoading?: boolean;
  /** Creates the visit (the parent owns the list); rejects with an Error on failure. */
  onCreateVisit: (input: VisitInput) => Promise<void>;
  /** Clicking an analysis badge opens the session it belongs to. */
  onSelectSession?: (sessionId: number) => void;
  /** Locks the form while a reply is streaming. */
  disabled?: boolean;
}

const SYMPTOM_EXCERPT = 48;
const TREATMENT_EXCERPT = 30;

function riskClass(level: RiskLevel): string {
  if (level === "high") return styles.riskHigh;
  if (level === "moderate") return styles.riskModerate;
  return styles.riskLow;
}

function excerpt(text: string | null | undefined, max: number): string {
  if (!text) return "";
  const t = text.trim().replace(/\s+/g, " ");
  return t.length > max ? `${t.slice(0, max)}…` : t;
}

// Compact collapsible strip above the session bar: visit list, inline "방문 추가" form and the trend chart.
export default function VisitsPanel({
  visits,
  loading = false,
  timeline,
  timelineLoading = false,
  onCreateVisit,
  onSelectSession,
  disabled = false,
}: VisitsPanelProps) {
  const [listOpen, setListOpen] = useState(false);
  const [chartOpen, setChartOpen] = useState(false);
  const [formOpen, setFormOpen] = useState(false);
  const [visitedAt, setVisitedAt] = useState("");
  const [symptoms, setSymptoms] = useState("");
  const [treatment, setTreatment] = useState("");
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const baseId = useId();

  const openForm = () => {
    if (formOpen) {
      setFormOpen(false);
      return;
    }
    setVisitedAt((prev) => prev || todayIsoDate());
    setFormError(null);
    setFormOpen(true);
    setListOpen(true);
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (saving) return;
    if (!visitedAt) {
      setFormError("방문 날짜를 입력해주세요.");
      return;
    }
    setSaving(true);
    setFormError(null);
    try {
      const input: VisitInput = { visited_at: visitedAt };
      if (symptoms.trim()) input.symptoms = symptoms.trim();
      if (treatment.trim()) input.treatment = treatment.trim();
      await onCreateVisit(input);
      setSymptoms("");
      setTreatment("");
      setVisitedAt("");
      setFormOpen(false);
      setListOpen(true);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : "방문 기록을 추가하지 못했습니다.");
    } finally {
      setSaving(false);
    }
  };

  const latest = visits[0];
  const pointCount = timeline?.points.length ?? 0;

  return (
    <div className={styles.panel}>
      <div className={styles.strip}>
        <button
          type="button"
          className={styles.toggle}
          onClick={() => setListOpen((o) => !o)}
          aria-expanded={listOpen}
          aria-controls={`${baseId}-list`}
        >
          {listOpen ? <ChevronDown size={14} aria-hidden="true" /> : <ChevronRight size={14} aria-hidden="true" />}
          <CalendarDays size={14} aria-hidden="true" /> 방문 기록
          <span className={styles.countBadge}>{loading ? "…" : `${visits.length}건`}</span>
        </button>

        <button
          type="button"
          className={styles.toggle}
          onClick={() => setChartOpen((o) => !o)}
          aria-expanded={chartOpen}
          aria-controls={`${baseId}-chart`}
        >
          {chartOpen ? <ChevronDown size={14} aria-hidden="true" /> : <ChevronRight size={14} aria-hidden="true" />}
          <TrendingUp size={14} aria-hidden="true" /> 변화 추이
          <span className={styles.countBadge}>{timelineLoading ? "…" : `${pointCount}건`}</span>
        </button>

        {!listOpen && latest && (
          <span className={styles.latest} title={latest.symptoms ?? undefined}>
            최근 {formatDate(latest.visited_at)}
            {latest.symptoms && ` · ${excerpt(latest.symptoms, SYMPTOM_EXCERPT)}`}
          </span>
        )}

        <button
          type="button"
          className={styles.addButton}
          onClick={openForm}
          disabled={disabled}
          aria-expanded={formOpen}
          aria-controls={`${baseId}-form`}
        >
          <CalendarPlus size={14} aria-hidden="true" /> 방문 추가
        </button>
      </div>

      {formOpen && (
        <form id={`${baseId}-form`} className={styles.form} onSubmit={handleSubmit}>
          <label className={styles.formField}>
            <span className={styles.formLabel}>방문일</span>
            <input
              type="date"
              className={`${styles.input} ${styles.dateInput}`}
              value={visitedAt}
              onChange={(e) => setVisitedAt(e.target.value)}
              disabled={saving}
              required
            />
          </label>
          <label className={`${styles.formField} ${styles.grow}`}>
            <span className={styles.formLabel}>증상</span>
            <input
              className={styles.input}
              placeholder="예: 2주째 기침, 미열"
              value={symptoms}
              onChange={(e) => setSymptoms(e.target.value)}
              disabled={saving}
            />
          </label>
          <label className={`${styles.formField} ${styles.grow}`}>
            <span className={styles.formLabel}>처치</span>
            <input
              className={styles.input}
              placeholder="예: 객담 검사 의뢰"
              value={treatment}
              onChange={(e) => setTreatment(e.target.value)}
              disabled={saving}
            />
          </label>
          <div className={styles.formActions}>
            <button type="submit" className={`${styles.smallButton} ${styles.smallButtonPrimary}`} disabled={saving}>
              {saving ? <LoaderCircle size={12} className={styles.spinner} aria-hidden="true" /> : null} 저장
            </button>
            <button type="button" className={styles.smallButton} onClick={() => setFormOpen(false)} disabled={saving}>
              취소
            </button>
          </div>
          {formError && (
            <p className={styles.error} role="alert">
              {formError}
            </p>
          )}
        </form>
      )}

      {listOpen && (
        <div id={`${baseId}-list`} className={styles.list}>
          {loading && <p className={styles.muted}>불러오는 중...</p>}
          {!loading && visits.length === 0 && <p className={styles.muted}>아직 방문 기록이 없습니다.</p>}
          {visits.map((v) => (
            <div key={v.id} className={styles.visit}>
              <span className={styles.visitDate}>{formatDate(v.visited_at)}</span>
              <span className={styles.visitSymptoms} title={v.symptoms ?? undefined}>
                {excerpt(v.symptoms, SYMPTOM_EXCERPT) || <span className={styles.muted}>증상 미기재</span>}
              </span>
              {v.treatment && (
                <span className={styles.visitTreatment} title={v.treatment}>
                  처치: {excerpt(v.treatment, TREATMENT_EXCERPT)}
                </span>
              )}
              <span className={styles.visitCounts}>
                상담 {v.session_count} · 분석 {v.analyses.length}
              </span>
              {v.analyses.length > 0 && (
                <span className={styles.badges} aria-label="분석 결과">
                  {v.analyses.map((a) => (
                    <button
                      key={a.diagnosis_id}
                      type="button"
                      className={`${styles.riskBadge} ${riskClass(a.risk_level)}`}
                      title={`${labelKo(a.primary_label)} · 위험도 ${RISK_KO[a.risk_level] ?? a.risk_level} · ${formatDateTime(a.created_at)}${onSelectSession ? " · 상담 열기" : ""}`}
                      onClick={() => onSelectSession?.(a.session_id)}
                      disabled={!onSelectSession}
                    >
                      {a.tb_probability != null ? formatPercent(a.tb_probability, 0) : RISK_KO[a.risk_level] ?? a.risk_level}
                    </button>
                  ))}
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      {chartOpen && (
        <div id={`${baseId}-chart`} className={styles.chart}>
          {timeline ? (
            <TrendChart points={timeline.points} />
          ) : (
            <p className={styles.muted}>{timelineLoading ? "변화 추이를 불러오는 중..." : "변화 추이를 불러오지 못했습니다."}</p>
          )}
        </div>
      )}
    </div>
  );
}
