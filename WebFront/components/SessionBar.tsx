"use client";

import { useState } from "react";
import { LoaderCircle, MessageSquarePlus, MessagesSquare, Share2 } from "lucide-react";
import styles from "./SessionBar.module.css";
import ShareLinkModal from "./ShareLinkModal";
import { formatDate, formatDateTime } from "@/utils/format";
import { Session, Visit } from "@/types";

interface SessionBarProps {
  patientId: number;
  sessions: Session[];
  activeSessionId: number | null;
  onSelect: (sessionId: number) => void;
  onCreate: () => void;
  creating?: boolean;
  /** Locks the controls while a reply is streaming. */
  disabled?: boolean;
  /** Visits offered in the "새 상담" flow; the chosen one is attached to the new session. */
  visits?: Visit[];
  newSessionVisitId?: number | null;
  onNewSessionVisitChange?: (visitId: number | null) => void;
}

const VISIT_EXCERPT = 18;

function visitLabel(visit: Visit): string {
  const date = formatDate(visit.visited_at);
  const symptoms = visit.symptoms?.trim().replace(/\s+/g, " ") ?? "";
  if (!symptoms) return date;
  return `${date} · ${symptoms.length > VISIT_EXCERPT ? `${symptoms.slice(0, VISIT_EXCERPT)}…` : symptoms}`;
}

export default function SessionBar({
  patientId,
  sessions,
  activeSessionId,
  onSelect,
  onCreate,
  creating = false,
  disabled = false,
  visits = [],
  newSessionVisitId = null,
  onNewSessionVisitChange,
}: SessionBarProps) {
  const [shareOpen, setShareOpen] = useState(false);
  const active = sessions.find((s) => s.id === activeSessionId);
  const locked = disabled || creating;

  return (
    <div className={styles.bar}>
      <span className={styles.label}>
        <MessagesSquare size={14} aria-hidden="true" /> 상담
      </span>

      <select
        className={styles.select}
        aria-label="상담 세션 선택"
        value={activeSessionId ?? ""}
        onChange={(e) => onSelect(Number(e.target.value))}
        disabled={locked || sessions.length === 0}
      >
        {sessions.length === 0 ? (
          <option value="">상담 없음 (첫 메시지를 보내면 자동 생성)</option>
        ) : (
          sessions.map((s) => {
            const date = formatDateTime(s.created_at);
            const visit = s.visit_id != null ? visits.find((v) => v.id === s.visit_id) : undefined;
            const visitTag = visit ? ` · 방문 ${formatDate(visit.visited_at)}` : "";
            return (
              <option key={s.id} value={s.id}>
                {date ? `${s.title} · ${date}${visitTag}` : `${s.title}${visitTag}`}
              </option>
            );
          })
        )}
      </select>

      {active && <span className={styles.count}>메시지 {active.message_count}건</span>}

      <div className={styles.actions}>
        {onNewSessionVisitChange && visits.length > 0 && (
          <label className={styles.visitPick} title="새 상담을 시작할 때 연결할 방문 기록">
            방문 연결
            <select
              className={`${styles.select} ${styles.visitSelect}`}
              aria-label="새 상담을 연결할 방문"
              value={newSessionVisitId ?? ""}
              onChange={(e) => onNewSessionVisitChange(e.target.value ? Number(e.target.value) : null)}
              disabled={locked}
            >
              <option value="">없음</option>
              {visits.map((v) => (
                <option key={v.id} value={v.id}>
                  {visitLabel(v)}
                </option>
              ))}
            </select>
          </label>
        )}

        <button
          type="button"
          className={styles.shareButton}
          onClick={() => setShareOpen(true)}
          disabled={sessions.length === 0}
          title={sessions.length === 0 ? "공유할 상담이 없습니다" : "환자가 로그인 없이 설명을 보고 질문할 수 있는 링크"}
        >
          <Share2 size={14} aria-hidden="true" />
          환자 링크 공유
        </button>

        <button
          type="button"
          className={styles.newButton}
          onClick={onCreate}
          disabled={locked}
          title={newSessionVisitId != null ? "선택한 방문에 연결된 새 상담 시작" : "새 상담 시작"}
          aria-label="새 상담 시작"
        >
          {creating ? (
            <LoaderCircle size={14} className={styles.spinner} aria-hidden="true" />
          ) : (
            <MessageSquarePlus size={14} aria-hidden="true" />
          )}
          새 상담
        </button>
      </div>

      {shareOpen && (
        <ShareLinkModal
          patientId={patientId}
          sessions={sessions}
          activeSessionId={activeSessionId}
          onClose={() => setShareOpen(false)}
        />
      )}
    </div>
  );
}
