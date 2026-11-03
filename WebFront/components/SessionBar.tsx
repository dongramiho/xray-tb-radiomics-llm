"use client";

import { LoaderCircle, MessageSquarePlus, MessagesSquare } from "lucide-react";
import styles from "./SessionBar.module.css";
import { Session } from "@/types";

interface SessionBarProps {
  sessions: Session[];
  activeSessionId: number | null;
  onSelect: (sessionId: number) => void;
  onCreate: () => void;
  creating?: boolean;
  /** Locks the controls while a reply is streaming. */
  disabled?: boolean;
}

function formatSessionDate(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

export default function SessionBar({
  sessions,
  activeSessionId,
  onSelect,
  onCreate,
  creating = false,
  disabled = false,
}: SessionBarProps) {
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
            const date = formatSessionDate(s.created_at);
            return (
              <option key={s.id} value={s.id}>
                {date ? `${s.title} · ${date}` : s.title}
              </option>
            );
          })
        )}
      </select>

      {active && <span className={styles.count}>메시지 {active.message_count}건</span>}

      <button
        type="button"
        className={styles.newButton}
        onClick={onCreate}
        disabled={locked}
        title="새 상담 시작"
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
  );
}
