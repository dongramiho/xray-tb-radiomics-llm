"use client";

import { useEffect, useId, useRef, useState } from "react";
import { Check, ChevronDown, ChevronRight, Copy, Link2, LoaderCircle, X } from "lucide-react";
import styles from "./ShareLinkModal.module.css";
import { createLink, fetchLinkAccesses, fetchLinks, revokeLink } from "@/utils/api";
import { formatDateTime } from "@/utils/format";
import type {
  CreateLinkInput,
  LinkAccess,
  LinkAccessKind,
  LinkExpirePolicy,
  LinkStatus,
  PatientLink,
  Session,
} from "@/types";

interface ShareLinkModalProps {
  patientId: number;
  sessions: Session[];
  /** Session preselected for a new link (the one open in the chat). */
  activeSessionId: number | null;
  onClose: () => void;
}

const POLICY_OPTIONS: { key: LinkExpirePolicy; label: string; hint: string }[] = [
  { key: "time", label: "시간 만료", hint: "설정한 시간이 지나면 자동으로 닫힙니다." },
  { key: "manual", label: "수동 종료", hint: "담당 의사가 종료할 때까지 유효합니다." },
  { key: "both", label: "둘 다", hint: "시간이 지나거나 의사가 종료하면 닫힙니다." },
];

const HOUR_OPTIONS: { value: number; label: string }[] = [
  { value: 24, label: "24시간" },
  { value: 72, label: "72시간 (3일)" },
  { value: 168, label: "168시간 (7일)" },
];

const DEFAULT_MAX_QUESTIONS = 5;

const STATUS_KO: Record<LinkStatus, string> = { active: "활성", expired: "만료", revoked: "종료" };
const ACCESS_KIND_KO: Record<LinkAccessKind, string> = { view: "열람", question: "질문", denied: "거부" };

type AccessState = { loading: boolean; items: LinkAccess[]; error: string | null };

// The share page lives in this frontend, so a relative / missing url falls back to our own origin.
function shareUrlOf(link: PatientLink): string {
  if (/^https?:\/\//i.test(link.url ?? "")) return link.url;
  if (typeof window === "undefined") return link.url ?? "";
  return `${window.location.origin}/share/${link.token}`;
}

function statusClass(status: LinkStatus): string {
  if (status === "active") return styles.statusActive;
  if (status === "expired") return styles.statusExpired;
  return styles.statusRevoked;
}

function expiryText(link: PatientLink): string {
  return link.expires_at ? `${formatDateTime(link.expires_at)} 까지` : "수동 종료";
}

function questionsText(link: PatientLink): string {
  return link.max_questions != null
    ? `질문 ${link.questions_used}/${link.max_questions}`
    : `질문 ${link.questions_used}회 (제한 없음)`;
}

// navigator.clipboard first; otherwise select the text so the user can copy it by hand.
async function copyToClipboard(text: string, input: HTMLInputElement | null): Promise<boolean> {
  try {
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* permission denied / insecure context: fall through to the selection fallback */
  }
  if (!input) return false;
  input.focus();
  input.select();
  try {
    return document.execCommand("copy");
  } catch {
    return false;
  }
}

export default function ShareLinkModal({ patientId, sessions, activeSessionId, onClose }: ShareLinkModalProps) {
  const [sessionId, setSessionId] = useState<number | null>(activeSessionId ?? sessions[0]?.id ?? null);
  const [policy, setPolicy] = useState<LinkExpirePolicy>("time");
  const [hours, setHours] = useState<number>(HOUR_OPTIONS[0].value);
  const [maxQuestions, setMaxQuestions] = useState<string>(String(DEFAULT_MAX_QUESTIONS));
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [created, setCreated] = useState<PatientLink | null>(null);
  const [copiedId, setCopiedId] = useState<number | null>(null);

  const [links, setLinks] = useState<PatientLink[]>([]);
  const [linksLoading, setLinksLoading] = useState(true);
  const [linksError, setLinksError] = useState<string | null>(null);
  const [revokingId, setRevokingId] = useState<number | null>(null);
  const [openAccess, setOpenAccess] = useState<Record<number, boolean>>({});
  const [accesses, setAccesses] = useState<Record<number, AccessState>>({});

  const urlInputRef = useRef<HTMLInputElement>(null);
  const baseId = useId();

  // Existing links for this patient.
  useEffect(() => {
    let cancelled = false;
    fetchLinks(patientId)
      .then((list) => {
        if (!cancelled) setLinks(list);
      })
      .catch((err: unknown) => {
        if (!cancelled) setLinksError(err instanceof Error ? err.message : "공유 링크 목록을 불러오지 못했습니다.");
      })
      .finally(() => {
        if (!cancelled) setLinksLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [patientId]);

  // Escape closes the dialog.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // "복사됨" feedback resets after 2s.
  useEffect(() => {
    if (copiedId === null) return;
    const timer = setTimeout(() => setCopiedId(null), 2000);
    return () => clearTimeout(timer);
  }, [copiedId]);

  const handleCreate = async () => {
    if (creating || sessionId == null) return;
    setCreating(true);
    setCreateError(null);
    try {
      const input: CreateLinkInput = { session_id: sessionId, expire_policy: policy };
      if (policy !== "manual") input.hours = hours;
      const max = Number.parseInt(maxQuestions, 10);
      if (Number.isFinite(max) && max > 0) input.max_questions = max;
      const link = await createLink(patientId, input);
      setCreated(link);
      setLinks((prev) => [link, ...prev.filter((l) => l.id !== link.id)]);
    } catch (err) {
      setCreateError(err instanceof Error ? err.message : "공유 링크를 만들지 못했습니다.");
    } finally {
      setCreating(false);
    }
  };

  const handleCopy = async (link: PatientLink) => {
    const url = shareUrlOf(link);
    const ok = await copyToClipboard(url, created?.id === link.id ? urlInputRef.current : null);
    if (ok) setCopiedId(link.id);
    else window.prompt("자동 복사에 실패했습니다. 아래 링크를 직접 복사하세요.", url);
  };

  const handleRevoke = async (link: PatientLink) => {
    if (revokingId !== null) return;
    if (!window.confirm("이 링크를 종료할까요? 환자는 더 이상 접속할 수 없습니다.")) return;
    setRevokingId(link.id);
    try {
      const revoked = await revokeLink(link.id);
      const merge = (l: PatientLink): PatientLink => ({ ...l, ...revoked, status: revoked.status ?? "revoked" });
      setLinks((prev) => prev.map((l) => (l.id === link.id ? merge(l) : l)));
      setCreated((prev) => (prev && prev.id === link.id ? merge(prev) : prev));
    } catch (err) {
      alert(err instanceof Error ? err.message : "공유 링크를 종료하지 못했습니다.");
    } finally {
      setRevokingId(null);
    }
  };

  // Access log is fetched every time the row is expanded so new visits show up.
  const toggleAccesses = (linkId: number) => {
    const willOpen = !openAccess[linkId];
    setOpenAccess((prev) => ({ ...prev, [linkId]: willOpen }));
    if (!willOpen) return;
    setAccesses((prev) => ({ ...prev, [linkId]: { loading: true, items: prev[linkId]?.items ?? [], error: null } }));
    fetchLinkAccesses(linkId)
      .then((items) => setAccesses((prev) => ({ ...prev, [linkId]: { loading: false, items, error: null } })))
      .catch((err: unknown) =>
        setAccesses((prev) => ({
          ...prev,
          [linkId]: {
            loading: false,
            items: [],
            error: err instanceof Error ? err.message : "접속 기록을 불러오지 못했습니다.",
          },
        })),
      );
  };

  const policyHint = POLICY_OPTIONS.find((o) => o.key === policy)?.hint;

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div
        className={styles.modal}
        role="dialog"
        aria-modal="true"
        aria-labelledby={`${baseId}-title`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className={styles.titleRow}>
          <h2 id={`${baseId}-title`} className={styles.title}>
            <Link2 size={16} aria-hidden="true" /> 환자 링크 공유
          </h2>
          <button type="button" className={styles.iconButton} onClick={onClose} aria-label="닫기" title="닫기">
            <X size={16} />
          </button>
        </div>
        <p className={styles.description}>
          환자가 로그인 없이 환자용 설명을 읽고 질문할 수 있는 링크를 만듭니다. 질문에 대한 답변은 환자용 설명 범위 안에서만
          제공됩니다.
        </p>

        {/* ---- 새 링크 ---- */}
        <section className={styles.section} aria-label="새 링크 만들기">
          <div className={styles.field}>
            <label className={styles.label} htmlFor={`${baseId}-session`}>
              공유할 상담
            </label>
            <select
              id={`${baseId}-session`}
              className={styles.select}
              value={sessionId ?? ""}
              onChange={(e) => setSessionId(e.target.value ? Number(e.target.value) : null)}
              disabled={creating || sessions.length === 0}
            >
              {sessions.length === 0 && <option value="">공유할 상담이 없습니다</option>}
              {sessions.map((s) => {
                const date = formatDateTime(s.created_at);
                return (
                  <option key={s.id} value={s.id}>
                    {date ? `${s.title} · ${date}` : s.title}
                  </option>
                );
              })}
            </select>
          </div>

          <fieldset className={styles.fieldset}>
            <legend className={styles.label}>만료 정책</legend>
            <div className={styles.radioRow} role="radiogroup" aria-label="만료 정책">
              {POLICY_OPTIONS.map((opt) => (
                <label
                  key={opt.key}
                  className={`${styles.radio} ${policy === opt.key ? styles.radioActive : ""}`}
                  title={opt.hint}
                >
                  <input
                    type="radio"
                    name={`${baseId}-policy`}
                    value={opt.key}
                    checked={policy === opt.key}
                    onChange={() => setPolicy(opt.key)}
                    disabled={creating}
                  />
                  {opt.label}
                </label>
              ))}
            </div>
            {policyHint && <p className={styles.hint}>{policyHint}</p>}
          </fieldset>

          <div className={styles.inlineFields}>
            {policy !== "manual" && (
              <div className={styles.field}>
                <label className={styles.label} htmlFor={`${baseId}-hours`}>
                  유효 시간
                </label>
                <select
                  id={`${baseId}-hours`}
                  className={styles.select}
                  value={hours}
                  onChange={(e) => setHours(Number(e.target.value))}
                  disabled={creating}
                >
                  {HOUR_OPTIONS.map((h) => (
                    <option key={h.value} value={h.value}>
                      {h.label}
                    </option>
                  ))}
                </select>
              </div>
            )}
            <div className={styles.field}>
              <label className={styles.label} htmlFor={`${baseId}-max`}>
                최대 질문 수
              </label>
              <input
                id={`${baseId}-max`}
                type="number"
                min={0}
                max={100}
                step={1}
                className={styles.input}
                value={maxQuestions}
                onChange={(e) => setMaxQuestions(e.target.value)}
                disabled={creating}
              />
              <span className={styles.hint}>0 이면 제한 없음</span>
            </div>
            <button
              type="button"
              className={styles.primaryButton}
              onClick={handleCreate}
              disabled={creating || sessionId == null}
            >
              {creating ? (
                <LoaderCircle size={14} className={styles.spinner} aria-hidden="true" />
              ) : (
                <Link2 size={14} aria-hidden="true" />
              )}
              링크 만들기
            </button>
          </div>

          {createError && (
            <p className={styles.error} role="alert">
              {createError}
            </p>
          )}

          {created && (
            <div className={styles.createdBox}>
              <span className={styles.createdLabel}>
                <Check size={14} aria-hidden="true" /> 링크가 만들어졌습니다. 환자에게 전달하세요.
              </span>
              <div className={styles.urlRow}>
                <input
                  ref={urlInputRef}
                  className={styles.urlInput}
                  readOnly
                  value={shareUrlOf(created)}
                  onFocus={(e) => e.currentTarget.select()}
                  aria-label="공유 링크 URL"
                />
                <button type="button" className={styles.copyButton} onClick={() => handleCopy(created)}>
                  {copiedId === created.id ? (
                    <>
                      <Check size={14} aria-hidden="true" /> 복사됨
                    </>
                  ) : (
                    <>
                      <Copy size={14} aria-hidden="true" /> 복사
                    </>
                  )}
                </button>
              </div>
              <p className={styles.hint}>
                {expiryText(created)} · {questionsText(created)}
              </p>
            </div>
          )}
        </section>

        {/* ---- 기존 링크 ---- */}
        <section className={styles.section} aria-label="이 환자의 링크">
          <h3 className={styles.subtitle}>이 환자의 링크{links.length > 0 && ` (${links.length})`}</h3>
          {linksLoading && <p className={styles.muted}>불러오는 중...</p>}
          {linksError && <p className={styles.error}>{linksError}</p>}
          {!linksLoading && !linksError && links.length === 0 && (
            <p className={styles.muted}>아직 만든 링크가 없습니다.</p>
          )}

          <ul className={styles.linkList}>
            {links.map((link) => {
              const session = sessions.find((s) => s.id === link.session_id);
              const access = accesses[link.id];
              const isOpen = openAccess[link.id] === true;
              return (
                <li key={link.id} className={styles.linkItem}>
                  <div className={styles.linkHead}>
                    <span className={`${styles.statusBadge} ${statusClass(link.status)}`}>
                      {STATUS_KO[link.status] ?? link.status}
                    </span>
                    <span className={styles.linkTitle}>{session?.title ?? `상담 #${link.session_id}`}</span>
                    <span className={styles.linkMeta}>생성 {formatDateTime(link.created_at)}</span>
                  </div>
                  <div className={styles.linkMetaRow}>
                    <span>{expiryText(link)}</span>
                    <span>{questionsText(link)}</span>
                    {link.last_accessed_at && <span>마지막 접속 {formatDateTime(link.last_accessed_at)}</span>}
                  </div>
                  <div className={styles.linkActions}>
                    <button type="button" className={styles.smallButton} onClick={() => handleCopy(link)}>
                      {copiedId === link.id ? (
                        <>
                          <Check size={12} aria-hidden="true" /> 복사됨
                        </>
                      ) : (
                        <>
                          <Copy size={12} aria-hidden="true" /> 복사
                        </>
                      )}
                    </button>
                    {link.status === "active" && (
                      <button
                        type="button"
                        className={`${styles.smallButton} ${styles.dangerButton}`}
                        onClick={() => handleRevoke(link)}
                        disabled={revokingId === link.id}
                      >
                        {revokingId === link.id ? (
                          <LoaderCircle size={12} className={styles.spinner} aria-hidden="true" />
                        ) : (
                          <X size={12} aria-hidden="true" />
                        )}
                        종료
                      </button>
                    )}
                    <button
                      type="button"
                      className={styles.smallButton}
                      onClick={() => toggleAccesses(link.id)}
                      aria-expanded={isOpen}
                      aria-controls={`${baseId}-access-${link.id}`}
                    >
                      {isOpen ? (
                        <ChevronDown size={12} aria-hidden="true" />
                      ) : (
                        <ChevronRight size={12} aria-hidden="true" />
                      )}
                      접속 기록
                    </button>
                  </div>

                  {isOpen && (
                    <div id={`${baseId}-access-${link.id}`} className={styles.accessBox}>
                      {access?.loading && <p className={styles.muted}>불러오는 중...</p>}
                      {access?.error && <p className={styles.error}>{access.error}</p>}
                      {access && !access.loading && !access.error && access.items.length === 0 && (
                        <p className={styles.muted}>아직 접속 기록이 없습니다.</p>
                      )}
                      {access && access.items.length > 0 && (
                        <ul className={styles.accessList}>
                          {access.items.map((a, i) => (
                            <li key={`${a.accessed_at}-${i}`} className={styles.accessItem}>
                              <span className={styles.accessTime}>{formatDateTime(a.accessed_at)}</span>
                              <span className={`${styles.accessKind} ${a.kind === "denied" ? styles.accessDenied : ""}`}>
                                {ACCESS_KIND_KO[a.kind] ?? a.kind}
                              </span>
                              {a.question && <span className={styles.accessQuestion}>{a.question}</span>}
                              {a.client_ip && <span className={styles.accessIp}>{a.client_ip}</span>}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </section>
      </div>
    </div>
  );
}
