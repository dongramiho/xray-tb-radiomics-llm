"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { useParams } from "next/navigation";
import { CircleAlert, Clock, Hourglass, LoaderCircle, Send, Sparkles, Stethoscope, TriangleAlert } from "lucide-react";
import styles from "./share.module.css";
import FormattedText from "@/components/FormattedText";
import { askShare, fetchShare, resolveMediaUrl, ShareError } from "@/utils/api";
import { formatDateTime } from "@/utils/format";
import type { ShareMessage, ShareView } from "@/types";

type PageStatus = "loading" | "ready" | "gone" | "missing" | "error";

const MAX_FOLLOW_UPS = 3;
const QUESTION_MAX_LENGTH = 500;

// Public patient page: no login, no sidebar (AuthGate passes /share/* straight through).
export default function SharePage() {
  const params = useParams<{ token: string }>();
  const token = typeof params?.token === "string" ? params.token : "";

  const [status, setStatus] = useState<PageStatus>("loading");
  const [detail, setDetail] = useState<string | null>(null);
  const [view, setView] = useState<ShareView | null>(null);
  const [question, setQuestion] = useState("");
  const [sending, setSending] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Load the share view; 410 → expired / revoked, 404 → unknown link.
  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    fetchShare(token)
      .then((data) => {
        if (cancelled) return;
        setView(data);
        setStatus("ready");
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof ShareError && err.status === 410) {
          setStatus("gone");
          setDetail(err.message);
        } else if (err instanceof ShareError && err.status === 404) {
          setStatus("missing");
        } else {
          setStatus("error");
          setDetail(err instanceof Error ? err.message : "공유 링크를 불러오지 못했습니다.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [token]);

  useEffect(() => {
    document.title = "환자 설명 · BodyTalk AI";
  }, []);

  const messageCount = view?.messages.length ?? 0;
  useEffect(() => {
    if (messageCount > 0 || sending) bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [messageCount, sending]);

  const remaining = view?.questions_remaining ?? 0;
  const canAsk = status === "ready" && remaining > 0 && !sending;

  const handleAsk = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const text = question.trim();
    if (!text || !canAsk || !view) return;

    setSending(true);
    setAskError(null);
    setQuestion("");
    const patientMsg: ShareMessage = { role: "patient", content: text, created_at: new Date().toISOString() };
    setView((v) => (v ? { ...v, messages: [...v.messages, patientMsg] } : v));

    try {
      const result = await askShare(token, text);
      const reply: ShareMessage = {
        role: "assistant",
        content: result.message.content,
        created_at: new Date().toISOString(),
      };
      setView((v) =>
        v
          ? {
              ...v,
              messages: [...v.messages, reply],
              follow_ups: Array.isArray(result.message.meta?.follow_ups) ? result.message.meta.follow_ups : [],
              questions_remaining: result.questions_remaining,
            }
          : v,
      );
    } catch (err) {
      // Put the question back so it can be retried.
      setView((v) => (v ? { ...v, messages: v.messages.filter((m) => m !== patientMsg) } : v));
      setQuestion(text);
      if (err instanceof ShareError && err.status === 410) {
        setStatus("gone");
        setDetail(err.message);
      } else if (err instanceof ShareError && err.status === 429) {
        setAskError(err.message || "남은 질문이 없거나 요청이 너무 잦습니다. 잠시 후 다시 시도해주세요.");
      } else {
        setAskError(err instanceof Error ? err.message : "질문을 보내지 못했습니다.");
      }
    } finally {
      setSending(false);
    }
  };

  const applyFollowUp = (text: string) => {
    setQuestion(text);
    inputRef.current?.focus();
  };

  if (status === "loading") {
    return (
      <div className={styles.page}>
        <div className={styles.container}>
          <p className={styles.centered}>
            <LoaderCircle size={18} className={styles.spinner} aria-hidden="true" /> 불러오는 중...
          </p>
        </div>
      </div>
    );
  }

  if (status !== "ready" || !view) {
    const title =
      status === "gone"
        ? "이 링크는 더 이상 사용할 수 없습니다"
        : status === "missing"
          ? "링크를 찾을 수 없습니다"
          : "페이지를 불러오지 못했습니다";
    const body =
      status === "gone"
        ? "유효 기간이 지났거나 담당 의사가 공유를 종료했습니다. 다시 보시려면 담당 의사에게 새 링크를 요청하세요."
        : status === "missing"
          ? "주소가 정확한지 확인해주세요. 링크는 담당 의사가 보내 드린 그대로 열어야 합니다."
          : "잠시 후 다시 시도해주세요.";
    return (
      <div className={styles.page}>
        <div className={styles.container}>
          <Brand />
          <section className={`${styles.card} ${styles.stateCard}`} role="alert">
            {status === "gone" ? (
              <Hourglass size={28} className={styles.stateIcon} aria-hidden="true" />
            ) : (
              <CircleAlert size={28} className={styles.stateIcon} aria-hidden="true" />
            )}
            <h1 className={styles.stateTitle}>{title}</h1>
            <p className={styles.stateBody}>{body}</p>
            {status !== "missing" && detail && <p className={styles.stateDetail}>{detail}</p>}
          </section>
        </div>
      </div>
    );
  }

  const followUps = (view.follow_ups ?? []).filter((q) => typeof q === "string" && q.trim() !== "").slice(0, MAX_FOLLOW_UPS);
  const analysisDate = view.analysis?.created_at ?? view.session.created_at;

  return (
    <div className={styles.page}>
      <div className={styles.container}>
        <Brand />

        <header className={styles.header}>
          <h1 className={styles.title}>{view.patient.name} 님의 X-ray 분석 설명</h1>
          <p className={styles.subtitle}>
            담당 의사 <strong>{view.doctor.full_name}</strong>
            {analysisDate && ` · ${formatDateTime(analysisDate)}`}
            {view.session.title && ` · ${view.session.title}`}
          </p>
        </header>

        {/* ---- 분석 설명 ---- */}
        <section className={styles.card} aria-label="분석 설명">
          {view.analysis?.overlay_url && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={resolveMediaUrl(view.analysis.overlay_url)}
              alt="병변 영역이 표시된 흉부 X-ray 영상"
              className={styles.overlay}
            />
          )}
          {view.analysis?.patient_summary?.trim() ? (
            <div className={styles.summary}>
              <FormattedText text={view.analysis.patient_summary} />
            </div>
          ) : (
            <p className={styles.muted}>아직 준비된 설명이 없습니다. 담당 의사에게 문의해주세요.</p>
          )}
          <p className={styles.note}>
            이 설명은 담당 의사의 검토를 거친 참고 자료입니다. 진단과 치료는 담당 의사와 상의하세요.
          </p>
        </section>

        {/* ---- 질문과 답변 ---- */}
        <section className={styles.card} aria-label="질문과 답변">
          <h2 className={styles.sectionTitle}>궁금한 점을 물어보세요</h2>

          {view.messages.length === 0 && !sending && (
            <p className={styles.muted}>아직 주고받은 질문이 없습니다. 아래 입력창에 질문을 남겨보세요.</p>
          )}

          <div className={styles.messages}>
            {view.messages.map((m, i) => (
              <div
                key={`${m.created_at ?? ""}-${i}`}
                className={`${styles.messageRow} ${m.role === "patient" ? styles.patientRow : ""}`}
              >
                <div className={`${styles.bubble} ${m.role === "patient" ? styles.patientBubble : styles.assistantBubble}`}>
                  <span className={styles.bubbleLabel}>{m.role === "patient" ? "내 질문" : "답변"}</span>
                  {m.role === "patient" ? <p className={styles.bubbleText}>{m.content}</p> : <FormattedText text={m.content} />}
                  {m.created_at && <span className={styles.bubbleTime}>{formatDateTime(m.created_at)}</span>}
                </div>
              </div>
            ))}
            {sending && (
              <div className={styles.messageRow}>
                <div className={`${styles.bubble} ${styles.assistantBubble}`} role="status" aria-live="polite">
                  <span className={styles.pending}>
                    <span className={styles.pulseDot} aria-hidden="true" /> 답변을 준비하고 있습니다...
                  </span>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {followUps.length > 0 && remaining > 0 && (
            <div className={styles.followUps}>
              <span className={styles.followUpLabel}>
                <Sparkles size={12} aria-hidden="true" /> 이런 질문은 어떠세요
              </span>
              <div className={styles.followUpChips} role="group" aria-label="추천 질문">
                {followUps.map((q, i) => (
                  <button key={i} type="button" className={styles.followUpChip} onClick={() => applyFollowUp(q)} disabled={sending}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          <form className={styles.askForm} onSubmit={handleAsk}>
            <div className={styles.askRow}>
              <input
                ref={inputRef}
                className={styles.askInput}
                value={question}
                onChange={(e) => setQuestion(e.target.value.slice(0, QUESTION_MAX_LENGTH))}
                placeholder={remaining > 0 ? "예: 결과가 나쁘다는 뜻인가요?" : "남은 질문이 없습니다"}
                disabled={!canAsk}
                aria-label="질문 입력"
                maxLength={QUESTION_MAX_LENGTH}
                // isComposing: Korean IME commits on Enter; do not send the half-composed text.
                onKeyDown={(e) => {
                  if (e.key === "Enter" && e.nativeEvent.isComposing) e.preventDefault();
                }}
              />
              <button type="submit" className={styles.askButton} disabled={!canAsk || question.trim() === ""} aria-label="질문 보내기">
                {sending ? <LoaderCircle size={16} className={styles.spinner} aria-hidden="true" /> : <Send size={16} aria-hidden="true" />}
              </button>
            </div>
            <div className={styles.askMeta}>
              <span className={remaining === 0 ? styles.remainingZero : styles.remaining}>남은 질문 {remaining}개</span>
              {askError && (
                <span className={styles.askError} role="alert">
                  <TriangleAlert size={13} aria-hidden="true" /> {askError}
                </span>
              )}
            </div>
          </form>
        </section>

        <p className={styles.expiry}>
          <Clock size={13} aria-hidden="true" />
          {view.expires_at
            ? `이 링크는 ${formatDateTime(view.expires_at)} 까지 유효합니다`
            : "이 링크는 담당 의사가 종료할 때까지 유효합니다"}
        </p>
      </div>
    </div>
  );
}

function Brand() {
  return (
    <div className={styles.brand}>
      <span className={styles.brandIcon}>
        <Stethoscope size={16} aria-hidden="true" />
      </span>
      BodyTalk AI
    </div>
  );
}
