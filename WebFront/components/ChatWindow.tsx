"use client";

import { useEffect, useRef, type ReactNode } from "react";
import styles from "./ChatWindow.module.css";
import { ShieldAlert, Sparkles, TriangleAlert, User } from "lucide-react";
import AnalysisCard from "./AnalysisCard";
import FormattedText from "./FormattedText";
import { resolveMediaUrl } from "@/utils/api";
import { Patient, Message, FeedbackResult } from "@/types";

interface ChatWindowProps {
  messages: Message[];
  activePatient: Patient;
  /** Slim bar rendered between the patient header and the message list (e.g. session picker). */
  toolbar?: ReactNode;
  loading?: boolean;
  /** A send is in progress: follow-up chips are locked so a second request cannot start. */
  busy?: boolean;
  onFeedbackSaved?: (messageId: number, result: FeedbackResult) => void;
  /** A follow-up chip was clicked; the text goes through the same send path as a typed message. */
  onFollowUp?: (text: string) => void;
}

const MAX_FOLLOW_UPS = 3;

// Suggested next questions from the pipeline; tolerates missing / empty entries.
function followUpsOf(msg: Message): string[] {
  const raw = msg.meta?.follow_ups;
  if (!Array.isArray(raw)) return [];
  return raw.filter((q) => typeof q === "string" && q.trim() !== "").slice(0, MAX_FOLLOW_UPS);
}

// The clinician's upload that this reply analysed: the closest preceding user message's image.
function precedingUserImage(messages: Message[], index: number): string | undefined {
  for (let i = index - 1; i >= 0; i--) {
    if (messages[i].role === "user") return messages[i].image_path ?? undefined;
  }
  return undefined;
}

// Keyed by diagnosis so the card keeps its toggles when the provisional streaming
// message is swapped for the persisted one at the end of the stream.
// A repeated key (same diagnosis on two replies) falls back to the message index.
function messageKeys(messages: Message[]): string[] {
  const seen = new Set<string>();
  return messages.map((msg, idx) => {
    const base = msg.analysis ? `d${msg.analysis.diagnosis_id}` : `m${msg.id}`;
    const key = seen.has(base) ? `${base}-${idx}` : base;
    seen.add(key);
    return key;
  });
}

export default function ChatWindow({
  messages,
  activePatient,
  toolbar,
  loading = false,
  busy = false,
  onFeedbackSaved,
  onFollowUp,
}: ChatWindowProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const keys = messageKeys(messages);

  return (
    <>
      {/* 헤더 */}
      <header className={styles.header}>
        <div className={styles.avatar}>
          <User size={18} />
        </div>
        <div>
          <h2 className={styles.name}>{activePatient.name}</h2>
          <p className={styles.lastVisit}>최근 방문: {activePatient.lastVisit}</p>
        </div>
      </header>

      {toolbar}

      {/* 메시지 영역 */}
      <section className={styles.chatArea}>
        <div className={styles.centeredContainer}>
          {loading && <p className={styles.emptyState}>대화를 불러오는 중...</p>}
          {!loading && messages.length === 0 && (
            <p className={styles.emptyState}>아직 대화가 없습니다. X-ray 영상을 첨부하거나 질문을 입력해주세요.</p>
          )}

          {messages.map((msg, idx) => {
            const blocked = msg.role === "assistant" && msg.meta?.blocked === true;
            const followUps = msg.role === "assistant" ? followUpsOf(msg) : [];

            return (
              <div
                key={keys[idx]}
                className={`${styles.messageRow} ${msg.role === "user" ? styles.userMessage : ""}`}
              >
                {msg.role === "assistant" && <div className={styles.aiIcon}>AI</div>}

                <div
                  className={`${styles.messageBubble} ${msg.role === "user" ? styles.userBubble : styles.aiBubble} ${
                    blocked ? styles.blockedBubble : ""
                  }`}
                >
                  {msg.role === "user" ? (
                    <>
                      {msg.image_path && (
                        <img src={resolveMediaUrl(msg.image_path)} className={styles.messageImage} alt="업로드한 X-ray 영상" />
                      )}
                      {msg.content && <p className={styles.messageText}>{msg.content}</p>}
                    </>
                  ) : (
                    <>
                      {blocked && (
                        <span className={styles.blockedLabel} title={msg.meta?.block_reason || undefined}>
                          <ShieldAlert size={13} aria-hidden="true" /> 범위 외 질문
                        </span>
                      )}

                      {/* With an analysis the card owns the explanation text (의사용 / 환자용 toggle). */}
                      {msg.analysis ? (
                        <AnalysisCard
                          analysis={msg.analysis}
                          content={msg.content}
                          originalImageUrl={precedingUserImage(messages, idx)}
                          onFeedbackSaved={onFeedbackSaved ? (result) => onFeedbackSaved(msg.id, result) : undefined}
                        />
                      ) : (
                        msg.content && <FormattedText text={msg.content} />
                      )}

                      {msg.error && (
                        <p className={styles.errorText} role="alert">
                          <TriangleAlert size={14} aria-hidden="true" /> {msg.error}
                        </p>
                      )}

                      {msg.streaming && (
                        <div className={styles.streamingRow} role="status" aria-live="polite">
                          <span className={styles.pulseDot} aria-hidden="true" />
                          <span>{msg.status ?? "응답 생성 중..."}</span>
                        </div>
                      )}

                      {followUps.length > 0 && onFollowUp && (
                        <div className={styles.followUps}>
                          <span className={styles.followUpLabel}>
                            <Sparkles size={12} aria-hidden="true" /> 다음 질문 제안
                          </span>
                          <div className={styles.followUpChips} role="group" aria-label="다음 질문 제안">
                            {followUps.map((question, i) => (
                              <button
                                key={i}
                                type="button"
                                className={styles.followUpChip}
                                disabled={busy}
                                onClick={() => onFollowUp(question)}
                              >
                                {question}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            );
          })}
          <div ref={scrollRef} />
        </div>
      </section>
    </>
  );
}
