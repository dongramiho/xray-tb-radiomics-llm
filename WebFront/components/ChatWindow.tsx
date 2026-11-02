"use client";

import { useEffect, useRef, type ReactNode } from "react";
import styles from "./ChatWindow.module.css";
import { TriangleAlert, User } from "lucide-react";
import AnalysisCard from "./AnalysisCard";
import { resolveMediaUrl } from "@/utils/api";
import { Patient, Message, FeedbackResult } from "@/types";

interface ChatWindowProps {
  messages: Message[];
  activePatient: Patient;
  /** Slim bar rendered between the patient header and the message list (e.g. session picker). */
  toolbar?: ReactNode;
  loading?: boolean;
  onFeedbackSaved?: (messageId: number, result: FeedbackResult) => void;
}

const BULLET_RE = /^\s*[-•*]\s+/;

type TextGroup = { type: "ul" | "p"; lines: string[] };

// Plain-text formatting without a markdown library: blank lines split paragraphs,
// consecutive "- " / "• " lines become a bullet list, single line breaks are kept.
function FormattedText({ text }: { text: string }) {
  const blocks = text.split(/\n[ \t]*\n/);
  return (
    <div className={styles.contentBlocks}>
      {blocks.map((block, blockIdx) => {
        const groups: TextGroup[] = [];
        for (const line of block.split("\n")) {
          if (line.trim() === "") continue;
          const type: TextGroup["type"] = BULLET_RE.test(line) ? "ul" : "p";
          const last = groups[groups.length - 1];
          if (last && last.type === type) last.lines.push(line);
          else groups.push({ type, lines: [line] });
        }
        return groups.map((group, groupIdx) =>
          group.type === "ul" ? (
            <ul key={`${blockIdx}-${groupIdx}`} className={styles.bulletList}>
              {group.lines.map((line, lineIdx) => (
                <li key={lineIdx}>{line.replace(BULLET_RE, "")}</li>
              ))}
            </ul>
          ) : (
            <p key={`${blockIdx}-${groupIdx}`} className={styles.messageText}>
              {group.lines.join("\n")}
            </p>
          ),
        );
      })}
    </div>
  );
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

export default function ChatWindow({ messages, activePatient, toolbar, loading = false, onFeedbackSaved }: ChatWindowProps) {
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

          {messages.map((msg, idx) => (
            <div
              key={keys[idx]}
              className={`${styles.messageRow} ${msg.role === "user" ? styles.userMessage : ""}`}
            >
              {msg.role === "assistant" && <div className={styles.aiIcon}>AI</div>}

              <div className={`${styles.messageBubble} ${msg.role === "user" ? styles.userBubble : styles.aiBubble}`}>
                {msg.role === "user" ? (
                  <>
                    {msg.image_path && (
                      <img src={resolveMediaUrl(msg.image_path)} className={styles.messageImage} alt="업로드한 X-ray 영상" />
                    )}
                    {msg.content && <p className={styles.messageText}>{msg.content}</p>}
                  </>
                ) : (
                  <>
                    {msg.analysis && (
                      <AnalysisCard
                        analysis={msg.analysis}
                        originalImageUrl={precedingUserImage(messages, idx)}
                        onFeedbackSaved={onFeedbackSaved ? (result) => onFeedbackSaved(msg.id, result) : undefined}
                      />
                    )}

                    {msg.content && <FormattedText text={msg.content} />}

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
                  </>
                )}
              </div>
            </div>
          ))}
          <div ref={scrollRef} />
        </div>
      </section>
    </>
  );
}
