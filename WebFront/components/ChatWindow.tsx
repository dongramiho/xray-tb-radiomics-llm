"use client";

import { useEffect, useRef } from "react";
import styles from "./ChatWindow.module.css";
import { User } from "lucide-react";
import { Patient, Message } from "@/types";

interface ChatWindowProps {
  messages: Message[];
  activePatient: Patient;
}

export default function ChatWindow({ messages, activePatient }: ChatWindowProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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

      {/* 메시지 영역 */}
      <section className={styles.chatArea}>
        <div className={styles.centeredContainer}>
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`${styles.messageRow} ${
                msg.role === "user" ? styles.userMessage : ""
              }`}
            >
              {msg.role === "assistant" && <div className={styles.aiIcon}>AI</div>}

              <div
                className={`${styles.messageBubble} ${
                  msg.role === "user" ? styles.userBubble : styles.aiBubble
                }`}
              >
                {msg.image_path && (
                  <img
                    src={msg.image_path}
                    className={styles.messageImage}
                    alt="xray"
                  />
                )}

                <p className={styles.messageText}>{msg.content}</p>
              </div>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}