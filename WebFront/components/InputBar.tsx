"use client";

import { useState, useRef } from "react";
import styles from "./InputBar.module.css";
import { Send, Image as ImageIcon, X } from "lucide-react";

interface InputBarProps {
  onSendMessage: (text: string, file: File | null) => void;
  disabled: boolean;
  /** A reply is in progress: keep the bar visible but block another send. */
  busy?: boolean;
}

export default function InputBar({ onSendMessage, disabled, busy = false }: InputBarProps) {
  const [inputText, setInputText] = useState("");
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleSend = () => {
    if (busy) return;
    if (!inputText.trim() && !selectedImage) return;
    onSendMessage(inputText, selectedImage);

    setInputText("");
    setSelectedImage(null);
    setPreviewUrl(null);
  };

  if (disabled) return null;

  const canSend = !busy && (inputText.trim() !== "" || selectedImage !== null);

  return (
    <section className={styles.wrapper}>
      <div className={styles.container}>
        {previewUrl && (
          <div className={styles.imagePreviewWrapper}>
            <img src={previewUrl} className={styles.previewImage} alt="첨부 이미지 미리보기" />
            <button
              type="button"
              onClick={() => {
                setPreviewUrl(null);
                setSelectedImage(null);
              }}
              className={styles.removeButton}
              aria-label="첨부 이미지 제거"
              title="첨부 이미지 제거"
            >
              <X size={12} />
            </button>
          </div>
        )}

        <div className={styles.inputRow}>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className={styles.imageButton}
            aria-label="X-ray 이미지 첨부"
            title="X-ray 이미지 첨부"
          >
            <ImageIcon size={20} />
          </button>

          <input
            ref={fileInputRef}
            type="file"
            onChange={handleImageSelect}
            className="hidden"
            accept="image/*"
            aria-label="X-ray 이미지 파일"
          />

          <input
            className={styles.textInput}
            placeholder={busy ? "분석이 진행 중입니다..." : "X-ray 분석 요청..."}
            aria-label="메시지 입력"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            // isComposing: Korean IME commits on Enter; do not send the half-composed text.
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.nativeEvent.isComposing) handleSend();
            }}
          />

          <button
            type="button"
            onClick={handleSend}
            disabled={busy}
            className={`${styles.sendButton} ${canSend ? styles.sendActive : styles.sendInactive}`}
            aria-label="전송"
            title={busy ? "분석이 진행 중입니다" : "전송"}
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </section>
  );
}
