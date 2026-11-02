"use client";

import { useState, useRef } from "react";
import styles from "./InputBar.module.css";
import { Send, Image as ImageIcon, X } from "lucide-react";

interface InputBarProps {
  onSendMessage: (text: string, file: File | null) => void;
  disabled: boolean;
}

export default function InputBar({ onSendMessage, disabled }: InputBarProps) {
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
    if (!inputText.trim() && !selectedImage) return;
    onSendMessage(inputText, selectedImage);

    setInputText("");
    setSelectedImage(null);
    setPreviewUrl(null);
  };

  if (disabled) return null;

  return (
    <section className={styles.wrapper}>
      <div className={styles.container}>
        {previewUrl && (
          <div className={styles.imagePreviewWrapper}>
            <img src={previewUrl} className={styles.previewImage} alt="preview" />
            <button
              onClick={() => {
                setPreviewUrl(null);
                setSelectedImage(null);
              }}
              className={styles.removeButton}
            >
              <X size={12} />
            </button>
          </div>
        )}

        <div className={styles.inputRow}>
          <button
            onClick={() => fileInputRef.current?.click()}
            className={styles.imageButton}
          >
            <ImageIcon size={20} />
          </button>

          <input
            ref={fileInputRef}
            type="file"
            onChange={handleImageSelect}
            className="hidden"
            accept="image/*"
          />

          <input
            className={styles.textInput}
            placeholder="X-ray 분석 요청..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
          />

          <button
            onClick={handleSend}
            className={`${styles.sendButton} ${
              inputText || selectedImage ? styles.sendActive : styles.sendInactive
            }`}
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </section>
  );
}
