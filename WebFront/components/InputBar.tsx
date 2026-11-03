"use client";

import { useRef, useState, type ReactNode } from "react";
import styles from "./InputBar.module.css";
import { FileBox, Image as ImageIcon, Layers, Send, X } from "lucide-react";
import type { Modality } from "@/types";
import { formatFileSize, isNiftiPath } from "@/utils/format";

interface InputBarProps {
  /** X-ray: `file` is a PNG/JPEG. CT: `file` is the NIfTI volume and `mask` the NIfTI label map. */
  onSendMessage: (text: string, file: File | null, mask?: File | null) => void;
  disabled: boolean;
  /** A reply is in progress: keep the bar visible but block another send. */
  busy?: boolean;
}

const MODE_OPTIONS: { key: Modality; label: string; title: string }[] = [
  { key: "xray", label: "X-ray", title: "흉부 X-ray 영상 (PNG/JPEG)" },
  { key: "ct", label: "CT", title: "CT 볼륨 + 마스크 (NIfTI)" },
];

type CtSlot = "volume" | "mask";

const CT_SLOT_LABEL: Record<CtSlot, string> = {
  volume: "CT 볼륨 (.nii/.nii.gz)",
  mask: "마스크 (.nii/.nii.gz)",
};

const NIFTI_ACCEPT = ".nii,.nii.gz";
// Backend accepts volumes up to ~300 MB; refuse early instead of failing after a long upload.
const MAX_NIFTI_BYTES = 300 * 1024 * 1024;

// One hidden file input + a button showing the chosen file (name · size) or the slot label.
function NiftiPicker({
  slot,
  file,
  icon,
  onSelect,
  onClear,
}: {
  slot: CtSlot;
  file: File | null;
  icon: ReactNode;
  onSelect: (file: File) => void;
  onClear: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const label = CT_SLOT_LABEL[slot];

  return (
    <div className={`${styles.picker} ${file ? styles.pickerFilled : ""}`}>
      <input
        ref={inputRef}
        type="file"
        accept={NIFTI_ACCEPT}
        className="hidden"
        aria-label={label}
        onChange={(e) => {
          const picked = e.target.files?.[0];
          if (picked) onSelect(picked);
          e.target.value = ""; // allow re-picking the same file after a clear / rejection
        }}
      />
      <button
        type="button"
        className={styles.pickerButton}
        onClick={() => inputRef.current?.click()}
        title={file ? `${label}: ${file.name}` : `${label} 선택`}
      >
        {icon}
        <span className={styles.pickerText}>
          <span className={styles.pickerLabel}>{label}</span>
          {file ? (
            <span className={styles.pickerFile}>
              {file.name} · {formatFileSize(file.size)}
            </span>
          ) : (
            <span className={styles.pickerHint}>파일 선택</span>
          )}
        </span>
      </button>
      {file && (
        <button
          type="button"
          onClick={onClear}
          className={styles.pickerClear}
          aria-label={`${label} 제거`}
          title={`${label} 제거`}
        >
          <X size={12} />
        </button>
      )}
    </div>
  );
}

export default function InputBar({ onSendMessage, disabled, busy = false }: InputBarProps) {
  const [mode, setMode] = useState<Modality>("xray");
  const [inputText, setInputText] = useState("");
  // X-ray mode
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  // CT mode: both files are required together
  const [ctVolume, setCtVolume] = useState<File | null>(null);
  const [ctMask, setCtMask] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  // Switching modality drops the other mode's attachment so only one kind of upload is ever pending.
  const switchMode = (next: Modality) => {
    if (next === mode) return;
    setMode(next);
    setSelectedImage(null);
    setPreviewUrl(null);
    setCtVolume(null);
    setCtMask(null);
    setFileError(null);
  };

  const pickNifti = (slot: CtSlot, file: File) => {
    const name = slot === "volume" ? "CT 볼륨은" : "마스크는";
    if (!isNiftiPath(file.name)) {
      setFileError(`${name} .nii 또는 .nii.gz 파일만 첨부할 수 있습니다. (선택한 파일: ${file.name})`);
      return;
    }
    if (file.size > MAX_NIFTI_BYTES) {
      setFileError(`${name} 최대 약 300 MB 까지 첨부할 수 있습니다. (선택한 파일: ${formatFileSize(file.size)})`);
      return;
    }
    setFileError(null);
    if (slot === "volume") setCtVolume(file);
    else setCtMask(file);
  };

  const ctReady = ctVolume !== null && ctMask !== null;

  const handleSend = () => {
    if (busy) return;
    if (mode === "ct") {
      if (!ctVolume || !ctMask) {
        setFileError("CT 분석에는 볼륨과 마스크 파일이 모두 필요합니다.");
        return;
      }
      onSendMessage(inputText, ctVolume, ctMask);
      setInputText("");
      setCtVolume(null);
      setCtMask(null);
      setFileError(null);
      return;
    }

    if (!inputText.trim() && !selectedImage) return;
    onSendMessage(inputText, selectedImage);

    setInputText("");
    setSelectedImage(null);
    setPreviewUrl(null);
  };

  if (disabled) return null;

  const canSend = !busy && (mode === "ct" ? ctReady : inputText.trim() !== "" || selectedImage !== null);
  const placeholder = busy
    ? "분석이 진행 중입니다..."
    : mode === "ct"
      ? "CT 분석 요청 (볼륨 + 마스크 첨부 필요)..."
      : "X-ray 분석 요청...";

  return (
    <section className={styles.wrapper}>
      <div className={styles.container}>
        {mode === "xray" && previewUrl && (
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

        {mode === "ct" && (
          <div className={styles.ctPickers} role="group" aria-label="CT 파일 첨부">
            <NiftiPicker
              slot="volume"
              file={ctVolume}
              icon={<FileBox size={18} aria-hidden="true" />}
              onSelect={(file) => pickNifti("volume", file)}
              onClear={() => setCtVolume(null)}
            />
            <NiftiPicker
              slot="mask"
              file={ctMask}
              icon={<Layers size={18} aria-hidden="true" />}
              onSelect={(file) => pickNifti("mask", file)}
              onClear={() => setCtMask(null)}
            />
          </div>
        )}

        {fileError && (
          <p className={styles.fileError} role="alert">
            {fileError}
          </p>
        )}

        <div className={styles.inputRow}>
          <div className={styles.modeSwitch} role="group" aria-label="영상 종류">
            {MODE_OPTIONS.map((option) => (
              <button
                key={option.key}
                type="button"
                className={`${styles.modeButton} ${option.key === mode ? styles.modeActive : ""}`}
                aria-pressed={option.key === mode}
                title={option.title}
                onClick={() => switchMode(option.key)}
              >
                {option.label}
              </button>
            ))}
          </div>

          {mode === "xray" && (
            <>
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
            </>
          )}

          <input
            className={styles.textInput}
            placeholder={placeholder}
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
            disabled={busy || (mode === "ct" && !ctReady)}
            className={`${styles.sendButton} ${canSend ? styles.sendActive : styles.sendInactive}`}
            aria-label="전송"
            title={busy ? "분석이 진행 중입니다" : mode === "ct" && !ctReady ? "CT 볼륨과 마스크를 모두 첨부하세요" : "전송"}
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </section>
  );
}
