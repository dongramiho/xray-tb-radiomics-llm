"use client";

import { useState } from "react";
import styles from "./PatientModal.module.css";

interface PatientModalProps {
  onClose: () => void;
  onSubmit: (formData: { name: string; birthDate: string; gender: string }) => void;
}

export default function PatientModal({ onClose, onSubmit }: PatientModalProps) {
  const [form, setForm] = useState({ name: "", birthDate: "", gender: "M" });

  const handleSubmit = () => {
    if (!form.name || !form.birthDate) {
      alert("이름과 생년월일을 입력해주세요.");
      return;
    }
    onSubmit(form);
  };

  return (
    <div className={styles.overlay}>
      <div className={styles.modal}>
        <h2 className={styles.title}>환자 등록</h2>

        <div className={styles.formGroup}>
          {/* 이름 */}
          <div>
            <label className={styles.label}>이름</label>
            <input
              className={styles.input}
              placeholder="예: 홍길동"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
            />
          </div>

          {/* 생년월일 */}
          <div>
            <label className={styles.label}>생년월일</label>
            <input
              type="date"
              className={styles.input}
              value={form.birthDate}
              onChange={(e) => setForm({ ...form, birthDate: e.target.value })}
            />
          </div>

          {/* 성별 */}
          <div>
            <label className={styles.label}>성별</label>
            <div className={styles.radioRow}>
              <label>
                <input
                  type="radio"
                  name="gender"
                  checked={form.gender === "M"}
                  onChange={() => setForm({ ...form, gender: "M" })}
                />{" "}
                남성
              </label>

              <label>
                <input
                  type="radio"
                  name="gender"
                  checked={form.gender === "F"}
                  onChange={() => setForm({ ...form, gender: "F" })}
                />{" "}
                여성
              </label>
            </div>
          </div>
        </div>

        <div className={styles.footer}>
          <button className={styles.cancelButton} onClick={onClose}>
            취소
          </button>
          <button className={styles.confirmButton} onClick={handleSubmit}>
            등록
          </button>
        </div>
      </div>
    </div>
  );
}
