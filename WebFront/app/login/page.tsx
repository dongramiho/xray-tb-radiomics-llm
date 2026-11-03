"use client";

import { useEffect, useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { LoaderCircle, LogIn, Stethoscope } from "lucide-react";
import styles from "./login.module.css";
import { login } from "@/utils/api";
import { getToken } from "@/utils/auth";

const DEV_ACCOUNT = { email: "doctor@test.com", password: "changeme" };

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // 이미 로그인된 상태면 메인으로
  useEffect(() => {
    if (getToken()) router.replace("/");
  }, [router]);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (loading) return;

    if (!email.trim() || !password) {
      setError("이메일과 비밀번호를 입력해주세요.");
      return;
    }

    setError(null);
    setLoading(true);
    try {
      await login(email.trim(), password); // stores token + user
      router.replace("/");
    } catch (err) {
      const message =
        err instanceof TypeError
          ? "서버에 연결할 수 없습니다."
          : err instanceof Error
            ? err.message
            : "알 수 없는 오류";
      setError(`로그인 실패: ${message}`);
      setLoading(false);
    }
  };

  return (
    <div className={styles.page}>
      <form className={styles.card} onSubmit={handleSubmit} noValidate>
        <div className={styles.brand}>
          <div className={styles.brandIcon}>
            <Stethoscope size={22} />
          </div>
          <h1 className={styles.title}>BodyTalk AI</h1>
          <p className={styles.subtitle}>의료진 계정으로 로그인하세요.</p>
        </div>

        <label className={styles.label} htmlFor="login-email">
          이메일
        </label>
        <input
          id="login-email"
          type="email"
          autoComplete="email"
          autoFocus
          placeholder={DEV_ACCOUNT.email}
          className={styles.input}
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          disabled={loading}
        />

        <label className={styles.label} htmlFor="login-password">
          비밀번호
        </label>
        <input
          id="login-password"
          type="password"
          autoComplete="current-password"
          placeholder="비밀번호"
          className={styles.input}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          disabled={loading}
        />

        {error && (
          <p className={styles.error} role="alert">
            {error}
          </p>
        )}

        <button type="submit" className={styles.submitButton} disabled={loading}>
          {loading ? (
            <>
              <LoaderCircle size={18} className={styles.spinner} /> 로그인 중...
            </>
          ) : (
            <>
              <LogIn size={18} /> 로그인
            </>
          )}
        </button>

        <p className={styles.hint}>
          개발용 계정: {DEV_ACCOUNT.email} / {DEV_ACCOUNT.password}
        </p>
      </form>
    </div>
  );
}
