"use client";

import { useEffect, useState, type ReactNode } from "react";
import { usePathname, useRouter } from "next/navigation";
import Sidebar from "./Sidebar";
import styles from "./AuthGate.module.css";
import { fetchMe } from "@/utils/api";
import { getToken, useHasToken } from "@/utils/auth";

const LOGIN_PATH = "/login";
// Public patient share pages: no login, no sidebar.
const SHARE_PREFIX = "/share";

interface AuthGateProps {
  children: ReactNode;
}

// /login 과 /share/* 는 사이드바 없이 페이지만, 그 외는 토큰 확인이 끝난 뒤에만 사이드바 + 페이지 렌더링
export default function AuthGate({ children }: AuthGateProps) {
  const pathname = usePathname();
  const router = useRouter();
  const hasToken = useHasToken(); // null = 아직 모름 (SSR / hydration)
  const [verified, setVerified] = useState(false);

  const isLoginPage = pathname === LOGIN_PATH;
  const isPublicPage = isLoginPage || pathname === SHARE_PREFIX || pathname.startsWith(`${SHARE_PREFIX}/`);

  useEffect(() => {
    if (isPublicPage || hasToken === null) return;

    if (!hasToken) {
      router.replace(LOGIN_PATH);
      return;
    }
    if (verified) return;

    // 토큰 유효성은 세션당 한 번만 서버에 확인
    let cancelled = false;
    fetchMe()
      .then(() => {
        if (!cancelled) setVerified(true);
      })
      .catch(() => {
        // 401 → apiFetch 가 이미 토큰을 지우고 /login 으로 보냄.
        // 그 외 (서버 미응답 등) 는 통과시키고, 이후 요청이 각자 에러를 보여줌.
        if (!cancelled && getToken()) setVerified(true);
      });
    return () => {
      cancelled = true;
    };
  }, [isPublicPage, hasToken, verified, router]);

  if (isPublicPage) return <>{children}</>;

  if (!hasToken || !verified) {
    return <div className={styles.loading}>인증 확인 중...</div>;
  }

  return (
    <>
      <Sidebar />
      {children}
    </>
  );
}
