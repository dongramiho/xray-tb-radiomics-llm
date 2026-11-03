import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import styles from "./layout.module.css";
import AuthGate from "@/components/AuthGate";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "BodyTalk AI",
  description: "Doctor Assistant Chatbot",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${styles.body}`}
      >
        {/* 전체 레이아웃을 잡는 컨테이너 */}
        <div className={styles.layoutContainer}>

          {/* AuthGate: 토큰 확인 후 사이드바 + 페이지 렌더링. /login 은 사이드바 없이 페이지만 */}
          <AuthGate>
            {/* 페이지 내용(children)만 바뀜 */}
            <main className={styles.mainContent}>
              {children}
            </main>
          </AuthGate>

        </div>
      </body>
    </html>
  );
}
