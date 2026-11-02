import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import styles from "./layout.module.css";
import Sidebar from "@/components/Sidebar"; // 👈 추가

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
        {/* 전체 레이아웃을 잡는 컨테이너 추가 */}
        <div className={styles.layoutContainer}>
          
          {/* 사이드바는 여기서 한 번 로드되고 고정됨 */}
          <Sidebar /> 
          
          {/* 페이지 내용(children)만 바뀜 */}
          <main className={styles.mainContent}>
            {children}
          </main>
          
        </div>
      </body>
    </html>
  );
}