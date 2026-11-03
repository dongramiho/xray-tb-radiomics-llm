"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ChatWindow from "./ChatWindow";
import InputBar from "./InputBar";
import SessionBar from "./SessionBar";
import VisitsPanel from "./VisitsPanel";
import {
  createSession,
  createVisit,
  fetchMessages,
  fetchSessions,
  fetchTimeline,
  fetchVisits,
  sendMessage,
  streamAnalysis,
  StreamUnavailableError,
} from "@/utils/api";
import { FeedbackResult, Message, Patient, Session, Timeline, Visit, VisitInput } from "@/types";
import { formatFileSize } from "@/utils/format";

interface PatientChatProps {
  /** Mount with `key={patient.id}` so all chat state resets when the patient changes. */
  patient: Patient;
}

// Sessions + messages + sending (streaming with non-streaming fallback) for one patient.
export default function PatientChat({ patient }: PatientChatProps) {
  const patientId = patient.id;

  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<number | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loadingMessages, setLoadingMessages] = useState(true);
  const [creatingSession, setCreatingSession] = useState(false);
  const [busy, setBusy] = useState(false);
  // Visits / trend: optional context (older backends without the endpoints just leave them empty).
  const [visits, setVisits] = useState<Visit[]>([]);
  const [visitsLoading, setVisitsLoading] = useState(true);
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [timelineLoading, setTimelineLoading] = useState(true);
  // Visit the next "새 상담" is attached to (optional).
  const [newSessionVisitId, setNewSessionVisitId] = useState<number | null>(null);
  // Sequence number so a slow message load cannot overwrite a newer session selection.
  const loadSeq = useRef(0);

  const loadMessages = useCallback(
    async (sessionId: number | null) => {
      const seq = ++loadSeq.current;
      setLoadingMessages(true);
      try {
        const msgs = await fetchMessages(patientId, sessionId ?? undefined);
        if (seq === loadSeq.current) setMessages(msgs);
      } catch (err) {
        console.error("메시지 로드 실패:", err);
      } finally {
        if (seq === loadSeq.current) setLoadingMessages(false);
      }
    },
    [patientId],
  );

  const refreshSessions = useCallback(async (): Promise<Session[] | null> => {
    try {
      const list = await fetchSessions(patientId);
      setSessions(list);
      return list;
    } catch (err) {
      console.error("상담 목록 로드 실패:", err);
      return null;
    }
  }, [patientId]);

  const refreshVisits = useCallback(async () => {
    try {
      setVisits(await fetchVisits(patientId));
    } catch (err) {
      console.warn("방문 기록 로드 실패:", err);
    } finally {
      setVisitsLoading(false);
    }
  }, [patientId]);

  const refreshTimeline = useCallback(async () => {
    try {
      setTimeline(await fetchTimeline(patientId));
    } catch (err) {
      console.warn("변화 추이 로드 실패:", err);
    } finally {
      setTimelineLoading(false);
    }
  }, [patientId]);

  // 초기 로드: 세션 목록 → 최신 세션 선택 → 그 세션의 메시지
  useEffect(() => {
    let cancelled = false;
    fetchSessions(patientId)
      .then((list) => {
        if (cancelled) return;
        setSessions(list);
        const newest = list[0]?.id ?? null;
        setActiveSessionId(newest);
        return loadMessages(newest);
      })
      .catch((err) => {
        console.error("상담 목록 로드 실패:", err);
        // 세션 API 가 없는 백엔드: 전체 이력으로 동작
        if (!cancelled) return loadMessages(null);
      });
    return () => {
      cancelled = true;
    };
  }, [patientId, loadMessages]);

  // 방문 기록 / 변화 추이는 채팅과 독립적으로 로드
  useEffect(() => {
    void refreshVisits();
    void refreshTimeline();
  }, [refreshVisits, refreshTimeline]);

  const handleSelectSession = (sessionId: number) => {
    if (sessionId === activeSessionId) return;
    setActiveSessionId(sessionId);
    void loadMessages(sessionId);
  };

  const handleNewSession = async () => {
    if (creatingSession || busy) return;
    setCreatingSession(true);
    try {
      const session = await createSession(patientId, undefined, newSessionVisitId ?? undefined);
      loadSeq.current += 1; // drop any message load still in flight
      setSessions((prev) => [session, ...prev.filter((s) => s.id !== session.id)]);
      setActiveSessionId(session.id);
      setMessages([]);
      setLoadingMessages(false);
      if (newSessionVisitId != null) void refreshVisits(); // session_count changed
    } catch (err) {
      console.error("새 상담 생성 실패:", err);
      alert(err instanceof Error ? err.message : "새 상담을 만들지 못했습니다.");
    } finally {
      setCreatingSession(false);
    }
  };

  // The new visit is preselected for the next "새 상담" so the usual flow is: 방문 추가 → 새 상담 → 분석.
  const handleCreateVisit = async (input: VisitInput) => {
    const visit = await createVisit(patientId, input);
    setVisits((prev) => [visit, ...prev.filter((v) => v.id !== visit.id)]);
    setNewSessionVisitId(visit.id);
  };

  const patchMessage = (id: number, patch: (msg: Message) => Message) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? patch(m) : m)));
  };

  const handleFeedbackSaved = (messageId: number, result: FeedbackResult) => {
    patchMessage(messageId, (m) =>
      m.analysis
        ? { ...m, analysis: { ...m.analysis, doctor_feedback: result.doctor_feedback, doctor_note: result.doctor_note } }
        : m,
    );
  };

  // `mask` is set for CT uploads only (`file` = NIfTI volume, `mask` = NIfTI label map).
  const handleSend = async (text: string, file: File | null, mask?: File | null) => {
    if (busy) return;
    setBusy(true);

    const sessionId = activeSessionId ?? undefined;
    const ctUpload = file && mask ? { volume: file, mask } : null;
    const now = Date.now();
    const userMsg: Message = {
      id: now,
      role: "user",
      content: text,
      // A NIfTI volume has no browser preview: describe the upload instead of an <img>.
      image_path: file && !ctUpload ? URL.createObjectURL(file) : undefined,
      attachment_label: ctUpload
        ? `CT 볼륨 ${ctUpload.volume.name} (${formatFileSize(ctUpload.volume.size)}) · 마스크 ${ctUpload.mask.name} (${formatFileSize(ctUpload.mask.size)})`
        : undefined,
      created_at: new Date(now).toISOString(),
    };
    // Provisional assistant bubble; negative id so it can never collide with a persisted message.
    const provisionalId = -now;
    const provisional: Message = {
      id: provisionalId,
      role: "assistant",
      content: "",
      streaming: true,
      status: ctUpload ? "CT 볼륨 업로드 중..." : file ? "영상 업로드 중..." : "요청 전송 중...",
    };
    setMessages((prev) => [...prev, userMsg, provisional]);

    // Object holder (not a `let`): TS cannot see assignments made inside the stream callbacks.
    const outcome: { persisted: Message | null } = { persisted: null };

    const finish = (assistant: Message) => {
      outcome.persisted = assistant;
      setMessages((prev) =>
        prev.map((m) =>
          m.id === provisionalId
            ? { ...assistant, analysis: assistant.analysis ?? m.analysis } // keep the streamed analysis if the persisted copy lacks it
            : m,
        ),
      );
      // Auto-generated session title travels with the reply: show it now, before the list refetch lands.
      const title = assistant.meta?.title?.trim();
      const titledSessionId = assistant.session_id ?? sessionId;
      if (title && titledSessionId != null) {
        setSessions((prev) => prev.map((s) => (s.id === titledSessionId ? { ...s, title } : s)));
      }
    };

    const fail = (detail: string) => {
      patchMessage(provisionalId, (m) => ({ ...m, streaming: false, status: undefined, error: detail }));
      alert(detail);
    };

    try {
      await streamAnalysis(
        patientId,
        text,
        file,
        sessionId,
        {
          // Stage names are open-ended: show whatever message the server sent (keep the last one if empty).
          onStatus: (status) => patchMessage(provisionalId, (m) => ({ ...m, status: status.message || m.status })),
          onAnalysis: (analysis) => patchMessage(provisionalId, (m) => ({ ...m, analysis })),
          onDelta: (chunk) => patchMessage(provisionalId, (m) => ({ ...m, content: m.content + chunk })),
          onDone: finish,
          onError: fail,
        },
        mask,
      );
    } catch (err) {
      if (err instanceof StreamUnavailableError) {
        // Older backend without /analyze/stream: same request through the non-streaming endpoint.
        console.warn("스트리밍 사용 불가, 일반 요청으로 전환:", err.message);
        patchMessage(provisionalId, (m) => ({ ...m, status: ctUpload ? "CT 분석 중..." : "분석 중..." }));
        try {
          finish(await sendMessage(patientId, text, file, sessionId, mask));
        } catch (fallbackErr) {
          fail(fallbackErr instanceof Error ? fallbackErr.message : "메시지 전송에 실패했습니다.");
        }
      } else {
        fail(err instanceof Error ? err.message : "메시지 전송에 실패했습니다.");
      }
    } finally {
      setBusy(false);
    }

    // Session list changed server-side (implicit session on first send, message_count, last_message_at).
    if (outcome.persisted) {
      const list = await refreshSessions();
      if (list && sessionId == null) {
        setActiveSessionId(outcome.persisted.session_id ?? list[0]?.id ?? null);
      }
      // A new analysis shows up in the visit badges and the trend chart.
      if (outcome.persisted.analysis) {
        void refreshVisits();
        void refreshTimeline();
      }
    }
  };

  // Follow-up chip: same path as a typed message, without an attachment.
  const handleFollowUp = (text: string) => {
    void handleSend(text, null);
  };

  return (
    <>
      <ChatWindow
        messages={messages}
        activePatient={patient}
        loading={loadingMessages}
        busy={busy}
        onFeedbackSaved={handleFeedbackSaved}
        onFollowUp={handleFollowUp}
        toolbar={
          <>
            <VisitsPanel
              visits={visits}
              loading={visitsLoading}
              timeline={timeline}
              timelineLoading={timelineLoading}
              onCreateVisit={handleCreateVisit}
              onSelectSession={handleSelectSession}
              disabled={busy}
            />
            <SessionBar
              patientId={patientId}
              sessions={sessions}
              activeSessionId={activeSessionId}
              onSelect={handleSelectSession}
              onCreate={handleNewSession}
              creating={creatingSession}
              disabled={busy}
              visits={visits}
              newSessionVisitId={newSessionVisitId}
              onNewSessionVisitChange={setNewSessionVisitId}
            />
          </>
        }
      />
      <InputBar onSendMessage={handleSend} disabled={false} busy={busy} />
    </>
  );
}
