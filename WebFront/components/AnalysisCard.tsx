"use client";

import { useId, useState } from "react";
import { ChevronDown, ChevronRight, LoaderCircle, ThumbsDown, ThumbsUp, TriangleAlert } from "lucide-react";
import styles from "./AnalysisCard.module.css";
import FormattedText from "./FormattedText";
import { resolveMediaUrl, sendFeedback } from "@/utils/api";
import type {
  Analysis,
  CtLabelName,
  CtLabelStats,
  CtSegmentationSummary,
  FeedbackResult,
  LlmMeta,
  Modality,
  QcResult,
  Report,
  RiskLevel,
  SegmentationSummary,
} from "@/types";

interface AnalysisCardProps {
  analysis: Analysis;
  /** Clinician-facing explanation (the message `content`), rendered under the card. */
  content: string;
  /** The clinician's uploaded image (preceding user message); shown under the "원본" toggle. */
  originalImageUrl?: string;
  onFeedbackSaved?: (result: FeedbackResult) => void;
}

// Shared with the visits panel / dashboard / trend chart (same Korean labels everywhere).
export const PRIMARY_LABEL_KO: Record<string, string> = {
  tb_suspected: "결핵 의심",
  low_tb_likelihood: "결핵 가능성 낮음",
  no_lesion_detected: "병변 미검출",
  lesion_detected_unclassified: "병변 검출(미분류)",
  ct_lesion_characterized: "CT 병변 특성화",
  ct_tumor_label_missing: "종양 라벨 없음 (종양 없음 아님)",
};

export const RISK_KO: Record<RiskLevel, string> = { low: "낮음", moderate: "중간", high: "높음" };
const CONFIDENCE_KO: Record<RiskLevel, string> = { low: "낮음", moderate: "보통", high: "높음" };

export const MODALITY_KO: Record<Modality, string> = { xray: "X-ray", ct: "CT" };

const CT_LABEL_KO: Record<CtLabelName, string> = { liver: "간", tumor: "종양" };

// Known QC flags → Korean. Unknown flags fall back to the server's own description.
const QC_FLAG_KO: Record<string, string> = {
  image_too_small: "입력 영상이 너무 작음",
  low_contrast: "대비가 거의 없음",
  unusual_aspect: "정면 흉부 영상 비율이 아님",
  tiny_roi: "분할 영역이 매우 작음",
  huge_roi: "분할 영역이 비정상적으로 큼",
  low_confidence: "분할 신뢰도 낮음",
  nan_features: "일부 특징 계산 실패",
  geometry_mismatch: "마스크와 볼륨의 크기/간격 불일치",
  tumor_label_missing: "종양 라벨 없음 (종양 없음 아님)",
  liver_label_missing: "간 라벨 없음",
  coarse_slices: "슬라이스 간격이 큼",
  not_3d: "3차원 볼륨이 아님",
};

type QcVerdict = "pass" | "warn" | "fail";

const QC_VERDICT_KO: Record<QcVerdict, string> = { pass: "QC 통과", warn: "QC 주의", fail: "QC 실패" };

type ImageView = "overlay" | "original" | "mask";

type Audience = "doctor" | "patient";

export function labelKo(label: string): string {
  return PRIMARY_LABEL_KO[label] ?? label;
}

/** Records persisted before CT support carry no `modality`; they are all X-rays. */
export function modalityOf(analysis: Analysis): Modality {
  return analysis.modality ?? "xray";
}

function isCtSegmentation(seg: SegmentationSummary): seg is CtSegmentationSummary {
  return "labels_present" in seg;
}

function qcVerdict(qc: QcResult): QcVerdict {
  if (!qc.passed) return "fail";
  return (qc.flags?.length ?? 0) === 0 ? "pass" : "warn";
}

// One line per flag: Korean for known flags, otherwise the server's description (same index), then the raw key.
export function qcFlagLines(qc: QcResult): string[] {
  const flags = qc.flags ?? [];
  const descriptions = qc.flag_descriptions ?? [];
  const lines = flags.map((flag, i) => QC_FLAG_KO[flag] ?? descriptions[i] ?? flag);
  // Descriptions beyond the flag list (defensive: the server should keep both arrays aligned).
  for (let i = flags.length; i < descriptions.length; i++) lines.push(descriptions[i]);
  return lines;
}

function riskClass(level: RiskLevel | undefined): string {
  if (level === "high") return styles.riskHigh;
  if (level === "moderate") return styles.riskModerate;
  return styles.riskLow;
}

function qcClass(verdict: QcVerdict): string {
  if (verdict === "fail") return styles.riskHigh;
  if (verdict === "warn") return styles.riskModerate;
  return styles.riskLow;
}

function percent(fraction: number, digits: number): string {
  return `${(fraction * 100).toFixed(digits)}%`;
}

// 12.345 + "ml" → "12.3 ml"; missing / non-finite → "–".
function formatUnit(value: number | null | undefined, unit: string, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return "–";
  return `${value.toFixed(digits)} ${unit}`;
}

// "52.1 ± 18.3 HU" (std omitted when absent).
function formatHu(stats: CtLabelStats | undefined): string {
  if (!stats || stats.mean_hu == null || !Number.isFinite(stats.mean_hu)) return "–";
  const mean = stats.mean_hu.toFixed(1);
  const std = stats.std_hu != null && Number.isFinite(stats.std_hu) ? ` ± ${stats.std_hu.toFixed(1)}` : "";
  return `${mean}${std} HU`;
}

// 3 significant digits without scientific notation for ordinary magnitudes (12345 → "12300", 0.0012345 → "0.00123").
function formatSig(value: number | null | undefined): string {
  if (value == null) return "–";
  if (typeof value !== "number") return String(value); // tolerate older records with stringified numbers
  if (!Number.isFinite(value)) return "–";
  if (value === 0) return "0";
  return Number(value.toPrecision(3)).toString();
}

function formatCost(usd: number): string {
  return `$${Number(usd.toFixed(4))}`;
}

function formatLlmMeta(llm: LlmMeta): string {
  const seconds = (llm.latency_ms / 1000).toFixed(1);
  const tokens = (llm.input_tokens + llm.output_tokens).toLocaleString("en-US");
  return `${llm.report_model} → ${llm.summary_model} · ${seconds}s · ${tokens} tokens · ${formatCost(llm.cost_usd)}`;
}

function stripFeaturePrefix(name: string): string {
  return name.startsWith("original_") ? name.slice("original_".length) : name;
}

function ImageArea({ analysis, originalImageUrl }: { analysis: Analysis; originalImageUrl?: string }) {
  const ct = modalityOf(analysis) === "ct";
  const views: { key: ImageView; label: string; url: string; alt: string }[] = [];
  if (analysis.overlay_url) {
    views.push({
      key: "overlay",
      label: "오버레이",
      url: analysis.overlay_url,
      alt: ct ? "CT 축상 슬라이스 분할 오버레이 (종양 빨강, 간 파랑)" : "병변 분할 오버레이 영상",
    });
  }
  // The CT "original" is a NIfTI volume the browser cannot render; `mask_url` is null for CT.
  if (originalImageUrl && !ct) {
    views.push({ key: "original", label: "원본", url: originalImageUrl, alt: "업로드한 원본 X-ray 영상" });
  }
  if (analysis.mask_url) {
    views.push({ key: "mask", label: "마스크", url: analysis.mask_url, alt: "병변 분할 마스크 영상" });
  }

  const [view, setView] = useState<ImageView>("overlay");
  if (views.length === 0) return null;
  const current = views.find((v) => v.key === view) ?? views[0];

  return (
    <div className={styles.imageArea}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={resolveMediaUrl(current.url)} alt={current.alt} className={styles.image} />
      {views.length > 1 && (
        <div className={styles.viewToggle} role="group" aria-label="영상 보기 전환">
          {views.map((v) => (
            <button
              key={v.key}
              type="button"
              className={`${styles.viewButton} ${v.key === current.key ? styles.viewActive : ""}`}
              aria-pressed={v.key === current.key}
              title={`${v.label} 보기`}
              onClick={() => setView(v.key)}
            >
              {v.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function ReportSection({ report }: { report: Report }) {
  const [open, setOpen] = useState(false);
  const [jsonOpen, setJsonOpen] = useState(false);
  const bodyId = useId();
  const jsonId = useId();

  const lesion = report.lesion_summary;
  const ml = report.ml_summary;
  const recommendations = report.recommendations_for_clinician ?? [];
  const evidence = report.supporting_evidence ?? [];
  const limitations = report.limitations;

  return (
    <section className={styles.section}>
      <button
        type="button"
        className={styles.collapseButton}
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-controls={bodyId}
      >
        {open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        구조화 리포트
      </button>

      {open && (
        <div id={bodyId} className={styles.report}>
          <div className={styles.reportBlock}>
            <h5>임상 개요</h5>
            <p>{report.clinical_overview}</p>
          </div>

          {report.tb_risk_summary && (
            <div className={styles.reportBlock}>
              <h5>결핵 위험도</h5>
              <p>
                <span className={`${styles.riskBadge} ${riskClass(report.tb_risk_summary.level)}`}>
                  {RISK_KO[report.tb_risk_summary.level] ?? report.tb_risk_summary.level}
                </span>{" "}
                {report.tb_risk_summary.rationale}
              </p>
            </div>
          )}

          {lesion && (
            <div className={styles.reportBlock}>
              <h5>병변 요약</h5>
              <p>
                영역 {lesion.num_regions}개 · 구역 {lesion.zones?.length ? lesion.zones.join(", ") : "–"} · 면적 비율{" "}
                {lesion.area_fraction != null ? percent(lesion.area_fraction, 2) : "–"}
              </p>
              <p>형태: {lesion.morphology || "–"}</p>
              <p>질감: {lesion.texture || "–"}</p>
            </div>
          )}

          {ml && (
            <div className={styles.reportBlock}>
              <h5>ML 신뢰도</h5>
              <p>
                {labelKo(ml.primary_label)}
                {ml.tb_probability != null && ` · 결핵 확률 ${percent(ml.tb_probability, 1)}`} · 모델 신뢰도{" "}
                {CONFIDENCE_KO[ml.model_confidence] ?? ml.model_confidence}
              </p>
              <p>{ml.reliability_comment}</p>
            </div>
          )}

          {recommendations.length > 0 && (
            <div className={styles.reportBlock}>
              <h5>임상의 권고 사항</h5>
              <ul className={styles.reportList}>
                {recommendations.map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
            </div>
          )}

          {evidence.length > 0 && (
            <div className={styles.reportBlock}>
              <h5>근거</h5>
              <ul className={`${styles.reportList} ${styles.evidenceList}`}>
                {evidence.map((item, i) => (
                  <li key={`${item.json_key}-${i}`} className={styles.evidenceItem}>
                    <span className={styles.evidenceMeta}>
                      <span className={styles.chip}>{item.json_key}</span>
                      <span>
                        {item.category} · {item.role}
                      </span>
                    </span>
                    <p>{item.explanation}</p>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {limitations && (
            <div className={styles.reportBlock}>
              <h5>제한 사항</h5>
              {limitations.data_limitations?.length > 0 && (
                <>
                  <p className={styles.subLabel}>데이터</p>
                  <ul className={styles.reportList}>
                    {limitations.data_limitations.map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ul>
                </>
              )}
              {limitations.model_limitations?.length > 0 && (
                <>
                  <p className={styles.subLabel}>모델</p>
                  <ul className={styles.reportList}>
                    {limitations.model_limitations.map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ul>
                </>
              )}
              {limitations.usage_restrictions && (
                <>
                  <p className={styles.subLabel}>사용 제한</p>
                  <p>{limitations.usage_restrictions}</p>
                </>
              )}
            </div>
          )}

          <div className={styles.reportBlock}>
            <button
              type="button"
              className={styles.collapseButton}
              onClick={() => setJsonOpen((o) => !o)}
              aria-expanded={jsonOpen}
              aria-controls={jsonId}
            >
              {jsonOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              원본 JSON
            </button>
            {jsonOpen && (
              <pre id={jsonId} className={styles.jsonPre}>
                {JSON.stringify(report, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}
    </section>
  );
}

function FeedbackRow({
  analysis,
  onFeedbackSaved,
}: {
  analysis: Analysis;
  onFeedbackSaved?: (result: FeedbackResult) => void;
}) {
  const [feedback, setFeedback] = useState<boolean | null>(analysis.doctor_feedback);
  const [savedNote, setSavedNote] = useState<string | null>(analysis.doctor_note ?? null);
  const [noteOpen, setNoteOpen] = useState(false);
  const [noteDraft, setNoteDraft] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const noteId = useId();

  const submit = async (agree: boolean, note: string | null, afterSave: () => void) => {
    if (submitting) return;
    setSubmitting(true);
    try {
      const result = await sendFeedback(analysis.diagnosis_id, agree, note?.trim() || undefined);
      setFeedback(result.doctor_feedback);
      setSavedNote(result.doctor_note);
      onFeedbackSaved?.(result);
      afterSave();
    } catch (err) {
      alert(err instanceof Error ? err.message : "피드백 저장에 실패했습니다.");
    } finally {
      setSubmitting(false);
    }
  };

  // Choosing a verdict saves it right away; the note is an optional follow-up.
  const choose = (agree: boolean) =>
    submit(agree, savedNote, () => {
      setNoteDraft(savedNote ?? "");
      setNoteOpen(true);
    });

  const saveNote = () => {
    if (feedback === null) return;
    submit(feedback, noteDraft, () => setNoteOpen(false));
  };

  const openNoteEditor = () => {
    setNoteDraft(savedNote ?? "");
    setNoteOpen(true);
  };

  return (
    <div className={styles.feedbackRow}>
      <div className={styles.feedbackButtons}>
        <span>이 분석 결과에</span>
        <button
          type="button"
          className={`${styles.feedbackButton} ${feedback === true ? styles.agreeActive : ""}`}
          aria-pressed={feedback === true}
          title="분석 결과에 동의"
          disabled={submitting}
          onClick={() => choose(true)}
        >
          <ThumbsUp size={14} /> 동의
        </button>
        <button
          type="button"
          className={`${styles.feedbackButton} ${feedback === false ? styles.disagreeActive : ""}`}
          aria-pressed={feedback === false}
          title="분석 결과에 비동의"
          disabled={submitting}
          onClick={() => choose(false)}
        >
          <ThumbsDown size={14} /> 비동의
        </button>
        {submitting && <LoaderCircle size={14} className={styles.spinner} aria-label="저장 중" />}
        {!submitting && feedback !== null && (
          <span className={styles.savedHint}>{feedback ? "동의로 저장됨" : "비동의로 저장됨"}</span>
        )}
      </div>

      {noteOpen ? (
        <div className={styles.noteArea}>
          <label htmlFor={noteId} className={styles.subLabel}>
            메모 (선택)
          </label>
          <textarea
            id={noteId}
            className={styles.noteTextarea}
            placeholder="판독 소견이나 분석 결과에 대한 의견을 남겨주세요."
            value={noteDraft}
            onChange={(e) => setNoteDraft(e.target.value)}
            disabled={submitting}
          />
          <div className={styles.noteActions}>
            <button
              type="button"
              className={`${styles.smallButton} ${styles.smallButtonPrimary}`}
              onClick={saveNote}
              disabled={submitting}
            >
              메모 저장
            </button>
            <button type="button" className={styles.smallButton} onClick={() => setNoteOpen(false)} disabled={submitting}>
              닫기
            </button>
          </div>
        </div>
      ) : (
        feedback !== null && (
          <div className={styles.noteArea}>
            {savedNote && <p className={styles.savedNote}>메모: {savedNote}</p>}
            <div className={styles.noteActions}>
              <button type="button" className={styles.smallButton} onClick={openNoteEditor}>
                {savedNote ? "메모 수정" : "메모 추가"}
              </button>
            </div>
          </div>
        )
      )}
    </div>
  );
}

// Badge next to the risk level; with flags it is a button that toggles the flag list under the header row.
function QcBadge({ qc, open, listId, onToggle }: { qc: QcResult; open: boolean; listId: string; onToggle: () => void }) {
  const verdict = qcVerdict(qc);
  const lines = qcFlagLines(qc);
  const className = `${styles.riskBadge} ${qcClass(verdict)}`;
  if (lines.length === 0) {
    return (
      <span className={className} title={verdict === "pass" ? "품질 검사에서 문제가 발견되지 않았습니다." : QC_VERDICT_KO[verdict]}>
        {QC_VERDICT_KO[verdict]}
      </span>
    );
  }
  return (
    <button
      type="button"
      className={`${className} ${styles.qcButton}`}
      title={lines.join("\n")}
      aria-expanded={open}
      aria-controls={listId}
      onClick={onToggle}
    >
      {QC_VERDICT_KO[verdict]} · {lines.length}
    </button>
  );
}

const CT_LABEL_ORDER: CtLabelName[] = ["liver", "tumor"];

// CT counterpart of the X-ray zone chips / stats: tumor volume, extent, HU, slice counts.
function CtSegmentationBlock({ seg, overlaySlice }: { seg: CtSegmentationSummary; overlaySlice: number | null | undefined }) {
  const present = seg.labels_present ?? [];
  const tumor = seg.tumor;
  const hasTumor = present.includes("tumor");

  return (
    <div className={styles.segmentation}>
      <div className={styles.labelRow}>
        <span className={styles.labelRowTitle}>라벨 존재</span>
        <div className={styles.chips} aria-label="마스크 라벨">
          {CT_LABEL_ORDER.map((name) =>
            present.includes(name) ? (
              <span key={name} className={styles.chip}>
                {CT_LABEL_KO[name]}
              </span>
            ) : (
              <span key={name} className={`${styles.chip} ${styles.chipMuted}`}>
                {CT_LABEL_KO[name]} 라벨 없음
              </span>
            ),
          )}
        </div>
      </div>
      {!seg.detected && <p className={styles.note}>마스크에서 분할 라벨을 찾지 못했습니다.</p>}
      {seg.detected && !hasTumor && (
        <p className={styles.note}>마스크에 종양 라벨(2)이 없습니다. 종양이 없다는 뜻은 아닙니다.</p>
      )}
      <dl className={styles.stats}>
        <div>
          <dt>종양 부피</dt>
          <dd>{formatUnit(tumor?.volume_ml, "ml")}</dd>
        </div>
        <div>
          <dt>최대 길이</dt>
          <dd>{formatUnit(tumor?.max_extent_mm, "mm")}</dd>
        </div>
        <div>
          <dt>HU 평균±표준편차</dt>
          <dd>{formatHu(tumor)}</dd>
        </div>
        <div>
          <dt>라벨 포함 슬라이스</dt>
          <dd>{tumor?.slices_with_label != null ? `${tumor.slices_with_label}장` : "–"}</dd>
        </div>
        <div>
          <dt>표시 슬라이스</dt>
          <dd>{overlaySlice != null ? `#${overlaySlice}` : "–"}</dd>
        </div>
      </dl>
    </div>
  );
}

const AUDIENCE_OPTIONS: { key: Audience; label: string; title: string }[] = [
  { key: "doctor", label: "의사용", title: "임상의용 설명 보기" },
  { key: "patient", label: "환자용", title: "환자용 설명 보기" },
];

// Explanation under the card: the clinician text (`content`) and, when the pipeline produced one, a
// patient-friendly version behind a segmented toggle. Lives in the card so the choice survives streaming
// updates and the provisional → persisted message swap (same diagnosis key).
function ExplanationArea({
  content,
  patientSummary,
  patientRestricted = false,
}: {
  content: string;
  patientSummary?: string | null;
  /** QC marked the result as not usable for the patient: the 환자용 text was generated with restricted content. */
  patientRestricted?: boolean;
}) {
  const [audience, setAudience] = useState<Audience>("doctor");
  const patientText = typeof patientSummary === "string" && patientSummary.trim() !== "" ? patientSummary : null;
  if (patientText === null) return content ? <FormattedText text={content} /> : null;

  const showPatient = audience === "patient";
  const text = showPatient ? patientText : content;

  return (
    <div className={styles.explanation}>
      <div className={styles.viewToggle} role="group" aria-label="설명 대상 전환">
        {AUDIENCE_OPTIONS.map((option) => (
          <button
            key={option.key}
            type="button"
            className={`${styles.viewButton} ${option.key === audience ? styles.viewActive : ""}`}
            aria-pressed={option.key === audience}
            title={option.title}
            onClick={() => setAudience(option.key)}
          >
            {option.label}
          </button>
        ))}
      </div>
      {patientRestricted && (
        <p className={styles.qcPatientNote}>
          <TriangleAlert size={13} aria-hidden="true" /> QC 결과 환자용 설명은 제한된 내용으로 생성되었습니다.
        </p>
      )}
      {text && <FormattedText text={text} />}
      {showPatient && <p className={styles.patientNote}>환자에게 보여주기 전 담당 의사가 확인하세요.</p>}
    </div>
  );
}

export default function AnalysisCard({ analysis, content, originalImageUrl, onFeedbackSaved }: AnalysisCardProps) {
  const [qcOpen, setQcOpen] = useState(false);
  const qcListId = useId();
  const modality = modalityOf(analysis);
  const ml = analysis.ml_results;
  const seg = analysis.segmentation;
  const qc = analysis.qc ?? null;
  const qcLines = qc ? qcFlagLines(qc) : [];
  const features = ml?.top_features ?? [];
  const maxImportance = features.reduce((max, f) => Math.max(max, Math.abs(f.importance)), 0);

  return (
    <>
      <div className={styles.card}>
        <div className={styles.topRow}>
          <span
            className={`${styles.modalityBadge} ${modality === "ct" ? styles.modalityCt : styles.modalityXray}`}
            title={modality === "ct" ? "CT 볼륨 + 마스크 분석" : "흉부 X-ray 분석"}
          >
            {MODALITY_KO[modality]}
          </span>
        </div>

        <ImageArea analysis={analysis} originalImageUrl={originalImageUrl} />

        {ml && (
          <div className={styles.headerRow}>
            <span className={`${styles.riskBadge} ${riskClass(ml.risk_level)}`}>
              위험도 {RISK_KO[ml.risk_level] ?? ml.risk_level}
            </span>
            {qc && <QcBadge qc={qc} open={qcOpen} listId={qcListId} onToggle={() => setQcOpen((o) => !o)} />}
            <span className={styles.primaryLabel}>{labelKo(ml.primary_label)}</span>
            {ml.tb_probability != null && (
              <div
                className={styles.probability}
                role="meter"
                aria-label="결핵 확률"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(ml.tb_probability * 100)}
                title="결핵 확률"
              >
                <div className={styles.probTrack}>
                  <div
                    className={`${styles.probFill} ${riskClass(ml.risk_level)}`}
                    style={{ width: `${Math.min(100, Math.max(0, ml.tb_probability * 100))}%` }}
                  />
                </div>
                <span className={styles.probValue}>{percent(ml.tb_probability, 1)}</span>
              </div>
            )}
          </div>
        )}

        {qc && qcOpen && qcLines.length > 0 && (
          <ul id={qcListId} className={styles.qcList} aria-label="QC 플래그">
            {qcLines.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ul>
        )}

        {ml?.qc_warning && <p className={styles.note}>{ml.qc_warning}</p>}

        {seg && isCtSegmentation(seg) && <CtSegmentationBlock seg={seg} overlaySlice={analysis.image?.overlay_slice} />}

        {seg && !isCtSegmentation(seg) && (
          <div className={styles.segmentation}>
            {seg.zones?.length > 0 && (
              <div className={styles.chips} aria-label="병변 구역">
                {seg.zones.map((zone) => (
                  <span key={zone} className={styles.chip}>
                    {zone}
                  </span>
                ))}
              </div>
            )}
            {!seg.detected && <p className={styles.note}>분할 모델이 병변을 검출하지 못했습니다.</p>}
            <dl className={styles.stats}>
              <div>
                <dt>면적 비율</dt>
                <dd>{percent(seg.mask_area_fraction, 2)}</dd>
              </div>
              <div>
                <dt>연결 성분 수</dt>
                <dd>{seg.num_components}</dd>
              </div>
              <div>
                <dt>최대 확률</dt>
                <dd>{percent(seg.max_probability, 1)}</dd>
              </div>
            </dl>
          </div>
        )}

        {features.length > 0 && (
          <section className={styles.section}>
            <h4 className={styles.sectionTitle}>주요 기여 특징</h4>
            <table className={styles.featureTable}>
              <thead>
                <tr>
                  <th scope="col">특징</th>
                  <th scope="col">값</th>
                  <th scope="col">중요도</th>
                </tr>
              </thead>
              <tbody>
                {features.map((f) => (
                  <tr key={f.feature}>
                    <td className={styles.featureName}>{stripFeaturePrefix(f.feature)}</td>
                    <td className={styles.featureValue}>{formatSig(f.value)}</td>
                    <td>
                      <div className={styles.importanceCell}>
                        <div className={styles.importanceTrack} aria-hidden="true">
                          <div
                            className={styles.importanceFill}
                            style={{
                              width: maxImportance > 0 ? `${(Math.abs(f.importance) / maxImportance) * 100}%` : "0%",
                            }}
                          />
                        </div>
                        <span className={styles.featureValue}>{formatSig(f.importance)}</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        {analysis.report && <ReportSection report={analysis.report} />}

        <FeedbackRow analysis={analysis} onFeedbackSaved={onFeedbackSaved} />

        {analysis.llm && (
          <p className={styles.meta} title={`prompt ${analysis.llm.prompt_version}`}>
            {formatLlmMeta(analysis.llm)}
          </p>
        )}
      </div>
      <ExplanationArea
        content={content}
        patientSummary={analysis.patient_summary}
        patientRestricted={qc?.usable_for_patient === false}
      />
    </>
  );
}
